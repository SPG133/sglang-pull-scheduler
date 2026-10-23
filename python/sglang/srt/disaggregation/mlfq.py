"""Experimental 1P-nD MLFQ: FCFS prefill, host checkpoints, decoder pull.

The baseline deliberately uses one running request per D. No affinity, matching,
recomputation or decode migration shortcuts: every expired slice returns to P.
"""

import io
import logging
import math
import queue
import threading
import time
from array import array
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Job:
    state: dict
    peak_tokens: int
    reservation: int
    kv: bytes = b""
    level: int = 0
    ready_at: float | None = None
    lease: int = 0
    last_seen: float = 0


class MLFQPool:
    """P-owned state. Serialize admission/claims/returns under one small lock."""

    def __init__(self, quanta_ms, aging_ms, host_bytes, layout, timeout_s=120):
        values = [*quanta_ms, aging_ms, host_bytes, timeout_s]
        if len(quanta_ms) != 3 or any(not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError("MLFQ needs three positive quanta and positive limits")
        if list(quanta_ms) != sorted(quanta_ms):
            raise ValueError("MLFQ quanta must be nondecreasing")
        self.quanta_ms = tuple(quanta_ms)
        self.aging_s = aging_ms / 1000
        self.host_bytes = host_bytes
        self.layout = layout
        self.timeout_s = timeout_s
        self.jobs: dict[str, Job] = {}
        self.ready = [deque(), deque(), deque()]
        self.reserved = 0
        self.sequence = 0
        self.events = queue.SimpleQueue()
        self.lock = threading.Lock()

    def reserve(self, state, peak_tokens, reservation):
        """Reserve peak checkpoint space BEFORE prefill; False applies backpressure."""
        with self.lock:
            rid = state["rid"]
            if rid in self.jobs or reservation <= 0:
                raise ValueError("Duplicate request or invalid reservation")
            if self.reserved + reservation > self.host_bytes:
                return False
            self.jobs[rid] = Job(state, peak_tokens, reservation)
            self.reserved += reservation
            return True

    def publish(self, rid, output_ids, kv, now=None):
        """Publish a completed prefill only once its CPU checkpoint exists."""
        with self.lock:
            job = self.jobs.get(rid)
            if job is None:  # Cancelled during prefill.
                return
            if job.ready_at is not None or job.lease or job.kv:
                raise ValueError("Request was already published")
            self._store(rid, job, output_ids, kv, now)

    def _store(self, rid, job, output_ids, kv, now):
        if len(kv) > job.reservation:
            raise ValueError("Checkpoint exceeds its reserved host budget")
        job.state["output_ids"] = list(output_ids)
        job.kv = kv
        job.lease = 0
        job.ready_at = time.monotonic() if now is None else now
        self.ready[job.level].append(rid)

    def cancel(self, rid, error=None):
        with self.lock:
            self._remove(rid, error)

    def _remove(self, rid, error=None):
        job = self.jobs.pop(rid, None)
        if job is None:
            return
        if job.ready_at is not None:
            self.ready[job.level].remove(rid)
        self.reserved -= job.reservation
        if error:
            self.events.put((rid, None, error))

    def tick(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            self._expire(now)
            self._age(now)

    def _expire(self, now):
        for rid, job in list(self.jobs.items()):
            if job.lease and now - job.last_seen > self.timeout_s:
                self._remove(rid, "MLFQ decode worker timed out")

    def dispatch(self, message, now=None):
        """Two messages: pull a slice, or report its progress/return/completion."""
        now = time.monotonic() if now is None else now
        with self.lock:
            self._expire(now)
            if message["layout"] != self.layout:
                return {"error": "MLFQ model/KV layout differs between P and D"}
            if message["op"] == "pull":
                self._age(now)
                for pending in self.ready:
                    for rid in pending:
                        job = self.jobs[rid]
                        if job.peak_tokens > message["capacity"]:
                            continue
                        pending.remove(rid)
                        self.sequence += 1
                        job.lease = self.sequence
                        job.ready_at = None
                        job.last_seen = now
                        return {
                            "rid": rid,
                            "lease": job.lease,
                            "state": job.state,
                            "kv": job.kv,
                            "level": job.level,
                            "quantum_ms": self.quanta_ms[job.level],
                        }
                return {}
            rid = message["rid"]
            job = self.jobs.get(rid)
            if job is None or not job.lease or job.lease != message["lease"]:
                return {"cancelled": True}
            if message["op"] != "update":
                raise ValueError("Expected pull or update")
            kv, finish = message.get("kv"), message.get("finish")
            if kv is not None:
                elapsed = message["service_ms"]
                if not math.isfinite(elapsed) or elapsed < self.quanta_ms[job.level]:
                    raise ValueError("Only an exhausted slice can be returned")
                if len(kv) > job.reservation:
                    raise ValueError("Returned checkpoint exceeds host reservation")
                if finish is not None:
                    raise ValueError("A finished request cannot be requeued")
            job.last_seen = now
            job.kv = b""  # Receipt confirms D restored; retain the host reservation.
            self.events.put((rid, message["output_ids"], finish))
            if finish is not None:
                self._remove(rid)
            elif kv is not None:
                job.level = min(job.level + 1, 2)
                self._store(rid, job, message["output_ids"], kv, now)
                logger.info("MLFQ return rid=%s Q%d", rid, job.level)
            return {}

    def _age(self, now):
        # Promote oldest first, appending behind existing Q0 work.
        aged = sorted(
            (job.ready_at, rid)
            for rid, job in self.jobs.items()
            if job.level > 0
            and job.ready_at is not None
            and now - job.ready_at >= self.aging_s
        )
        for _, rid in aged:
            job = self.jobs[rid]
            self.ready[job.level].remove(rid)
            job.level = 0
            job.ready_at = now
            self.ready[0].append(rid)
            logger.info("MLFQ promote rid=%s Q0", rid)


class MLFQRuntime:
    """Small experimental execution leaf; requires live scheduler pools and I/O."""

    def __init__(self, scheduler):
        import torch
        import zmq

        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        from sglang.srt.runtime_context import (
            get_device,
            get_disagg,
            get_parallel,
            get_schedule,
            get_serving,
        )

        self.s = scheduler
        self.cfg = get_disagg()
        cfg, s, parallel = self.cfg, self.s, get_parallel()
        pool = s.token_to_kv_pool_allocator.get_kvcache()
        if (
            cfg.disaggregation_mode != "null"
            or get_device().device != "cuda"
            or not s.is_generation
            or not get_serving().skip_server_warmup
            or s.enable_pdmux
            or s.enable_overlap
            or not s.spec_algorithm.is_none()
            or s.enable_lora
            or s.is_hybrid_swa
            or s.tree_cache.supports_mamba()
            or not isinstance(pool, MHATokenToKVPool)
            or pool.use_hnd
            or pool.dtype not in (torch.float16, torch.bfloat16)
            or s.token_to_kv_pool_allocator.page_size != 1
            or not get_schedule().disable_radix_cache
            or get_schedule().max_running_requests != 1
            or any(
                getattr(parallel, size) != 1
                for size in (
                    "tp_size",
                    "pp_size",
                    "dp_size",
                    "attn_cp_size",
                    "attn_dcp_size",
                )
            )
        ):
            raise ValueError(
                "MLFQ baseline requires plain MHA FP16/BF16, TP/PP/DP/CP=1, "
                "page-size=1, max-running-requests=1, disabled radix/overlap, "
                "skip-server-warmup, no native PD, LoRA, SWA, Mamba or speculative decoding"
            )
        # Every D must load the same checkpoint revision and CPU-copy chunk layout.
        self.layout = (
            s.model_config.model_path,
            s.model_config.revision,
            str(pool.dtype),
            pool.layer_num,
            tuple(pool.k_buffer[0].shape[1:]),
            pool.cpu_offloading_chunk_size,
        )
        self.bytes_per_token = sum(
            tensor[0].numel() * tensor.element_size()
            for tensor in pool.k_buffer + pool.v_buffer
        )
        self.context = zmq.Context()
        self.original = {}
        self.pool = MLFQPool(
            cfg.mlfq_quanta_ms,
            cfg.mlfq_aging_ms,
            int(cfg.mlfq_host_memory_gb * 2**30),
            self.layout,
            cfg.mlfq_timeout_s,
        )
        self.socket = None
        self.stop = threading.Event()

    def run(self):
        import torch
        import zmq

        thread = None
        try:
            if self.cfg.mlfq_role == "prefill":
                # The broker owns only host data; it can answer D while P computes.
                started = queue.Queue()
                thread = threading.Thread(
                    target=self._serve,
                    args=(started, self.cfg.mlfq_endpoint),
                    daemon=True,
                )
                thread.start()
                error = started.get()
                if error:
                    raise error
                step = self._prefill
            else:
                self.socket = self.context.socket(zmq.REQ)
                self.socket.setsockopt(zmq.LINGER, 0)
                self.socket.setsockopt(
                    zmq.RCVTIMEO, int(self.cfg.mlfq_timeout_s * 1000)
                )
                self.socket.setsockopt(
                    zmq.SNDTIMEO, int(self.cfg.mlfq_timeout_s * 1000)
                )
                self.socket.connect(self.cfg.mlfq_endpoint)
                step = self._decode
            with torch.no_grad():
                while not self.s.gracefully_exit:
                    self.s.ingest_requests()
                    if self.cfg.mlfq_role == "prefill":
                        self._outputs()
                    if self.s._engine_paused or not step():
                        time.sleep(0.001)
        finally:
            self.stop.set()
            if self.socket is not None:
                self.socket.close()
            if thread is not None:
                thread.join(timeout=2)
            self.context.destroy(linger=0)

    def _serve(self, started, endpoint):
        import zmq

        socket = self.context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.bind(endpoint)
        except Exception as error:
            started.put(error)
            socket.close()
            return
        started.put(None)
        try:
            while not self.stop.is_set():
                if socket.poll(50):
                    message = socket.recv_pyobj()
                    try:
                        reply = self.pool.dispatch(message)
                    except Exception as error:
                        reply = {"error": str(error)}
                    socket.send_pyobj(reply)
        finally:
            socket.close()

    def _rpc(self, op, **message):
        self.socket.send_pyobj(dict(message, op=op, layout=self.layout))
        reply = self.socket.recv_pyobj()
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    def abort(self, request):
        for rid in list(self.original):
            if request.abort_all or rid.startswith(request.rid):
                self.pool.cancel(rid, "MLFQ request cancelled")

    def _reject(self, req, reason):
        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        req.finished_reason = FINISH_ABORT(reason, status_code=400)
        req.time_stats.set_completion_time()
        self.s.retire_unadmitted_request(req)

    def _state(self, req):
        # Keep client/stream state on P; only generation state travels to D.
        return dict(
            rid=req.rid,
            input_ids=req.origin_input_ids,
            output_ids=list(req.output_ids),
            sampling_params=req.sampling_params,
            eos_token_ids=req.eos_token_ids,
        )

    def reject_unsupported(self, req):
        params = req.sampling_params
        if self.cfg.mlfq_role != "prefill":
            self._reject(req, "Send generation requests to the MLFQ prefill server")
            return True
        unsupported = (
            req.multimodal_inputs is not None
            or req.input_embeds is not None
            or req.session is not None
            or req.grammar is not None
            or req.return_logprob
            or req.return_hidden_states
            or req.return_routed_experts
            or req.return_indexer_topk
            or req.return_sampling_mask
            or req.beam_group is not None
            or req.positional_embed_overrides is not None
            or req.custom_logit_processor is not None
            or params.top_k != 1  # temperature=0 is normalized to top_k=1.
            or params.frequency_penalty != 0
            or params.presence_penalty != 0
            or params.repetition_penalty != 1
            or params.max_new_tokens < 1
            or params.json_schema
            or params.regex
            or params.ebnf
            or params.structural_tag
        )
        if unsupported:
            self._reject(req, "MLFQ baseline supports plain text, greedy sampling only")
        return bool(unsupported)

    def _prefill(self):
        from sglang.srt.mem_cache.common import release_kv_cache

        s = self.s
        if not s.waiting_queue:
            return False
        req = s.waiting_queue[0]  # Strict FCFS: budget pressure blocks P, not D.
        params = req.sampling_params
        if req.to_finish is not None or req.finished():
            s.waiting_queue.pop(0)
            req.update_finish_state()
            req.time_stats.set_completion_time()
            s.output_streamer.stream_output([req], False)
            return True
        peak = len(req.origin_input_ids) + params.max_new_tokens
        # Covers peak KV plus conservative tensor/serialization/token metadata.
        reservation = (
            peak * (self.bytes_per_token + 16)
            + 65536 * s.model_config.num_hidden_layers
        )
        if (
            peak > s.token_to_kv_pool_allocator.available_size()
            or reservation > self.pool.host_bytes
        ):
            s.waiting_queue.pop(0)
            self._reject(req, "MLFQ request exceeds the configured device/host budget")
            return True
        if not self.pool.reserve(self._state(req), peak, reservation):
            return False
        s.waiting_queue.pop(0)
        self.original[req.rid] = req
        req.time_stats.set_forward_entry_time()
        batch = self._batch(req)
        self._step(batch)
        req.time_stats.set_prefill_finished_time()
        logger.info(
            "MLFQ prefill rid=%s input_tokens=%d",
            req.rid,
            len(req.origin_input_ids),
        )
        if req.finished():
            self.pool.cancel(req.rid)
            req.time_stats.set_completion_time()
            s.output_streamer.stream_output([req], False)
            del self.original[req.rid]
        else:
            self.pool.publish(req.rid, req.output_ids, self._checkpoint(req))
        release_kv_cache(req, s.tree_cache, checkpoint=False)
        return True

    def _outputs(self):
        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        self.pool.tick()
        while not self.pool.events.empty():
            rid, output, finish = self.pool.events.get()
            req = self.original.get(rid)
            if req is None:
                continue
            if output is not None:
                req.output_ids = array("q", output)
                req.time_stats.set_last_decode_finish_time()
            if finish is not None:
                req.finished_reason = (
                    FINISH_ABORT(finish, status_code=503)
                    if isinstance(finish, str)
                    else finish["reason"]
                )
                req.finished_len = (
                    len(req.output_ids) if isinstance(finish, str) else finish["length"]
                )
                req.time_stats.set_completion_time()
            self.s.output_streamer.stream_output([req], False)
            if req.finished():
                del self.original[rid]

    def _decode(self):
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.mem_cache.common import release_kv_cache

        s = self.s
        ticket = self._rpc(
            "pull", capacity=s.token_to_kv_pool_allocator.available_size()
        )
        if not ticket:
            return False
        state = ticket["state"]
        req = Req(
            state["rid"],
            None,
            state["input_ids"],
            state["sampling_params"],
            eos_token_ids=state["eos_token_ids"],
            vocab_size=s.model_config.vocab_size,
        )
        req.tokenizer = s.tokenizer
        req.output_ids = array("q", state["output_ids"])
        batch = self._batch(req, ticket["kv"])
        ticket.pop("kv")
        elapsed = 0.0
        logger.info("MLFQ start rid=%s Q%d", req.rid, ticket["level"])
        while True:
            elapsed += self._step(batch)
            update = dict(
                rid=req.rid, lease=ticket["lease"], output_ids=list(req.output_ids)
            )
            if req.finished():
                update["finish"] = dict(
                    reason=req.finished_reason, length=req.finished_len
                )
            elif elapsed >= ticket["quantum_ms"]:
                update.update(kv=self._checkpoint(req), service_ms=elapsed)
            reply = self._rpc("update", **update)
            if (
                req.finished()
                or elapsed >= ticket["quantum_ms"]
                or reply.get("cancelled")
            ):
                break
            batch.prepare_for_decode()
        # Return/finish has been acknowledged by P; now relinquish D's KV.
        release_kv_cache(req, s.tree_cache, checkpoint=False)
        return True

    def _batch(self, req, checkpoint=None):
        import torch

        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        s = self.s
        req.init_next_round_input()
        req.extend_end = req.seqlen
        if checkpoint is not None:
            # Exactly seqlen-1 KV tokens; the final sampled token is not computed yet.
            length = req.seqlen - 1
            indices = s.token_to_kv_pool_allocator.alloc(length)
            rows = s.req_to_token_pool.alloc([req])
            if indices is None or rows is None:
                raise RuntimeError("MLFQ restore allocation failed")
            req.kv.kv_allocated_len = req.kv.kv_committed_len = length
            s.req_to_token_pool.req_to_token[rows[0], :length] = indices
            cpu_kv = torch.load(
                io.BytesIO(checkpoint), map_location="cpu", weights_only=True
            )
            s.token_to_kv_pool_allocator.load_cpu_copy(cpu_kv, indices)
            req.prefix_len = length
        batch = ScheduleBatch.init_new(
            [req],
            s.req_to_token_pool,
            s.token_to_kv_pool_allocator,
            s.tree_cache,
            s.model_config,
            False,
            s.spec_algorithm,
        )
        batch.prepare_for_extend()
        if checkpoint is not None:
            # The one-token extend initializes SB metadata; execute it as decode.
            batch.forward_mode = ForwardMode.DECODE
        return batch

    def _step(self, batch):
        import torch

        # Time actual synchronous decode service, excluding transfer and RPC waits.
        torch.cuda.synchronize()
        start = time.monotonic()
        result = self.s.run_batch(batch)
        token = int(result.next_token_ids.item())
        torch.cuda.synchronize()
        elapsed_ms = (time.monotonic() - start) * 1000
        req = batch.reqs[0]
        req.output_ids.append(token)
        req.update_finish_state()
        return elapsed_ms

    def _checkpoint(self, req):
        import torch

        s = self.s
        indices = s.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, : req.seqlen - 1
        ]
        cpu_kv = s.token_to_kv_pool_allocator.get_cpu_copy(indices)
        with io.BytesIO() as buffer:
            torch.save(cpu_kv, buffer)
            return buffer.getvalue()
