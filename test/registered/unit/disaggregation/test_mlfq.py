"""MLFQ ownership, queue ordering and host-memory lifecycle contracts."""

import io
import queue
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import torch
import zmq

from sglang.srt.disaggregation.mlfq import MLFQPool, MLFQRuntime
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMLFQ(CustomTestCase):
    def setUp(self):
        self.pool = MLFQPool([10, 20, 40], 100, 1000, "model", timeout_s=10)

    def submit(self, rid, peak=10, reservation=100):
        self.assertTrue(self.pool.reserve({"rid": rid}, peak, reservation))
        self.pool.publish(rid, [1], b"prefilled-kv", now=0)

    def rpc(self, op, now=0, **kwargs):
        return self.pool.dispatch(dict(op=op, layout="model", **kwargs), now=now)

    def test_fifo_capacity_and_unique_claim(self):
        self.submit("large", peak=20)
        self.submit("small", peak=5)
        ticket = self.rpc("pull", capacity=10)
        self.assertEqual(ticket["rid"], "small")
        self.assertEqual(self.rpc("pull", capacity=30)["rid"], "large")
        self.assertEqual(self.rpc("pull", capacity=30), {})

    def test_checkpoint_ack_demotion_cross_worker_and_stale_return(self):
        self.submit("r")
        first = self.rpc("pull", capacity=30)
        identity = dict(rid="r", lease=first["lease"])
        self.assertEqual(self.pool.jobs["r"].kv, b"prefilled-kv")
        self.rpc("update", **identity, output_ids=[1])
        self.assertEqual(self.pool.jobs["r"].kv, b"")
        self.rpc(
            "update", **identity, output_ids=[1, 2], kv=b"full-new-kv", service_ms=11
        )
        second = self.rpc("pull", capacity=30)
        self.assertEqual(
            (second["rid"], second["level"], second["quantum_ms"]), ("r", 1, 20)
        )
        self.assertEqual(second["kv"], b"full-new-kv")
        self.assertEqual(second["state"]["output_ids"], [1, 2])
        self.assertTrue(
            self.rpc("update", **identity, output_ids=[99], kv=b"stale", service_ms=10)[
                "cancelled"
            ]
        )
        self.assertEqual(self.pool.jobs["r"].lease, second["lease"])

    def test_bottom_queue_rotation_and_both_levels_age_to_q0(self):
        for rid in ("a", "b"):
            self.submit(rid)
            for elapsed in (10, 20):
                ticket = self.rpc("pull", capacity=30)
                self.rpc(
                    "update",
                    rid=rid,
                    lease=ticket["lease"],
                    output_ids=[1],
                    kv=b"kv",
                    service_ms=elapsed,
                )
        self.assertEqual(list(self.pool.ready[2]), ["a", "b"])
        ticket = self.rpc("pull", capacity=30)
        self.rpc(
            "update",
            rid="a",
            lease=ticket["lease"],
            output_ids=[1],
            kv=b"kv",
            service_ms=40,
        )
        self.assertEqual(list(self.pool.ready[2]), ["b", "a"])
        self.submit("q1")
        ticket = self.rpc("pull", capacity=30)
        self.rpc(
            "update",
            rid="q1",
            lease=ticket["lease"],
            output_ids=[1],
            kv=b"kv",
            service_ms=10,
        )
        self.pool.tick(now=0.101)
        self.assertEqual(list(self.pool.ready[0]), ["a", "b", "q1"])
        self.assertTrue(all(job.level == 0 for job in self.pool.jobs.values()))

    def test_budget_backpressure_cancel_and_worker_loss(self):
        self.submit("r", reservation=1000)
        self.assertFalse(self.pool.reserve({"rid": "next"}, 10, 1))
        ticket = self.rpc("pull", capacity=30)
        self.rpc("update", rid="r", lease=ticket["lease"], output_ids=[1])
        self.assertEqual(self.pool.reserved, 1000)  # Reserved even while KV is on D.
        self.assertEqual(self.pool.events.get(), ("r", [1], None))
        self.pool.tick(now=11)
        self.assertEqual(self.pool.reserved, 0)
        self.assertIn("timed out", self.pool.events.get()[2])
        self.submit("cancel")
        self.pool.cancel("cancel")
        self.assertEqual(self.pool.reserved, 0)
        self.assertTrue(all(not pending for pending in self.pool.ready))

    def test_bad_return_is_atomic_and_layout_is_checked(self):
        self.submit("r")
        ticket = self.rpc("pull", capacity=30)
        for elapsed, kv in ((9, b"kv"), (10, b"x" * 101)):
            with self.assertRaises(ValueError):
                self.rpc(
                    "update",
                    rid="r",
                    lease=ticket["lease"],
                    output_ids=[9],
                    kv=kv,
                    service_ms=elapsed,
                )
            self.assertEqual(
                (self.pool.jobs["r"].lease, self.pool.jobs["r"].level),
                (ticket["lease"], 0),
            )
        reply = self.pool.dispatch(dict(op="pull", layout="other", capacity=30), now=0)
        self.assertIn("error", reply)

    def test_concurrent_decoders_cannot_claim_same_request(self):
        self.submit("r")
        with ThreadPoolExecutor(max_workers=8) as executor:
            replies = list(
                executor.map(lambda i: self.rpc("pull", capacity=30), range(8))
            )
        self.assertEqual(sum(bool(reply) for reply in replies), 1)

    def test_finish_releases_budget_and_old_lease_cannot_finish_reused_id(self):
        self.submit("r")
        first = self.rpc("pull", capacity=30)
        finish = {"reason": "length", "length": 2}
        self.rpc(
            "update", rid="r", lease=first["lease"], output_ids=[1, 2], finish=finish
        )
        self.assertEqual(self.pool.reserved, 0)
        self.assertEqual(self.pool.events.get(), ("r", [1, 2], finish))
        self.submit("r")
        second = self.rpc("pull", capacity=30)
        self.assertGreater(second["lease"], first["lease"])
        self.assertTrue(
            self.rpc(
                "update", rid="r", lease=first["lease"], output_ids=[99], finish=finish
            )["cancelled"]
        )
        self.assertIn("r", self.pool.jobs)

    def test_tcp_checkpoint_can_move_between_decoder_connections(self):
        context = zmq.Context()
        probe = context.socket(zmq.REP)
        port = probe.bind_to_random_port("tcp://127.0.0.1")
        probe.close()
        endpoint = f"tcp://127.0.0.1:{port}"
        # Exercise the real host broker without constructing any GPU scheduler.
        runtime = object.__new__(MLFQRuntime)
        runtime.context = context
        runtime.pool = MLFQPool([10, 20, 40], 100, 100000, "model")
        runtime.stop = threading.Event()
        started = queue.Queue()
        thread = threading.Thread(
            target=runtime._serve, args=(started, endpoint), daemon=True
        )
        clients = []
        thread.start()
        try:
            self.assertIsNone(started.get(timeout=3))
            for _ in range(2):
                client = context.socket(zmq.REQ)
                client.setsockopt(zmq.LINGER, 0)
                client.setsockopt(zmq.RCVTIMEO, 3000)
                client.connect(endpoint)
                clients.append(client)
            original = [[[torch.arange(8).reshape(2, 4), torch.ones(2, 4)]]]
            buffer = io.BytesIO()
            torch.save(original, buffer)
            self.assertTrue(runtime.pool.reserve({"rid": "r"}, 10, 50000))
            runtime.pool.publish("r", [1], buffer.getvalue())

            def send(index, op, **kw):
                clients[index].send_pyobj(dict(op=op, layout="model", **kw))
                return clients[index].recv_pyobj()

            first = send(0, "pull", capacity=20)
            send(0, "update", rid="r", lease=first["lease"], output_ids=[1])
            send(
                0,
                "update",
                rid="r",
                lease=first["lease"],
                output_ids=[1, 2],
                kv=first["kv"],
                service_ms=10,
            )
            second = send(1, "pull", capacity=20)
            restored = torch.load(io.BytesIO(second["kv"]), weights_only=True)
            self.assertTrue(torch.equal(original[0][0][0], restored[0][0][0]))
            self.assertTrue(torch.equal(original[0][0][1], restored[0][0][1]))
            self.assertEqual(
                (second["level"], second["state"]["output_ids"]), (1, [1, 2])
            )
        finally:
            runtime.stop.set()
            for client in clients:
                client.close()
            thread.join(timeout=3)
            context.destroy(linger=0)
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
