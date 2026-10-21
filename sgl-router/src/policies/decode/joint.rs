// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Joint admission for one scheduling round, with at most one request per D.
//! Maximizes admitted requests, then cache reuse plus waiting-time credit.
//! Inputs must share a resource snapshot; the caller must reserve and revalidate
//! the resulting plan before dispatch. This module performs no allocation or I/O.

/// A D-reported, compatible request/worker pair. Indices refer to input slices.
#[derive(Debug, Clone, Copy)]
pub struct DecodeCandidate {
    pub request: usize,
    pub worker: usize,
    /// Additional KV capacity, including output budget and page alignment.
    /// Only subtract reusable cache that the D can protect during admission.
    pub required_kv_tokens: u64,
    pub cached_tokens: u64,
}

struct Edge {
    from: usize,
    to: usize,
    available: bool,
    cost: i128,
}

fn add_edge(edges: &mut Vec<Edge>, from: usize, to: usize, cost: i128) -> usize {
    let index = edges.len();
    edges.push(Edge {
        from,
        to,
        available: true,
        cost,
    });
    edges.push(Edge {
        from: to,
        to: from,
        available: false,
        cost: -cost,
    });
    index
}

/// Exact maximum-cardinality, maximum-credit bipartite matching for this round.
///
/// `free_kv_tokens` must exclude existing reservations. Only include workers with
/// a free compute/request slot and fresh capacity/cache reports in `candidates`.
/// Waiting credit is in tokens/ms; a positive value increases older requests'
/// priority, but cannot guarantee a deadline under overload or infeasible demand.
/// Complexity is O(min(R, D) * (R + D) * (R + D + C)); bound the pending window.
pub fn joint_decode_match(
    waiting_ms: &[u64],
    free_kv_tokens: &[u64],
    candidates: &[DecodeCandidate],
    aging_tokens_per_ms: u32,
) -> Vec<(usize, usize)> {
    let workers_start = waiting_ms.len();
    let source = workers_start + free_kv_tokens.len();
    let sink = source + 1;
    let mut edges = Vec::new();
    for request in 0..waiting_ms.len() {
        add_edge(&mut edges, source, request, 0);
    }
    for worker in 0..free_kv_tokens.len() {
        add_edge(&mut edges, workers_start + worker, sink, 0);
    }
    let mut pairs = Vec::new();
    for &DecodeCandidate {
        request,
        worker,
        required_kv_tokens,
        cached_tokens,
    } in candidates
    {
        assert!(request < waiting_ms.len() && worker < free_kv_tokens.len());
        if required_kv_tokens > free_kv_tokens[worker] {
            continue;
        }
        let credit = i128::from(cached_tokens)
            + i128::from(waiting_ms[request]) * i128::from(aging_tokens_per_ms);
        let edge = add_edge(&mut edges, request, workers_start + worker, -credit);
        pairs.push((request, worker, edge));
    }

    // Residual reverse edges let later requests move an earlier assignment to
    // another D, preserving scarce capacity instead of committing greedily.
    loop {
        let mut distance = vec![i128::MAX; sink + 1];
        let mut previous = vec![None; sink + 1];
        distance[source] = 0;
        for _ in 0..sink {
            let mut changed = false;
            for (index, edge) in edges.iter().enumerate() {
                if !edge.available || distance[edge.from] == i128::MAX {
                    continue;
                }
                let next = distance[edge.from] + edge.cost;
                if next < distance[edge.to] {
                    distance[edge.to] = next;
                    previous[edge.to] = Some(index);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        if previous[sink].is_none() {
            break;
        }
        let mut to = sink;
        while to != source {
            let index = previous[to].unwrap();
            edges[index].available = false;
            // Forward/reverse edges are adjacent: 0 <-> 1, 2 <-> 3, ...
            edges[index ^ 1].available = true;
            to = edges[index].from;
        }
    }
    pairs
        .into_iter()
        .filter(|&(_, _, edge)| !edges[edge].available)
        .map(|(request, worker, _)| (request, worker))
        .collect()
}
