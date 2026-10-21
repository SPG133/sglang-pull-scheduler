// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use sgl_router::policies::decode::joint::{joint_decode_match, DecodeCandidate};

fn check(
    waiting_ms: &[u64],
    capacity: &[u64],
    entries: &[(usize, usize, u64, u64)],
    aging: u32,
    expected: &[(usize, usize)],
) {
    let candidates: Vec<_> = entries
        .iter()
        .map(|&(request, worker, required_kv_tokens, cached_tokens)| DecodeCandidate {
            request,
            worker,
            required_kv_tokens,
            cached_tokens,
        })
        .collect();
    let mut actual = joint_decode_match(waiting_ms, capacity, &candidates, aging);
    actual.sort_unstable();
    assert_eq!(actual, expected);
}

#[test]
fn reassigns_instead_of_committing_the_largest_cache_hit() {
    check(
        &[0, 0],
        &[100, 100],
        &[
            (0, 0, 1, 100),
            (0, 1, 1, 99),
            (1, 0, 1, 98),
            (1, 1, 1, 0),
        ],
        0,
        &[(0, 1), (1, 0)],
    );
}

#[test]
fn moves_two_assignments_to_admit_a_waiting_long_request() {
    // S: A/B, M: A/B/C, L: A only, X: A/B/C.
    check(
        &[0, 0, 60, 0],
        &[8000, 5000, 3000],
        &[
            (0, 0, 4000, 100),
            (0, 1, 4000, 80),
            (1, 0, 3000, 0),
            (1, 1, 3000, 90),
            (1, 2, 3000, 70),
            (2, 0, 7000, 0),
            (3, 0, 2000, 0),
            (3, 1, 2000, 0),
            (3, 2, 2000, 10),
        ],
        1,
        &[(0, 1), (1, 2), (2, 0)],
    );
}

#[test]
fn admits_more_requests_even_when_total_cache_credit_decreases() {
    check(
        &[0, 0],
        &[5000, 3000],
        &[(0, 0, 3000, 100), (0, 1, 3000, 20), (1, 0, 5000, 0)],
        0,
        &[(0, 1), (1, 0)],
    );
}

#[test]
fn rejects_over_capacity_even_with_maximum_cache_credit() {
    check(
        &[0, 0],
        &[3000],
        &[(0, 0, 3001, u64::MAX), (1, 0, 3000, 0)],
        0,
        &[(1, 0)],
    );
}

#[test]
fn aging_can_outweigh_cache_affinity_without_overflow() {
    let entries = [(0, 0, 1, u64::MAX), (1, 0, 1, 0)];
    check(&[0, u64::MAX], &[1], &entries, 0, &[(0, 0)]);
    check(&[0, u64::MAX], &[1], &entries, u32::MAX, &[(1, 0)]);
}

#[test]
fn leaves_unmatchable_requests_pending() {
    check(&[], &[100], &[], 1, &[]);
    check(&[100], &[], &[], 1, &[]);
    check(&[100], &[100], &[], 1, &[]);
    check(&[100], &[0], &[(0, 0, 1, 100)], 1, &[]);
}
