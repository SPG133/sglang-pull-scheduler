"""Run against live MLFQ P and reference servers; see examples/1pnd_mlfq/README.md."""

import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.test.test_utils import CustomTestCase


class TestMLFQLive(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.prefill = os.environ.get("MLFQ_PREFILL_URL")
        cls.reference = os.environ.get("MLFQ_REFERENCE_URL")
        if not cls.prefill or not cls.reference:
            raise unittest.SkipTest("Set MLFQ_PREFILL_URL and MLFQ_REFERENCE_URL")

    def generate(self, url, index, stream=False):
        return requests.post(
            url + "/generate",
            json={
                "text": f"Continue this numbered list: {index}, {index + 1},",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 48 + index,
                    "ignore_eos": True,
                },
                "stream": stream,
            },
            stream=stream,
            timeout=180,
        )

    def test_concurrent_outputs_equal_uninterrupted_reference(self):
        expected = []
        for index in range(6):
            with self.generate(self.reference, index) as response:
                response.raise_for_status()
                expected.append(response.json()["output_ids"])

        def run(index):
            with self.generate(self.prefill, index) as response:
                response.raise_for_status()
                return response.json()["output_ids"]

        with ThreadPoolExecutor(max_workers=6) as executor:
            actual = list(executor.map(run, range(6)))
        self.assertEqual(actual, expected)
        self.assertEqual([len(ids) for ids in actual], list(range(48, 54)))

    def test_streaming_has_no_lost_or_repeated_tokens(self):
        with self.generate(self.reference, 0) as response:
            response.raise_for_status()
            expected = response.json()["output_ids"]
        previous = []
        done = False
        # Use the default non-incremental stream mode on both servers.
        with self.generate(self.prefill, 0, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith(b"data: "):
                    continue
                data = line[6:]
                if data == b"[DONE]":
                    done = True
                    break
                current = json.loads(data)["output_ids"]
                self.assertEqual(current[: len(previous)], previous)
                self.assertEqual(current, expected[: len(current)])
                previous = current
        self.assertTrue(done)
        self.assertEqual(previous, expected)


if __name__ == "__main__":
    unittest.main()
