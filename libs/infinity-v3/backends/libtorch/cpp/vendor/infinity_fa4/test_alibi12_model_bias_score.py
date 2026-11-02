#!/usr/bin/env python3
# Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
"""CPU-only guard for the additive JinaBase twelve-head ALiBi score recipe."""
import pathlib
import struct
import sys
import unittest

VENDOR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(VENDOR))
import alibi12_model_bias_score as score  # noqa: E402


def f32(value):
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def bf16(value):
    return struct.unpack("<f", struct.pack("<f", score.round_f32_to_bf16(value)))[0]


class JinaBaseTwelveHeadScoreTests(unittest.TestCase):
    def test_export_is_twelve_heads(self):
        self.assertEqual(score.ALIBI_HEADS, 12)
        self.assertEqual(score.ALIBI_HEAD_DIM, 64)
        self.assertTrue(score.validate_alibi_shape(
            1, 8, 1, 12, 8, [2304, 64, 1, 2304, 64, 1,
                            2304, 64, 1, 768, 64, 1]))
        with self.assertRaises(ValueError):
            score.validate_alibi_shape(
                1, 8, 1, 8, 8, [1536, 64, 1, 1536, 64, 1,
                                1536, 64, 1, 512, 64, 1])

    def test_bias_is_rounded_once_to_the_model_dtype(self):
        slopes = [0.5, 0.25, 1.0 / (2.0 ** 0.5), (1.0 / (2.0 ** 0.5)) ** 7]
        for dtype in ("bf16", "fp16"):
            for slope in slopes:
                for distance in (0, 1, 2, 3, 7, 255, 8191):
                    product = f32(f32(slope) * f32(distance))
                    expected = score.round_to_model_dtype(-product, dtype)
                    actual = score.alibi_bias_f32(slope, distance, dtype)
                    self.assertEqual(actual, expected)
                    self.assertEqual(
                        score.alibi_score_f32(3.0, slope, distance, dtype),
                        f32(f32(f32(3.0) * f32(0.125)) + expected))

    def test_base_recipe_differs_from_unrounded_fma(self):
        slope = 1.0 / (2.0 ** 0.5)
        distance = 3
        model = score.alibi_bias_f32(slope, distance, "bf16")
        unrounded = f32(-f32(f32(slope) * f32(distance)))
        self.assertEqual(model, bf16(unrounded))
        raw_fma_score = f32(f32(1.0 * 0.125) + unrounded)
        base_score = score.alibi_score_f32(1.0, slope, distance, "bf16")
        self.assertNotEqual(base_score, raw_fma_score)


if __name__ == "__main__":
    unittest.main(verbosity=2)
