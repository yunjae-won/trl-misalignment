from __future__ import annotations

import unittest

from trl_misalignment.rewards import TokenVocabRewardProvider


def normalize(ids: list[int], *, eos_token_id: int | None, pad_token_id: int | None) -> list[int]:
    return TokenVocabRewardProvider._normalize_completion_ids(
        ids,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )


class NormalizeCompletionIdsTest(unittest.TestCase):
    def test_pad_aliasing_eos_is_not_stripped_as_padding(self) -> None:
        self.assertEqual(
            normalize([11, 22, 2, 2], eos_token_id=2, pad_token_id=2),
            [11, 22, 2, 2],
        )

    def test_trims_after_last_completion_eos(self) -> None:
        self.assertEqual(
            normalize([11, 2, 22, 2, 198, 198], eos_token_id=2, pad_token_id=0),
            [11, 2, 22, 2],
        )

    def test_strips_non_eos_padding_before_eos_trim(self) -> None:
        self.assertEqual(
            normalize([11, 22, 2, 0, 0], eos_token_id=2, pad_token_id=0),
            [11, 22, 2],
        )

    def test_no_completion_eos_leaves_completion_tokens_unchanged(self) -> None:
        self.assertEqual(
            normalize([11, 22, 198], eos_token_id=2, pad_token_id=0),
            [11, 22, 198],
        )


if __name__ == "__main__":
    unittest.main()
