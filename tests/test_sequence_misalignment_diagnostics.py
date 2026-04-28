from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from scripts.analyze_sequence_misalignment import token_diagnostics


class TinyTokenizer:
    def decode(self, ids, skip_special_tokens=False):  # noqa: ARG002
        return f"tok{ids[0]}"


def kl_from_logits(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    log_a = F.log_softmax(a, dim=-1)
    log_b = F.log_softmax(b, dim=-1)
    return (log_a.exp() * (log_a - log_b)).sum(dim=-1)


class SequenceMisalignmentDiagnosticsTest(unittest.TestCase):
    def test_constant_reward_reduces_j_to_policy_reference_kl(self) -> None:
        policy = torch.tensor([[1.0, 0.5, -0.5, -1.0, 0.0]], dtype=torch.float32)
        reference = torch.zeros_like(policy)
        reward = torch.full_like(policy, 3.0)

        rows, _ = token_diagnostics(
            tokenizer=TinyTokenizer(),
            policy_logits=policy,
            reference_logits=reference,
            rewards=reward,
            completion_ids=[0],
            beta=0.04,
            compute_dtype=torch.float32,
        )

        expected_kl = float(kl_from_logits(policy, reference)[0])
        self.assertAlmostEqual(rows[0]["J"], expected_kl, places=5)
        self.assertAlmostEqual(rows[0]["reward_vocab_std"], 0.0, places=6)
        self.assertAlmostEqual(rows[0]["gamma_abs_times_reward_std"], 0.0, places=6)
        self.assertGreater(rows[0]["grad_J_l2"], 0.0)

    def test_identical_policy_reference_and_constant_reward_has_zero_j(self) -> None:
        policy = torch.tensor([[0.3, -0.1, 0.2, -0.5, 0.0]], dtype=torch.float32)
        reference = policy.clone()
        reward = torch.ones_like(policy)

        rows, _ = token_diagnostics(
            tokenizer=TinyTokenizer(),
            policy_logits=policy,
            reference_logits=reference,
            rewards=reward,
            completion_ids=[2],
            beta=0.04,
            compute_dtype=torch.float32,
        )

        self.assertAlmostEqual(rows[0]["J"], 0.0, places=6)
        self.assertAlmostEqual(rows[0]["grad_J_l2"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
