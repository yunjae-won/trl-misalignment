from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from trl_misalignment.metrics import batched_vocablevel_misalignment


class RewardSignalStabilityTest(unittest.TestCase):
    def test_constant_reward_reduces_j_to_policy_reference_kl(self) -> None:
        policy_logits = torch.tensor([[2.0, 0.0, -1.0], [0.2, 1.1, -0.3]])
        reference_logits = torch.tensor([[1.0, 0.1, -0.2], [-0.4, 1.4, 0.3]])
        reward = torch.ones_like(policy_logits) * 0.01

        metrics = batched_vocablevel_misalignment(
            policy_logits,
            reference_logits,
            reward,
            beta=0.04,
            compute_dtype=torch.float64,
        )

        log_a = F.log_softmax(policy_logits.double(), dim=-1)
        log_b = F.log_softmax(reference_logits.double(), dim=-1)
        expected_kl = (log_a.exp() * (log_a - log_b)).sum(dim=-1)

        self.assertTrue(torch.allclose(metrics["J"], expected_kl, atol=1e-10))
        self.assertTrue(torch.allclose(metrics["gamma_star"], torch.zeros_like(metrics["gamma_star"])))
        self.assertTrue(metrics["gamma_bracketed"].all())

    def test_reward_scale_does_not_change_j_when_direction_is_same(self) -> None:
        torch.manual_seed(0)
        policy_logits = torch.randn(4, 9)
        reference_logits = torch.randn(4, 9)
        reward = torch.randn(4, 9)
        reward = reward - reward.mean(dim=-1, keepdim=True)

        small = batched_vocablevel_misalignment(
            policy_logits,
            reference_logits,
            1e-3 * reward,
            beta=0.04,
            compute_dtype=torch.float64,
            reward_tol=1e-14,
        )
        large = batched_vocablevel_misalignment(
            policy_logits,
            reference_logits,
            10.0 * reward,
            beta=0.04,
            compute_dtype=torch.float64,
            reward_tol=1e-14,
        )

        self.assertTrue(torch.allclose(small["J"], large["J"], atol=1e-8))
        self.assertTrue(torch.allclose(small["gamma_star"] * 1e-3, large["gamma_star"] * 10.0, atol=1e-8))


if __name__ == "__main__":
    unittest.main()
