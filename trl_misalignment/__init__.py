from .compat import apply_runtime_compatibility_patches

apply_runtime_compatibility_patches()

from .metrics import (
    VocabMisalignmentConfig,
    batched_vocablevel_misalignment,
    completion_vocab_misalignment,
)
from .rewards import (
    TokenVocabRewardProvider,
    VocabRewardOutput,
    make_token_vocab_reward_func,
    pad_vocab_reward,
)
from .trainers import (
    MisalignmentGRPOTrainer,
    MisalignmentOnlineDPOTrainer,
    MisalignmentPPOTrainer,
)

__all__ = [
    "VocabMisalignmentConfig",
    "batched_vocablevel_misalignment",
    "completion_vocab_misalignment",
    "TokenVocabRewardProvider",
    "VocabRewardOutput",
    "make_token_vocab_reward_func",
    "pad_vocab_reward",
    "MisalignmentGRPOTrainer",
    "MisalignmentOnlineDPOTrainer",
    "MisalignmentPPOTrainer",
]
