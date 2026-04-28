from __future__ import annotations

import io
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from fastapi.testclient import TestClient

from logprob_engine import LogprobEngine, create_app


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 6
    pad_token = "<pad>"

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [int(x) for x in text.split()]


class TinyBody(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


class TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 7, hidden_size: int = 5) -> None:
        super().__init__()
        self.model = TinyBody(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = SimpleNamespace(vocab_size=vocab_size)


def make_model() -> TinyCausalLM:
    torch.manual_seed(1234)
    model = TinyCausalLM()
    with torch.no_grad():
        model.model.embed.weight.copy_(
            torch.arange(model.config.vocab_size * 5, dtype=torch.float32).view(model.config.vocab_size, 5) / 17.0
        )
        model.lm_head.weight.copy_(
            torch.arange(model.config.vocab_size * 5, dtype=torch.float32).view(model.config.vocab_size, 5) / 23.0
        )
    return model


def expected_vocab(model: TinyCausalLM, prompt_ids: list[int], output_ids: list[int]) -> torch.Tensor:
    input_ids, labels = LogprobEngine._construct_input(prompt_ids, output_ids)
    shifted_labels = labels[1:]
    mask = shifted_labels != -100
    hidden = model.model.forward(input_ids.unsqueeze(0)).last_hidden_state[:, :-1][mask.unsqueeze(0)]
    return torch.log_softmax(model.lm_head(hidden), dim=-1)


class LogprobEngineFastPathTest(unittest.TestCase):
    def test_vocab_tensor_array_and_list_paths_match_expected_values(self) -> None:
        model = make_model()
        engine = LogprobEngine.from_components(
            model,
            TinyTokenizer(),
            device="cpu",
            compile=False,
            logprob_level="vocab",
            logprob_dtype="float32",
        )
        items = [
            {"prompt_ids": [1, 2, 3], "output_ids": [4, 5]},
            {"prompt_ids": [2], "output_ids": [3, 4, 5]},
        ]

        tensors = engine.process_tensors(items)
        arrays = engine.process_arrays(items)
        lists = engine.process(items)

        for item, tensor, array, list_value in zip(items, tensors, arrays, lists, strict=True):
            expected = expected_vocab(model, item["prompt_ids"], item["output_ids"])
            expected_np = expected.detach().numpy()
            self.assertEqual(tuple(tensor.shape), tuple(expected.shape))
            self.assertTrue(torch.allclose(tensor.cpu(), expected, atol=1e-6))
            self.assertIsInstance(array, np.ndarray)
            self.assertTrue(np.allclose(array, expected_np, atol=1e-6))
            self.assertTrue(np.allclose(np.asarray(list_value), expected_np, atol=1e-6))

    def test_token_and_seq_paths_match_vocab_gather_and_sum(self) -> None:
        items = [
            {"prompt_ids": [1, 2, 3], "output_ids": [4, 5]},
            {"prompt_ids": [2], "output_ids": [3, 4, 5]},
        ]
        token_engine = LogprobEngine.from_components(
            make_model(),
            TinyTokenizer(),
            device="cpu",
            compile=False,
            logprob_level="token",
            logprob_dtype="float32",
        )
        seq_engine = LogprobEngine.from_components(
            make_model(),
            TinyTokenizer(),
            device="cpu",
            compile=False,
            logprob_level="seq",
            logprob_dtype="float32",
        )

        token_tensors = token_engine.process_tensors(items)
        seq_tensors = seq_engine.process_tensors(items)
        expected_model = make_model()
        for item, token_tensor, seq_tensor in zip(items, token_tensors, seq_tensors, strict=True):
            vocab = expected_vocab(expected_model, item["prompt_ids"], item["output_ids"])
            labels = torch.as_tensor(item["output_ids"], dtype=torch.long)
            expected_tokens = vocab.gather(1, labels[:, None]).squeeze(1)
            self.assertTrue(torch.allclose(token_tensor.cpu(), expected_tokens, atol=1e-6))
            self.assertTrue(torch.allclose(seq_tensor.cpu(), expected_tokens.sum(), atol=1e-6))

    def test_http_npz_path_returns_arrays_without_json_roundtrip(self) -> None:
        engine = LogprobEngine.from_components(
            make_model(),
            TinyTokenizer(),
            device="cpu",
            compile=False,
            logprob_level="vocab",
            logprob_dtype="float32",
        )
        items = [{"prompt_ids": [1, 2], "output_ids": [3, 4]}]
        client = TestClient(create_app(engine))

        resp = client.post("/v1/logprobs", params={"format": "npz"}, json={"items": items})
        self.assertEqual(resp.status_code, 200)
        with np.load(io.BytesIO(resp.content)) as npz:
            actual = npz["item_0"]

        expected = engine.process_arrays(items)[0]
        self.assertTrue(np.allclose(actual, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
