import json
import math
import re
import unittest
from pathlib import Path

import torch

from src.evaluate_pass_receiver_model import (
    LABELS,
    candidate_token_ids,
    mean_logprob,
    rank_candidates,
    ranking_metrics,
    score_candidates_full,
)
from src.build_pass_receiver_dataset import TRAIN_MATCHES, VAL_MATCHES
from train.train_pass_receiver_qwen import encode_example, validate_rows

DATA = Path("data")


def load_tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base", local_files_only=True)
    except Exception:
        return None


class ToyLM(torch.nn.Module):
    """Causal toy model: logits depend on the running mean of token embeddings."""

    def __init__(self, vocab=50, dim=8):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(vocab, dim)
        self.out = torch.nn.Linear(dim, vocab)

    def forward(self, input_ids):
        e = self.embed(input_ids)
        steps = torch.arange(1, e.shape[1] + 1).view(1, -1, 1)
        return type("Out", (), {"logits": self.out(e.cumsum(1) / steps)})


class FakeTokenizer:
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=True, truncation=True, max_length=None):
        ids = [ord(c) % 40 + 1 for c in text.replace("<eos>", "")]
        if text.endswith("<eos>"):
            ids.append(49)
        if add_special_tokens:
            ids = [48] + ids
        return {"input_ids": ids[:max_length]}


class PassReceiverTrainingEvalTest(unittest.TestCase):
    def test_all_ten_labels_accepted(self):
        rows = [{"response": label} for label in LABELS]
        validate_rows(rows, "x")
        with self.assertRaises(ValueError):
            validate_rows([{"response": "TEAMMATE_11"}], "x")
        with self.assertRaises(ValueError):
            validate_rows([{"response": "PASS"}], "x")

    def test_teammate_10_not_truncated(self):
        tokenizer = load_tokenizer()
        if tokenizer is None:
            self.skipTest("Qwen tokenizer not cached")
        cands = candidate_token_ids(tokenizer)
        for label, ids in cands.items():
            decoded = tokenizer.decode(ids[:-1])
            self.assertEqual(decoded, label)
            self.assertEqual(ids[-1], tokenizer.convert_tokens_to_ids(tokenizer.eos_token))
        self.assertNotEqual(cands["TEAMMATE_1"], cands["TEAMMATE_10"])
        self.assertEqual(
            encode_example(tokenizer, {"prompt": "x", "response": "TEAMMATE_10"}, 2048)["labels"][-len(cands["TEAMMATE_10"]):],
            cands["TEAMMATE_10"],
        )

    def test_training_labels_mask_prompt(self):
        enc = encode_example(FakeTokenizer(), {"prompt": "abcdef", "response": "XY"}, 2048)
        n_prompt = len(FakeTokenizer()("abcdef")["input_ids"])
        self.assertTrue(all(v == -100 for v in enc["labels"][:n_prompt]))
        self.assertTrue(all(v != -100 for v in enc["labels"][n_prompt:]))
        self.assertEqual(set(enc), {"input_ids", "attention_mask", "labels"})

    def test_candidate_scoring_ignores_prompt_tokens(self):
        model = ToyLM().eval()
        prompt, ids = [3, 7, 11, 5], [20, 21, 22]
        got = score_candidates_full(model, prompt, {"TEAMMATE_1": ids})["TEAMMATE_1"]
        seq = torch.tensor([prompt + ids])
        logp = torch.log_softmax(model(seq).logits[0], -1)
        manual = sum(logp[len(prompt) - 1 + k, t].item() for k, t in enumerate(ids)) / 3
        self.assertAlmostEqual(got, manual, places=5)
        # prompt-token log-probs would change the number if they were included
        with_prompt = sum(logp[k, t].item() for k, t in enumerate(prompt + ids, start=0) if k > 0)
        self.assertNotAlmostEqual(got * 3, with_prompt, places=3)

    def test_mean_logprob_removes_length_bias(self):
        logits = torch.zeros(3, 4)
        self.assertAlmostEqual(mean_logprob(logits[:1], [0]), mean_logprob(logits, [0, 1, 2]), places=6)

    def test_ranking_metrics(self):
        def scores(best):
            s = {label: -float(i) for i, label in enumerate(LABELS)}
            s[best] = 1.0
            return s

        ranked = [rank_candidates(scores("TEAMMATE_1")),   # true rank 1
                  rank_candidates(scores("TEAMMATE_1")),   # true TEAMMATE_2 -> rank 2
                  rank_candidates(scores("TEAMMATE_4"))]   # true TEAMMATE_1 -> rank 2
        true = ["TEAMMATE_1", "TEAMMATE_2", "TEAMMATE_1"]
        m = ranking_metrics(true, ranked)
        self.assertAlmostEqual(m["top1"], 1 / 3)
        self.assertAlmostEqual(m["top3"], 1.0)
        self.assertAlmostEqual(m["mrr"], (1 + 0.5 + 0.5) / 3)
        self.assertAlmostEqual(m["mean_rank"], 5 / 3)
        self.assertEqual(m["median_rank"], 2)
        self.assertEqual(m["per_class"]["TEAMMATE_1"]["prediction_count"], 2)
        self.assertAlmostEqual(m["per_class"]["TEAMMATE_1"]["precision"], 0.5)
        self.assertAlmostEqual(m["per_class"]["TEAMMATE_1"]["recall"], 0.5)

    @unittest.skipUnless((DATA / "train_pass_receiver.jsonl").exists(), "dataset not built")
    def test_dataset_split_and_no_permanent_ids(self):
        for name, matches in (("train", TRAIN_MATCHES), ("val", VAL_MATCHES)):
            with open(DATA / f"{name}_pass_receiver.jsonl", encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream]
            self.assertEqual({r["match_id"] for r in rows}, set(matches))
            for r in rows[:300]:
                self.assertNotIn(r["passer_id"], r["prompt"])
                self.assertNotIn(r["receiver_id"], r["prompt"])
                self.assertIsNone(re.search(r"\b\d{5,}\b", r["prompt"].split("<|user|>")[1]))
                self.assertIn(r["response"], LABELS)


if __name__ == "__main__":
    unittest.main()
