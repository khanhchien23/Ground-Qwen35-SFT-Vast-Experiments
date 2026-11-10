"""CPU unit tests; no model download, transformers, or GPU required."""
import ast
import math
import re
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from ground_prompts import GROUND_CLASSES
from scheduled_cot import cosine_cot_weight, segment_normalized_loss


class ScheduledLossTests(unittest.TestCase):
    def test_cosine_and_resume(self):
        for step, expected in ((0, 1), (242, .875), (484, .625), (726, .5)):
            self.assertAlmostEqual(cosine_cot_weight(step, 726), expected)
        self.assertEqual(cosine_cot_weight(484, 726), cosine_cot_weight(484, 726))

    def test_loss_and_gradients_match_reference(self):
        torch.manual_seed(42)
        logits = torch.randn(2, 150, 7, requires_grad=True)
        labels = torch.randint(0, 7, (2, 150))
        labels[:, :3] = -100
        labels[1, 140:] = -100
        segments = torch.zeros_like(labels)
        segments[:, 3:130] = 1
        segments[:, 130:] = 2
        actual = segment_normalized_loss(logits, labels, segments, .7)
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, 7), labels[:, 1:].reshape(-1), reduction="none").reshape(2, -1)
        reference = []
        for row in range(2):
            parts = []
            for seg, weight in ((1, .7), (2, 1)):
                mask = (segments[row, 1:] == seg) & (labels[row, 1:] != -100)
                parts.append(weight * ce[row][mask].mean())
            reference.append(sum(parts))
        reference = torch.stack(reference).mean()
        self.assertTrue(torch.allclose(actual, reference))
        grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
        expected = torch.autograd.grad(reference, logits)[0]
        self.assertTrue(torch.allclose(grad, expected, atol=1e-7))
        self.assertEqual(grad[:, :2].abs().sum().item(), 0)

    def test_length_independence_and_missing_segment(self):
        for length in (8, 200):
            logits = torch.zeros(1, length, 4)
            labels = torch.zeros(1, length, dtype=torch.long)
            segments = torch.ones_like(labels)
            segments[:, -2:] = 2
            loss = segment_normalized_loss(logits, labels, segments, .5)
            self.assertAlmostEqual(loss.item(), 1.5 * math.log(4), places=5)
        with self.assertRaises(ValueError):
            segment_normalized_loss(logits, labels, torch.ones_like(labels), .5)

    def test_collator_segments_with_variable_padding(self):
        # Load only the real collator AST to avoid importing GPU/model packages.
        tree = ast.parse(Path(__file__).with_name("train_grd_vast.py").read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GroundDataCollator")
        scope = dict(torch=torch, re=re, dataclass=dataclass, Any=Any, Dict=Dict,
                     List=List, HAS_QWEN_UTILS=False, GROUND_CLASSES=GROUND_CLASSES)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "collator", "exec"), scope)

        class Processor:
            def tokenizer(self, text, **kwargs):
                return {"input_ids": [ord(c) for c in text]}

            def apply_chat_template(self, messages, **kwargs):
                prompt = "image<|im_start|>assistant\n<think>\n"
                if len(messages) == 1:
                    return prompt
                return prompt + messages[-1]["content"][0]["text"][len("<think>\n"):] + "!"

            def __call__(self, text, **kwargs):
                ids = torch.zeros(len(text), max(map(len, text)), dtype=torch.long)
                mask = torch.zeros_like(ids)
                for i, value in enumerate(text):
                    ids[i, :len(value)] = torch.tensor([ord(c) for c in value])
                    mask[i, :len(value)] = 1
                return dict(input_ids=ids, attention_mask=mask)

        features = []
        for reasoning, answer in (("short", "Grd.Himars"), ("a longer reasoning", "Grd.BMP2")):
            message = dict(content=[dict(text=f"<think>\n{reasoning}</think>\n<answer>{answer}</answer>")])
            features.append(dict(prompt_messages=[{}], full_messages=[{}, message], image=None))
        batch = scope["GroundDataCollator"](Processor(), 1024, "scheduled_cot")(features)
        for row, feature in enumerate(features):
            valid = batch["labels"][row] != -100
            self.assertTrue((batch["loss_segments"][row][~valid] == 0).all())
            answer_ids = batch["input_ids"][row][batch["loss_segments"][row] == 2].tolist()
            self.assertTrue("".join(map(chr, answer_ids)).startswith("</think>"))
            self.assertTrue((batch["loss_segments"][row] == 1).any())


if __name__ == "__main__":
    unittest.main()
