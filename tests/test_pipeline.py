"""Automated unit tests for the PaddyCare AI ML pipeline (runs with both unittest and pytest)."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EMBEDDING_DIM, get_entropy_threshold
from dataset_utils import PadToSquare
from model import build_model, DualOutputWrapper, forward_with_embedding, _remap_legacy_head_keys
from train import should_promote_checkpoint, make_param_groups
from extract_embeddings import _distances, calibrate_confidence_threshold
from verify_dataset import dhash_image


class TestPipeline(unittest.TestCase):

    def test_pad_to_square_preserves_center_and_size(self):
        img = Image.new("RGB", (100, 60), color=(255, 0, 0))
        squared = PadToSquare()(img)
        self.assertEqual(squared.size, (100, 100))
        self.assertEqual(squared.getpixel((50, 10)), (0, 0, 0))
        self.assertEqual(squared.getpixel((50, 50)), (255, 0, 0))
        self.assertEqual(squared.getpixel((50, 90)), (0, 0, 0))

    def test_dual_output_wrapper_matches_model(self):
        model = build_model(num_classes=3, pretrained=False)
        model.eval()
        wrapper = DualOutputWrapper(model)
        wrapper.eval()

        x = torch.randn(2, 3, 64, 64)
        with torch.inference_mode():
            direct_logits = model(x)
            f_logits, f_emb = forward_with_embedding(model, x)
            w_logits, w_emb = wrapper(x)

        self.assertEqual(tuple(direct_logits.shape), (2, 3))
        self.assertEqual(tuple(w_emb.shape), (2, EMBEDDING_DIM))
        self.assertTrue(torch.allclose(direct_logits, f_logits, atol=1e-6))
        self.assertTrue(torch.allclose(direct_logits, w_logits, atol=1e-6))
        self.assertTrue(torch.allclose(f_emb, w_emb, atol=1e-6))

    def test_remap_legacy_head_keys(self):
        legacy = {
            "features.0.weight": torch.ones(1),
            "classifier.1.weight": torch.ones(2, 2),
            "classifier.4.bias": torch.zeros(2),
            "classifier.7.weight": torch.ones(3, 2),
            "classifier.7.bias": torch.zeros(3),
        }
        remapped = _remap_legacy_head_keys(legacy)
        self.assertIn("classifier.embed.1.weight", remapped)
        self.assertIn("classifier.embed.4.bias", remapped)
        self.assertIn("classifier.head.1.weight", remapped)
        self.assertIn("classifier.head.1.bias", remapped)

    def test_should_promote_checkpoint_cases(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "model.pth")
            base_meta = {
                "class_names": ["Bacterial Blight", "Healthy"],
                "val_macro_f1": 0.90,
                "val_acc": 90.0,
                "val_fingerprint": "abc12345",
                "subset": 0,
            }

            # 1. No existing checkpoint -> promote
            ok, _ = should_promote_checkpoint(base_meta, ckpt_path)
            self.assertTrue(ok)

            torch.save(base_meta, ckpt_path)

            # 2. Subset run -> refuse promotion unless --force
            subset_meta = {**base_meta, "subset": 10, "val_macro_f1": 0.99}
            ok, _ = should_promote_checkpoint(subset_meta, ckpt_path)
            self.assertFalse(ok)
            ok_forced, _ = should_promote_checkpoint(subset_meta, ckpt_path, force=True)
            self.assertTrue(ok_forced)

            # 3. Class list changed -> refuse promotion without --force
            diff_cls_meta = {**base_meta, "class_names": ["Bacterial Blight", "Blast", "Healthy"], "val_macro_f1": 0.95}
            ok, _ = should_promote_checkpoint(diff_cls_meta, ckpt_path)
            self.assertFalse(ok)

            # 4. Different validation fingerprint -> refuse promotion
            diff_fp_meta = {**base_meta, "val_fingerprint": "different999", "val_macro_f1": 0.95}
            ok, _ = should_promote_checkpoint(diff_fp_meta, ckpt_path)
            self.assertFalse(ok)

            # 5. Comparable run with lower F1 -> refuse
            worse_meta = {**base_meta, "val_macro_f1": 0.85}
            ok, _ = should_promote_checkpoint(worse_meta, ckpt_path)
            self.assertFalse(ok)

            # 6. Comparable run with higher F1 -> promote
            better_meta = {**base_meta, "val_macro_f1": 0.93}
            ok, _ = should_promote_checkpoint(better_meta, ckpt_path)
            self.assertTrue(ok)

    def test_mahalanobis_distances_vectorized_matches_loop(self):
        rng = np.random.RandomState(0)
        emb = rng.randn(10, 4)
        preds = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        class_names = ["A", "B"]
        class_means = {"A": rng.randn(4), "B": rng.randn(4)}
        a = rng.randn(4, 4)
        precision = a.T @ a + np.eye(4)

        vec_dists = _distances(emb, preds, class_means, precision, class_names)
        for i in range(len(emb)):
            mean = class_means[class_names[preds[i]]]
            d = emb[i] - mean
            expected = np.sqrt(float(d @ precision @ d))
            self.assertTrue(np.isclose(vec_dists[i], expected, atol=1e-6))

    def test_calibrate_confidence_threshold(self):
        confs = np.array([0.55, 0.62, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.98])
        preds = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0])
        labels = np.array([1, 1, 1, 1, 1, 0, 1, 0, 1, 0])
        thr = calibrate_confidence_threshold(confs, preds, labels, target_precision=0.95, min_threshold=0.50)
        self.assertTrue(np.isclose(thr, 0.70, atol=1e-4))

    def test_param_groups_separate_decay_and_no_decay(self):
        m = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.BatchNorm1d(4))
        groups = make_param_groups([(m, 1e-3)], weight_decay=1e-4)
        self.assertEqual(len(groups), 2)
        wd_values = {g["weight_decay"] for g in groups}
        self.assertEqual(wd_values, {1e-4, 0.0})

    def test_entropy_threshold_and_dhash(self):
        self.assertGreater(get_entropy_threshold(2), 0.5)
        img = Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8))
        self.assertIsInstance(dhash_image(img), int)


if __name__ == "__main__":
    unittest.main(verbosity=2)
