"""Control handling, RC consistency, and expected-profile conversion."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch


@unittest.skipUnless(importlib.util.find_spec("cherimoya"), "requires regseqkit[cherimoya]")
class InferenceTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)

    def test_expected_signal_promotes_half_precision_and_conserves_counts(self):
        from regseqkit.inference import compute_signal

        logits = torch.zeros(2, 3, 4, dtype=torch.float16, requires_grad=True)
        counts = torch.tensor([[12., 2.], [10., -1.]], dtype=torch.float16)
        signal = compute_signal(logits, counts, [2, 1])
        self.assertEqual(signal.dtype, torch.float32)
        self.assertTrue(signal.isfinite().all())
        torch.testing.assert_close(signal[:, :2].sum((1, 2)), counts[:, 0].float().expm1())
        torch.testing.assert_close(signal[:, 2:].sum((1, 2)), counts[:, 1].float().expm1().clamp(min=0))
        signal[:, 0, 0].sum().backward()
        self.assertTrue(logits.grad.isfinite().all())

    def test_expected_signal_rejects_uncovered_channels(self):
        from regseqkit.inference import compute_signal

        with self.assertRaisesRegex(ValueError, "cover all"):
            compute_signal(torch.zeros(1, 3, 4), torch.zeros(1, 1), [1])

    def test_rc_prediction_aligns_heads_and_native_design_wrapper_preserves_gradients(self):
        from cherimoya import Cherimoya
        from cherimoya.wrappers import ControlWrapper, LogCountWrapper
        from regseqkit.inference import compute_signal
        from regseqkit.cherimoya.inference import predict_cherimoya

        model = Cherimoya(n_filters=8, n_layers=1, signal_groups=[2, 1],
                          trimming=3, compile=False, verbose=False).eval()
        X = torch.rand(3, 4, 32, requires_grad=True)
        scalars, profiles = predict_cherimoya(model, X.detach(), rc_average=True, batch_size=2)
        with torch.no_grad():
            logits, counts = model(X)
            reverse_logits, reverse_counts = model(X.flip((-2, -1)))
        expected_logits = (logits + reverse_logits[:, [1, 0, 2]].flip(-1)) / 2
        expected_counts = (counts + reverse_counts) / 2
        torch.testing.assert_close(scalars, expected_counts)
        torch.testing.assert_close(profiles, compute_signal(expected_logits, expected_counts, [2, 1]))
        module = LogCountWrapper(ControlWrapper(model)).eval()
        torch.testing.assert_close(module(X), counts)
        module(X).sum().backward()
        self.assertTrue(X.grad.isfinite().all())
        self.assertGreater(X.grad.abs().sum().item(), 0)
        for size in (0, -1):
            with self.assertRaisesRegex(ValueError, "batch_size"):
                predict_cherimoya(model, X, batch_size=size)

    def test_controls_follow_native_zero_fallback_and_rc_grouping(self):
        from cherimoya import Cherimoya
        from regseqkit.inference import compute_signal
        from regseqkit.cherimoya.inference import load_checkpoint, predict_cherimoya

        model = Cherimoya(n_filters=8, n_layers=1, n_control_tracks=2,
                          trimming=3, compile=False, verbose=False).eval()
        X, controls = torch.rand(3, 4, 32), torch.rand(3, 2, 32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model.save(root / "model.torch")
            (root / "fit.json").write_text(json.dumps({"in_window": 32, "out_window": 26}))
            loaded, _ = load_checkpoint(root / "model.torch", root / "fit.json", ["a"])
        for groups in (None, [2]):
            scalars, profiles = predict_cherimoya(loaded, X, X_ctl=controls, batch_size=2,
                                        rc_average=True, control_groups=groups)
            reverse_controls = controls.flip(-1) if groups is None else controls.flip((-2, -1))
            with torch.no_grad():
                logits, forward = loaded(X, controls)
                reverse_logits, reverse = loaded(X.flip((-2, -1)), reverse_controls)
            expected_counts = (forward + reverse) / 2
            expected_logits = (logits + reverse_logits.flip(-1)) / 2
            torch.testing.assert_close(scalars, expected_counts)
            torch.testing.assert_close(profiles, compute_signal(expected_logits, expected_counts, [1]))
        implicit_zero = predict_cherimoya(loaded, X, rc_average=True)
        observed_zero = predict_cherimoya(loaded, X, X_ctl=torch.zeros_like(controls), rc_average=True)
        for implicit, observed in zip(implicit_zero, observed_zero):
            torch.testing.assert_close(implicit, observed)
