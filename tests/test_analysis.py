"""Sequence and artifact round trips, calibration, metrics, and plotting."""

from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from regseqkit.calibrate import Calibration, Distribution, resolve_percentile_targets, summarize_predictions
from regseqkit.figures import plot_density_scatter, plot_profiles, subplots_with_plot_size
from regseqkit.io import load_arrays, save_arrays
from regseqkit.metrics import pearson_correlation, spearman_correlation, pool_channels, profile_metrics, scalar_metrics, window_mask
from regseqkit.sequences import OneHotEncoder, extract_loci_with_coords, read_fasta, read_npz, reverse_complement

class SequenceAndArtifactTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)

    def test_reverse_complement_strings_and_custom_mapping(self):
        self.assertEqual(reverse_complement("AACGN"), "NCGTT")
        self.assertEqual(reverse_complement(""), "")
        rna = {"A": "U", "U": "A", "C": "G", "G": "C"}
        self.assertEqual(reverse_complement("AACGU", complement_map=rna), "ACGUU")

    def test_reverse_complement_batched_tensors_in_custom_channel_order(self):
        encoder = OneHotEncoder(alphabet="ATCG")
        mapping = {"A": "T", "T": "A", "C": "G", "G": "C"}
        original = encoder.to_onehot(["AACGN", "TTCAN"])
        expected = encoder.to_onehot(["NCGTT", "NTGAA"])
        complemented = reverse_complement(original, complement_map=mapping)
        torch.testing.assert_close(complemented, expected)
        self.assertEqual(complemented.dtype, original.dtype)
        self.assertEqual(complemented.device, original.device)
        torch.testing.assert_close(reverse_complement(original[0], mapping), expected[0])
        torch.testing.assert_close(reverse_complement(complemented, mapping), original)
        torch.testing.assert_close(
            reverse_complement(original.unsqueeze(0), mapping), expected.unsqueeze(0)
        )
        self.assertEqual(encoder.from_onehot(original), ["AACGN", "TTCAN"])

    def test_reverse_complement_preserves_tensor_gradients(self):
        values = torch.rand(2, 4, 7, requires_grad=True)
        reverse_complement(values)[:, 0, 0].sum().backward()
        expected = torch.zeros_like(values)
        expected[:, 3, -1] = 1
        torch.testing.assert_close(values.grad, expected)

    def test_reverse_complement_rejects_mismatched_channels(self):
        with self.assertRaisesRegex(ValueError, "channel count"):
            reverse_complement(torch.zeros(2, 3, 5))

    def test_onehot_encoder_round_trip_and_unknowns(self):
        encoder = OneHotEncoder()
        onehot = encoder.to_onehot(["ACGT", "AANT"])
        self.assertEqual(onehot.shape, (2, 4, 4))
        self.assertEqual(onehot.dtype, torch.int8)
        # A base outside the alphabet is an all-zero column, not an error.
        self.assertEqual(int(onehot[1, :, 2].sum()), 0)
        self.assertEqual(encoder.from_onehot(onehot), ["ACGT", "AANT"])
        single = encoder.to_onehot("ACGT")
        self.assertEqual(single.shape, (4, 4))
        torch.testing.assert_close(single, torch.eye(4, dtype=torch.int8))

    def test_tensor_encoder_respects_explicit_layout_and_dtype(self):
        for axis in (1, -1, 2):
            encoder = OneHotEncoder(alphabet="ATCG", channel_axis=axis, dtype=torch.float32)
            encoded = encoder.to_onehot(["AGTT", "NCAT"])
            self.assertEqual(encoded.dtype, torch.float32)
            self.assertEqual(encoded.device.type, "cpu")
            self.assertEqual(encoder.from_onehot(encoded.requires_grad_()), ["AGTT", "NCAT"])
            self.assertEqual(encoder.from_onehot(encoded[0]), "AGTT")
            self.assertIsNone(encoded.grad)
            one_a = encoder.to_onehot("A")
            expected = torch.tensor([[1., 0., 0., 0.]])
            torch.testing.assert_close(one_a, expected.T if axis == 1 else expected)

    def test_tensor_encoder_handles_padding_and_empty_sequences(self):
        encoder = OneHotEncoder(dtype=torch.bool)
        encoded = encoder.to_onehot(["ac", "N"])
        self.assertEqual(encoded.dtype, torch.bool)
        self.assertEqual(encoder.from_onehot(encoded), ["AC", "NN"])
        self.assertEqual(encoder.from_onehot(encoder.to_onehot("")), "")
        self.assertEqual(encoder.from_onehot(encoder.to_onehot(["", ""])), ["", ""])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_tensor_encoder_decodes_on_cuda(self):
        encoder = OneHotEncoder()
        self.assertEqual(encoder.from_onehot(encoder.to_onehot(["AGTT", "NCAT"]).cuda()),
                         ["AGTT", "NCAT"])

    def test_sequence_readers_return_tensors_and_keep_coordinate_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fasta = root / "genome.fa"
            fasta.write_text(">chr1\nAAAAANNNNCCCCCGGGGGTTTTT\n")
            bed = root / "loci.bed"
            bed.write_text("chr1\t1\t5\nchr1\t5\t9\nchr1\t10\t14\n")
            extracted = extract_loci_with_coords(
                bed, fasta, in_window=4, out_window=4, drop_ambiguous=True,
            )
            self.assertIsInstance(extracted.onehot, torch.Tensor)
            self.assertEqual(extracted.onehot.dtype, torch.int8)
            self.assertEqual(OneHotEncoder().from_onehot(extracted.onehot), ["AAAA", "CCCC"])
            self.assertEqual(extracted.coords.source_row.tolist(), [0, 2])
            self.assertIsNone(extracted.signals)
            templates = root / "templates.fa"
            templates.write_text(">a\nAGTT\n>b\nNCAT\n")
            encoded, names = read_fasta(templates, length=4)
            self.assertIsInstance(encoded, torch.Tensor)
            self.assertEqual(OneHotEncoder().from_onehot(encoded), ["AGTT", "NCAT"])
            np.testing.assert_array_equal(names, ["a", "b"])

    def test_light_modules_import_without_torch(self):
        code = (
            "import sys, regseqkit.io, regseqkit.metrics; "
            "heavy = [m for m in ('torch', 'tangermeme', 'cherimoya') if m in sys.modules]; "
            "print(heavy)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "[]")

    def test_artifact_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_arrays(root / "saved.a.b", {"ids": np.array(["x"]), "observed": np.ones((1, 1))}, {})
            self.assertTrue((root / "saved.a.b.npz").is_file())  # appended, not substituted
            arrays, metadata = load_arrays(root / "saved.a.b.json")
            np.testing.assert_array_equal(arrays["ids"], ["x"])
            np.testing.assert_array_equal(arrays["observed"], np.ones((1, 1)))
            self.assertEqual(metadata, {})

    def test_read_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "examples.npz"
            np.savez(path, onehot=np.zeros((2, 4, 8), np.int8), ids=np.array(["a", "b"]),
                     output_names=np.array(["A", "C"]), observed=np.zeros((2, 2)))
            onehot, arrays, outputs = read_npz(path)
            self.assertIsInstance(onehot, torch.Tensor)
            self.assertEqual(onehot.dtype, torch.int8)
            self.assertEqual(onehot.shape, (2, 4, 8))
            self.assertTrue(all(isinstance(array, np.ndarray) for array in arrays.values()))
            self.assertEqual(outputs, ["A", "C"])
            self.assertEqual(sorted(arrays), ["ids", "observed"])


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)

    def test_distribution_summary_and_lookups(self):
        values = np.arange(101, dtype=float)  # 0..100, so pNN is NN
        d = Distribution.summarize(values)
        self.assertEqual(d.n, 101)
        self.assertAlmostEqual(d.mean, 50.0)
        self.assertAlmostEqual(d.value_at(90), 90.0, places=6)
        np.testing.assert_allclose(d.percentile_of(np.array([25.0, 75.0])), [25.0, 75.0], atol=1e-6)
        # Beyond the observed range it clamps rather than extrapolating.
        self.assertEqual(float(d.percentile_of(np.array([1e9]))[0]), 100.0)
        with self.assertRaises(ValueError):
            d.value_at(101)
        with self.assertRaises(ValueError):  # one sample has no spread
            Distribution.summarize(np.array([1.0]))

    def test_calibration_round_trip_and_center_scale(self):
        rng = np.random.default_rng(0)
        predicted = rng.normal(loc=[3.0, -1.0], scale=[2.0, 0.5], size=(4000, 2))
        references = {"negatives": summarize_predictions(predicted, ["A", "C"])}
        calibration = Calibration(references=references, provenance={"source": "test"})

        center, scale = calibration.center_scale("negatives", ["A", "C"])
        np.testing.assert_allclose(center, [3.0, -1.0], atol=0.1)
        np.testing.assert_allclose(scale, [2.0, 0.5], atol=0.1)
        # Standardizing the reference set puts it on zero mean and unit spread,
        # which is what makes a term weight mean what it says.
        standardized = (predicted - center) / scale
        np.testing.assert_allclose(standardized.mean(axis=0), [0, 0], atol=1e-9)
        np.testing.assert_allclose(standardized.std(axis=0, ddof=1), [1, 1], atol=1e-9)

        with tempfile.TemporaryDirectory() as tmp:
            path = calibration.save(Path(tmp) / "calibration.json")
            again = Calibration.load(path)
        self.assertEqual(again.outputs("negatives"), ["A", "C"])
        self.assertAlmostEqual(
            again.distribution("negatives", "A").sd,
            calibration.distribution("negatives", "A").sd,
        )
        with self.assertRaises(KeyError):
            calibration.center_scale("peaks", ["A"])
        with self.assertRaises(KeyError):
            calibration.distribution("negatives", "Z")

    def test_resolve_percentile_targets(self):
        references = {"peaks": summarize_predictions(
            np.stack([np.arange(101.0), np.arange(101.0) * 2], axis=1), ["A", "C"]
        )}
        calibration = Calibration(references=references)
        entries = [
            {"name": "x", "objectives": [
                {"scorer": "A", "mode": "max"},
                {"scorer": "C", "mode": "match", "target": "p50"},
            ]}
        ]
        resolved = resolve_percentile_targets(entries, calibration, "peaks")
        self.assertNotIn("target", resolved[0]["objectives"][0])
        self.assertAlmostEqual(resolved[0]["objectives"][1]["target"], 100.0, places=6)
        # A numeric value passes through untouched.
        numeric = resolve_percentile_targets(
            [{"name": "y", "objectives": [{"scorer": "A", "mode": "match", "target": 3.0}]}],
            calibration, "peaks",
        )
        self.assertEqual(numeric[0]["objectives"][0]["target"], 3.0)
        # A standalone target resolves without adding a composition.
        short = resolve_percentile_targets(
            [{"name": "s", "scorer": "C", "mode": "match", "target": "p50"}], calibration, "peaks"
        )
        self.assertAlmostEqual(short[0]["target"], 100.0, places=6)
        self.assertNotIn("objectives", short[0])
        with self.assertRaises(ValueError):  # a combined scorer has no distribution
            resolve_percentile_targets(
                [{"name": "z", "objectives": [{"scorer": "delta", "mode": "match", "target": "p90"}]}],
                calibration, "peaks",
            )
        with self.assertRaises(KeyError):  # nor does an unknown reference set
            resolve_percentile_targets(entries, calibration, "negatives")


class MetricTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)

    @unittest.skipUnless(importlib.util.find_spec("cherimoya"), "requires regseqkit[cherimoya]")
    def test_metrics_reference(self):
        from cherimoya.performance import calculate_performance_measures

        rng = np.random.default_rng(4)
        logits = torch.tensor(rng.normal(size=(3, 2, 12)), dtype=torch.float32)
        y = torch.tensor(rng.integers(0, 5, size=(3, 2, 12)), dtype=torch.float32)
        for smooth_p, smooth_t, kw in ((False, False, {}), (True, True, dict(kernel_width=5, kernel_sigma=1)),
                                       (False, True, dict(kernel_width=5, kernel_sigma=1))):
            reference = calculate_performance_measures(
                logits, y, torch.zeros(3, 2),
                measures=["profile_mnll", "profile_jsd", "profile_pearson"],
                signal_groups=[1, 1], smooth_predictions=smooth_p, smooth_true=smooth_t, **kw,
            )
            metrics = profile_metrics(
                y.numpy(), logits.softmax(-1).numpy(),
                smooth_predictions=smooth_p, smooth_true=smooth_t, **kw,
            )
            for key in reference:
                np.testing.assert_allclose(metrics[key], reference[key].numpy(), rtol=2e-5, atol=1e-5)

    def test_correlations_are_reusable(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(float(pearson_correlation(x, 2 * x + 1)), 1.0, places=9)
        # Monotone but not linear: Spearman is 1 where Pearson is not.
        y = np.array([1.0, 2.0, 4.0, 40.0])
        self.assertAlmostEqual(float(spearman_correlation(x, y)), 1.0, places=9)
        self.assertLess(float(pearson_correlation(x, y)), 0.95)
        # Ties take average ranks, so a constant vector is undefined, not zero.
        self.assertTrue(np.isnan(float(spearman_correlation(x, np.ones(4)))))
        self.assertEqual(float(spearman_correlation(x, np.ones(4), constant=0.0)), 0.0)
        # Both reduce the chosen axis.
        pair = np.stack([x, x[::-1]])
        self.assertEqual(pearson_correlation(pair, pair).shape, (2,))

    def test_metric_edge_cases(self):
        result = scalar_metrics(
            np.array([[1.0], [1.0], [np.nan]]), np.array([[2.0], [3.0], [4.0]]), ["x"]
        )[0]
        self.assertTrue(np.isnan(result["pearson"]))
        self.assertEqual(result["n_excluded"], 1)
        result = profile_metrics(np.zeros((1, 1, 3)), np.ones((1, 1, 3)))
        self.assertTrue(np.isnan(result["profile_jsd"]).all())
        self.assertEqual(result["profile_pearson"].item(), 0)
        np.testing.assert_array_equal(pool_channels(np.array([[1, 2, 4]]), [2, 1]), [[3, 4]])

    def test_window_mask_central_and_peak_body(self):
        import pandas as pd

        coords = pd.DataFrame({"start": [100, 200], "end": [104, 210]})
        self.assertTrue(window_mask(coords, 8).all())
        np.testing.assert_array_equal(window_mask(coords, 8, central=4)[0], [0, 0, 1, 1, 1, 1, 0, 0])
        with self.assertRaises(ValueError):
            window_mask(coords, 8, central=3)
        body = window_mask(coords, 8, peak_body=True)
        self.assertEqual(int(body[0].sum()), 4)
        self.assertEqual(int(body[1].sum()), 8)


class FigureTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)

    def test_fixed_plot_dimensions_and_overrides(self):
        fig, axes = subplots_with_plot_size(2, 3, (2.0, 1.5), extra_space={"right": 1.2}, wspace=0.7)
        fig.canvas.draw()
        for ax in axes.flat:
            np.testing.assert_allclose(ax.get_window_extent().size / fig.dpi, [2, 1.5], atol=1e-10)
        self.assertIs(plot_profiles([1, 2], [2, 1], ax=axes[0, 0], linewidth=3)[1], axes[0, 0])
        axes[1, 0].bar(["a"], [2], color="red")
        plot_density_scatter(np.arange(10), np.arange(10), ax=axes[0, 1], s=8)
        plt.close(fig)

    def test_smooth_averages_only_the_covered_positions(self):
        from regseqkit.figures import smooth

        # A flat trace stays flat to the edges: no dip where the kernel runs off the end.
        np.testing.assert_allclose(smooth(np.ones(10), 5), np.ones(10))
        np.testing.assert_allclose(smooth([0.0, 0.0, 3.0, 0.0, 0.0], 3), [0, 1, 1, 1, 0])
        self.assertEqual(smooth(np.arange(4.0), 1).tolist(), [0, 1, 2, 3])
        self.assertEqual(smooth(np.zeros((2, 3, 7)), 3).shape, (2, 3, 7))


class CherimoyaAdapterTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)

    @unittest.skipUnless(importlib.util.find_spec("cherimoya"), "requires regseqkit[cherimoya]")
    def test_cherimoya_checkpoint_groups_rc_and_gradients(self):
        from cherimoya import Cherimoya
        from cherimoya.wrappers import ControlWrapper, ExpectedCountsWrapper, LogCountWrapper

        from regseqkit.inference import compute_signal
        from regseqkit.cherimoya.inference import load_checkpoint, predict_cherimoya

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = Cherimoya(n_filters=8, n_layers=1, signal_groups=[2, 1], trimming=3,
                              compile=False, verbose=False).eval()
            model.save(root / "model.torch")
            (root / "model.fit.json").write_text(json.dumps(dict(
                in_window=32, out_window=26, n_filters=8, signals=[["plus.bw", "minus.bw"], "third.bw"],
            )))
            loaded, information = load_checkpoint(
                root / "model.torch", root / "model.fit.json", ["pair", "other"],
                device="cpu",
            )
            self.assertEqual(information["groups"], (2, 1))
            rng = torch.Generator().manual_seed(3)
            X = torch.nn.functional.one_hot(torch.randint(4, (3, 32), generator=rng), 4).permute(0, 2, 1).float()
            scalars, profiles = predict_cherimoya(loaded, X, batch_size=2, rc_average=True)
            with torch.no_grad():
                logits, counts = model(X)
                reverse, reverse_counts = model(X.flip((-2, -1)))
                averaged = (logits + reverse[:, [1, 0, 2]].flip(-1)) / 2
                torch.testing.assert_close(scalars, (counts + reverse_counts) / 2)
                expected_pair = averaged[:, :2].flatten(1).softmax(-1).reshape(3, 2, 26)
                expected_pair *= scalars[:, 0].expm1().clamp(min=0)[:, None, None]
                torch.testing.assert_close(profiles[:, :2], expected_pair)
                torch.testing.assert_close(
                    compute_signal(*loaded(X), loaded.signal_groups),
                    ExpectedCountsWrapper(model)(X).clamp(min=0)
                )
            X.requires_grad_()
            LogCountWrapper(ControlWrapper(loaded))(X).sum().backward()
            self.assertTrue(torch.isfinite(X.grad).all())


class MotifExportTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("modiscolite"), "requires regseqkit[motifs]")
    def test_exports_both_pattern_signs(self):
        import h5py
        from tangermeme.io import read_meme

        from regseqkit.motifs import export_matrices

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with h5py.File(root / "patterns.h5", "w") as f:
                for category in ("pos_patterns", "neg_patterns"):
                    group = f.create_group(f"{category}/pattern_0")
                    group["sequence"] = np.eye(4)
                    group["contrib_scores"] = np.eye(4) * (-1 if category.startswith("neg") else 1)
                    seqlets = group.create_group("seqlets")
                    for name, data in dict(example_idx=[0], start=[2], end=[5], is_revcomp=[False]).items():
                        seqlets[name] = data
            export_matrices(root / "patterns.h5", root / "motifs")
            self.assertEqual(len(read_meme(root / "motifs.ppm.meme")), 2)


if __name__ == "__main__":
    unittest.main()
