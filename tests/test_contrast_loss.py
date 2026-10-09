"""The contrast term folded into cherimoya's count loss."""

import unittest

import torch
try:
    import cherimoya.cherimoya
    from cherimoya.losses import _mixture_loss
except ModuleNotFoundError as exc:
    if exc.name == "cherimoya":
        raise unittest.SkipTest("requires regseqkit[cherimoya]") from exc
    raise

from regseqkit.cherimoya.contrast_loss import install, read_block, with_contrast

PAIRS = [(1, 0), (2, 0), (3, 0)]


def batch(n=16, tracks=4, length=50, seed=0):
    generator = torch.Generator().manual_seed(seed)
    y = torch.poisson(torch.rand(n, tracks, length, generator=generator) * 3)
    logits = torch.randn(n, tracks, length, generator=generator)
    logcounts = torch.randn(n, tracks, generator=generator) + 4
    return y, logits, logcounts


class WithContrast(unittest.TestCase):
    def test_zero_weight_is_the_stock_loss(self):
        y, logits, logcounts = batch()
        profile, counts = _mixture_loss(y, logits, logcounts, signal_groups=[1] * 4)
        profile_c, counts_c = with_contrast(PAIRS, 0.0)(y, logits, logcounts, signal_groups=[1] * 4)
        self.assertTrue(torch.equal(profile, profile_c))
        self.assertTrue(torch.equal(counts, counts_c))

    def test_term_is_half_the_pair_contrast_mse_on_each_track(self):
        y, logits, logcounts = batch()
        weight = 10.0
        _, counts = _mixture_loss(y, logits, logcounts, signal_groups=[1] * 4)
        _, counts_c = with_contrast(PAIRS, weight)(y, logits, logcounts, signal_groups=[1] * 4)

        true_log = torch.log(y.sum(-1) + 1)
        expected = torch.zeros(4)
        for us, control in PAIRS:
            delta_true = true_log[:, us] - true_log[:, control]
            delta_pred = logcounts[:, us] - logcounts[:, control]
            mse = ((delta_true - delta_pred) ** 2).mean()
            expected[us] += weight / 2 * mse
            expected[control] += weight / 2 * mse
        torch.testing.assert_close(counts_c - counts, expected)

    def test_gradient_reaches_the_count_head(self):
        y, logits, logcounts = batch()
        logcounts.requires_grad_(True)
        _, counts_c = with_contrast(PAIRS, 10.0)(y, logits, logcounts, signal_groups=[1] * 4)
        counts_c.sum().backward()
        self.assertTrue(torch.isfinite(logcounts.grad).all())
        self.assertGreater(logcounts.grad.abs().sum().item(), 0)

    def test_grouped_contrast_sums_channels_before_log1p(self):
        y, logits, logcounts = batch()
        logcounts = logcounts[:, :2].requires_grad_()
        _, native = _mixture_loss(y, logits, logcounts, signal_groups=[2, 2])
        _, augmented = with_contrast([(1, 0)], 2.0)(
            y, logits, logcounts, signal_groups=[2, 2]
        )
        totals = torch.stack([y[:, :2].sum((1, 2)), y[:, 2:].sum((1, 2))], dim=1)
        residual = totals.log1p() - logcounts
        expected = (residual[:, 1] - residual[:, 0]).square().mean()
        torch.testing.assert_close(augmented - native, expected.expand(2))
        augmented.sum().backward()
        self.assertTrue(torch.isfinite(logcounts.grad).all())

    def test_rejects_indices_outside_grouped_count_heads(self):
        y, logits, logcounts = batch()
        with self.assertRaises(ValueError):
            with_contrast(PAIRS, 1.0)(y, logits, logcounts[:, :2], signal_groups=[2, 2])


class Install(unittest.TestCase):
    def test_rebinds_the_name_the_fit_loop_calls(self):
        original = cherimoya.cherimoya._mixture_loss
        try:
            install(PAIRS, 10.0)
            self.assertIsNot(cherimoya.cherimoya._mixture_loss, original)
            self.assertIs(cherimoya.Cherimoya.fit.__globals__["_mixture_loss"], cherimoya.cherimoya._mixture_loss)
        finally:
            cherimoya.cherimoya._mixture_loss = original


class ReadBlock(unittest.TestCase):
    FIT = {
        "signals": [["/x/plus.bw", "/x/minus.bw"], "/x/a.bw", "/x/b.bw"],
        "contrast": {
            "weight": 10,
            "pairs": [
                [1, 0],
                [2, 0],
            ],
        },
    }

    def test_reads_indices_and_weight(self):
        pairs, weight = read_block(self.FIT)
        self.assertEqual(pairs, [(1, 0), (2, 0)])
        self.assertEqual(weight, 10.0)

    def test_rejects_indices_outside_count_outputs(self):
        fit = {**self.FIT, "contrast": {"weight": 1, "pairs": [[3, 0]]}}
        with self.assertRaises(ValueError):
            read_block(fit)

    def test_reads_legacy_index_dictionaries(self):
        fit = {**self.FIT, "contrast": {"weight": 1, "pairs": [{"indices": [1, 0]}]}}
        self.assertEqual(read_block(fit), ([(1, 0)], 1.0))


if __name__ == "__main__":
    unittest.main()
