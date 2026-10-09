"""Mutation axes, greedy convergence, and optimizer loss validation."""

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tangermeme.saturation_mutagenesis import saturation_mutagenesis

from regseqkit.calibrate import Calibration, resolve_percentile_targets, summarize_predictions
from regseqkit.config import build_objectives, build_scorers
from regseqkit.design import Objective, WeightedObjective, greedy_design, greedy_substitution, ledidi_design
from regseqkit.mutagenesis import single_site_saturation_mutagenesis, ssm_attribution
from regseqkit.scoring import ScalarScorer, ScoreModule

class Counts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batches = []

    def forward(self, X):
        self.batches.append(len(X))
        # Distinct position weights expose swaps in the position/base axes.
        weights = torch.arange(1, X.shape[-1] + 1, device=X.device)
        return (X * weights).sum(-1)


def template(n=1, length=4):
    X = torch.zeros(n, 4, length)
    X[:, 3] = 1
    return X


def objective(mode="max", target=None):
    scorer = ScalarScorer([1, 0, 0, 0])
    goal = Objective(scorer, mode, target)
    return goal


class MutationTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)

    def test_predictions_explicit_mutations_and_batch_bounds(self):
        X = template(3, 5)
        X[1, 3, 1] = 0
        X[1, 1, 1] = 1
        X[2, 3, 4] = 0
        X[2, 0, 4] = 1
        original = X.clone()
        positions = [1, 4]
        expected_reference = Counts()(X)
        expected = torch.empty(3, 2, 4, 4)
        for n in range(3):
            for site, position in enumerate(positions):
                for base in range(4):
                    mutated = X[n:n + 1].clone()
                    mutated[0, :, position] = 0
                    mutated[0, base, position] = 1
                    expected[n, site, base] = Counts()(mutated)[0]
        for batch_size in (1, 5, 128):
            with self.subTest(batch_size=batch_size):
                model = Counts().train()
                reference_predictions, mutant_predictions = single_site_saturation_mutagenesis(
                    model=model, X=X.requires_grad_(), positions=positions,
                    batch_size=batch_size,
                )
                torch.testing.assert_close(reference_predictions, expected_reference)
                torch.testing.assert_close(mutant_predictions, expected)
                self.assertEqual(sum(model.batches), 3 + 3 * 2 * 3)
                self.assertLessEqual(max(model.batches), batch_size)
                self.assertTrue(model.training)
                self.assertFalse(mutant_predictions.requires_grad)
                torch.testing.assert_close(X, original)

    def test_empty_positions_and_validation(self):
        X = template()
        expected = single_site_saturation_mutagenesis(Counts(), X, positions=range(X.shape[-1]))
        for kwargs in ({}, {"positions": None}):
            with self.subTest(kwargs=kwargs):
                reference, mutants = single_site_saturation_mutagenesis(Counts(), X, **kwargs)
                self.assertEqual(mutants.shape, (1, X.shape[-1], 4, 4))
                torch.testing.assert_close(reference, expected[0])
                torch.testing.assert_close(mutants, expected[1])
        base, predictions = single_site_saturation_mutagenesis(Counts(), X, positions=[])
        self.assertEqual(predictions.shape, (1, 0, 4, 4))
        self.assertEqual(base.shape, (1, 4))
        for positions in ([2, 1], [1, 1], [-1], [4], [1.5], [True], [[1]]):
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                single_site_saturation_mutagenesis(Counts(), X, positions=positions)
        for kwargs in ({"batch_size": 0}, {"batch_size": True}):
            with self.assertRaises(ValueError):
                single_site_saturation_mutagenesis(Counts(), X, positions=[1], **kwargs)
        with self.assertRaises(ValueError):
            single_site_saturation_mutagenesis(Counts(), X * 0.5, positions=[1])
        with self.assertRaises(ValueError):
            single_site_saturation_mutagenesis(torch.nn.Identity(), X, positions=[1])

        class Nonfinite(torch.nn.Module):
            def forward(self, X):
                return torch.full((len(X), 1), float("inf"))

        with self.assertRaises(ValueError):
            single_site_saturation_mutagenesis(Nonfinite(), X, positions=[1])

    def test_attribution_matches_upstream_and_preserves_precision(self):
        X = template(2, 5)
        model = Counts()
        baseline, raw = saturation_mutagenesis(
            model, X, start=1, end=4, raw_outputs=True,
            batch_size=5, dtype=torch.float32, device="cpu",
        )
        expected = raw - baseline[:, None, None, :]
        expected -= expected.mean(1, keepdim=True)
        expected = expected.permute(0, 3, 1, 2)
        before = torch.get_float32_matmul_precision()
        actual_baseline, mutants = single_site_saturation_mutagenesis(
            model, X, positions=range(1, 4),
            batch_size=5,
        )
        actual = ssm_attribution(actual_baseline, mutants)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_baseline, baseline)
        self.assertEqual(torch.get_float32_matmul_precision(), before)

    def test_ism_axis_semantics(self):
        """Each axis of (N, scorers, 4, width), pinned analytically."""

        class OnePosition(torch.nn.Module):
            def forward(self, X):
                return torch.stack((X[:, 0, 3], X[:, 1, 5]), dim=-1)

        X = torch.zeros(2, 4, 8)
        X[:, 3] = 1
        baseline, mutants = single_site_saturation_mutagenesis(
            OnePosition(), X, positions=range(2, 6),
            batch_size=16,
        )
        attr = ssm_attribution(baseline, mutants)
        self.assertEqual(tuple(attr.shape), (2, 2, 4, 4))
        self.assertEqual(int(attr[0, 0].abs().sum(0).argmax()), 3 - 2)
        self.assertEqual(int(attr[0, 1].abs().sum(0).argmax()), 5 - 2)
        self.assertEqual(int(attr[0, 0, :, 1].argmax()), 0)
        self.assertEqual(int(attr[0, 1, :, 3].argmax()), 1)
        torch.testing.assert_close(attr.sum(dim=2), torch.zeros(2, 2, 4))
        self.assertAlmostEqual(float(attr[0, 0, 3, 1]), -0.25, places=6)


class GreedyTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)

    def test_position_selection_and_validation(self):
        goal, X = objective(), template()
        stack = ScoreModule(Counts(), goal.scorers)
        for positions in ([0, 2], torch.tensor([0, 2])):
            with self.subTest(positions=positions):
                records = []
                result = greedy_substitution(
                    stack, X, goal, positions=positions, on_iteration=records.append,
                )
                torch.testing.assert_close(result[:, :, [1, 3]], X[:, :, [1, 3]])
                self.assertEqual(result.argmax(1).tolist(), [[0, 3, 0, 3]])
                self.assertEqual(records[0].candidate_scores.shape, (2, 4, 1))
        result = greedy_substitution(stack, X, goal)
        self.assertEqual(result.argmax(1).tolist(), [[0, 0, 0, 0]])
        for positions in ([2, 0], [1, 1], [-1], [4], [1.5], [[1]], [False, True]):
            with self.subTest(positions=positions), self.assertRaisesRegex(ValueError, "positions"):
                greedy_substitution(stack, X, goal, positions=positions)

    def test_reference_predictions_selection_and_stopping_round(self):
        model, X, goal = Counts(), template(), objective()
        stack = ScoreModule(model, goal.scorers)
        records = []
        result = greedy_substitution(
            stack, X, goal, positions=[1, 2],
            max_iter=5, batch_size=2, on_iteration=records.append,
        )
        self.assertIs(greedy_design, greedy_substitution)
        self.assertEqual([r.mutation for r in records], ["2T>A", "1T>A", None])
        self.assertEqual([r.accepted for r in records], [True, True, False])
        self.assertEqual(records[-1].stop_reason, "insufficient_improvement")
        self.assertEqual(sum(model.batches), 1 + 3 * (1 + 2 * 3))
        torch.testing.assert_close(result[:, :, [0, 3]], X[:, :, [0, 3]])
        for record in records:
            self.assertEqual(record.candidate_scores.shape, (2, 4, 1))
            calculated = goal(record.candidate_scores.reshape(-1, 1)).reshape(2, 4)
            torch.testing.assert_close(record.candidate_losses, calculated)
            positions = torch.tensor([1, 2])
            references = record.current_sequence[0].argmax(0)[positions]
            selected = calculated.clone()
            selected[torch.arange(2), references] = torch.inf
            site, base = divmod(int(selected.flatten().argmin()), 4)
            if record.accepted:
                position = int(positions[site])
                self.assertEqual(record.mutation, f"{position}{'ACGT'[references[site]]}>{'ACGT'[base]}")
                expected = record.current_sequence.clone()
                expected[0, :, position] = 0
                expected[0, base, position] = 1
                torch.testing.assert_close(record.selected_sequence, expected)
            else:
                self.assertIsNone(record.mutation)
                torch.testing.assert_close(record.selected_sequence, record.current_sequence)
                torch.testing.assert_close(record.selected_scores, record.current_scores)
                self.assertEqual(record.selected_loss, record.current_loss)
            self.assertEqual(record.selected_loss, float(goal(record.selected_scores)[0]))
            self.assertFalse(record.selected_sequence.requires_grad)
        torch.testing.assert_close(records[0].selected_scores, records[1].current_scores)

    def test_ties_tolerance_zero_and_iteration_budget(self):
        class FlatWeights(torch.nn.Module):
            def forward(self, X):
                return X.sum(-1)

        goal, X = objective(), template()
        stack = ScoreModule(FlatWeights(), goal.scorers)
        positions = range(4)
        records = []
        result = greedy_substitution(stack, X, goal, positions=positions, max_iter=1, on_iteration=records.append)
        self.assertEqual(records[0].mutation, "0T>A")
        self.assertEqual(records[0].stop_reason, "max_iter")
        self.assertEqual(len(records), 1)
        # Improvement exactly equal to tol must not be accepted.
        records.clear()
        rejected = greedy_substitution(stack, X, goal, positions=positions, tol=1, on_iteration=records.append)
        torch.testing.assert_close(rejected, X)
        self.assertFalse(records[0].accepted)
        self.assertIsNone(records[0].mutation)
        records.clear()
        torch.testing.assert_close(greedy_substitution(stack, X, goal, positions=positions, max_iter=0,
                                   on_iteration=records.append), X)
        self.assertEqual(records, [])
        torch.testing.assert_close(greedy_substitution(stack, X, goal, positions=[]), X)
        self.assertEqual(int(result[:, 0].sum()), 1)
        # Removing an A has equally good C/G/T choices; ACGT order picks C.
        minimum = objective("min")
        records.clear()
        greedy_substitution(stack, result, minimum, positions=positions, max_iter=1, on_iteration=records.append)
        self.assertEqual(records[0].mutation, "0A>C")

    def test_match_hold_and_repeated_edits(self):
        goal, X = objective("match", 3), template()
        stack = ScoreModule(Counts(), goal.scorers)
        result = greedy_substitution(stack, X, goal)
        self.assertEqual(float(stack(result)[0, 0]), 3)
        a, c = ScalarScorer([1, 0, 0, 0]), ScalarScorer([0, 1, 0, 0])
        X[0, 3, 3], X[0, 1, 3] = 0, 1
        stack = ScoreModule(Counts(), [a, c])
        reference_value = float(stack(X)[0, 1])
        held = WeightedObjective((Objective(a, "max"), Objective(c, "match", reference_value)), weights=(1, 100))
        result = greedy_substitution(stack, X, held)
        self.assertEqual(float(stack(result)[0, 1]), float(stack(X)[0, 1]))

        class Landscape(torch.nn.Module):
            def forward(self, X):
                # T,T -> A,T -> A,C -> G,C is the strictly improving path.
                score = (X[:, 0, 0] + 2 * X[:, 0, 0] * X[:, 1, 1]
                         + 4 * X[:, 2, 0] * X[:, 1, 1])
                return torch.stack([score, score * 0, score * 0, score * 0], dim=1)

        goal, records = objective(), []
        result = greedy_substitution(ScoreModule(Landscape(), goal.scorers), template(length=2),
                                    goal, positions=range(2), max_iter=3,
                                    on_iteration=records.append)
        self.assertEqual([r.mutation for r in records], ["0T>A", "1T>C", "0A>G"])
        self.assertEqual(result.argmax(1).tolist(), [[2, 1]])


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)

    def test_build_scorers_from_entries(self):
        entries = [
            {"name": "delta", "weights": {"A": 1.0, "C": -1.0}},
            {"name": "activity", "weights": {"A": 1.0}},
        ]
        scorers = build_scorers(entries, ["A", "C"])
        self.assertEqual(list(scorers), ["delta", "activity"])
        self.assertTrue(all(isinstance(s, ScalarScorer) for s in scorers.values()))
        torch.testing.assert_close(scorers["delta"].weights, torch.tensor([1.0, -1.0]))

        for field, value in (("head", "profile"), ("absolute", True), ("reduction", "sum")):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "model wrapper"):
                build_scorers([dict(entries[0], **{field: value})], ["A", "C"])

        with self.assertRaisesRegex(ValueError, "do not convert outputs"):
            build_scorers([dict(entries[0], units="log")], ["A", "C"])
        with self.assertRaises(ValueError):  # unknown output
            build_scorers([{"name": "x", "weights": {"Z": 1.0}}], ["A", "C"])
        with self.assertRaises(ValueError):  # duplicate name
            build_scorers(entries + entries[:1], ["A", "C"])

        # standardize reaches scalar scorers and an entry can opt out.
        center, scale = np.zeros(2), np.ones(2) * 2
        both = build_scorers(
            [{"name": "a", "weights": {"A": 1.0}},
             {"name": "b", "weights": {"A": 1.0}, "standardize": False}],
            ["A", "C"],
            standardize=(center, scale),
        )
        self.assertIsNotNone(both["a"].standardize)
        self.assertIsNone(both["b"].standardize)

    def test_build_objectives(self):
        scorers = {"A": ScalarScorer([1, 0]), "C": ScalarScorer([0, 1])}
        goals = build_objectives(
            [
                # A standalone objective needs no weighted wrapper.
                {"name": "up", "mode": "max", "scorer": "A"},
                {
                    "name": "up_hold",
                    "objectives": [
                        {"scorer": "A", "mode": "max"},
                        {"scorer": "C", "mode": "match", "target": "template", "weight": 0.5},
                    ],
                },
            ],
            scorers,
            template_scores={"C": 2.0},
        )
        self.assertEqual(list(goals), ["up", "up_hold"])
        self.assertIsInstance(goals["up"], Objective)
        self.assertIsInstance(goals["up_hold"], WeightedObjective)
        self.assertIs(goals["up"].scorers[0], scorers["A"])
        self.assertIs(goals["up_hold"].scorers[0], scorers["A"])
        self.assertIs(goals["up_hold"].scorers[1], scorers["C"])
        self.assertEqual([t.mode for t in goals["up_hold"].objectives], ["maximize", "match"])
        self.assertEqual(goals["up_hold"].objectives[1].target, 2.0)
        self.assertEqual(goals["up_hold"].weights, (1.0, 0.5))
        with self.assertRaisesRegex(ValueError, "missing template score"):
            build_objectives([{"name": "x", "mode": "match", "scorer": "C",
                               "target": "template"}], scorers)
        with self.assertRaisesRegex(ValueError, "requires mode 'match'"):
            build_objectives([{"name": "x", "mode": "max", "scorer": "C",
                               "target": "template"}], scorers, template_scores={"C": 2.0})
        with self.assertRaises(KeyError):
            build_objectives([{"name": "x", "mode": "max", "scorer": "Z"}], scorers)
        with self.assertRaises(ValueError):  # duplicate id
            build_objectives([{"name": "x", "mode": "max", "scorer": "A"}] * 2, scorers)
        with self.assertRaises(ValueError):  # no sub-objectives and no scorer
            build_objectives([{"name": "x", "mode": "max"}], scorers)

    def test_nested_config_resolves_targets_and_preserves_single_weight(self):
        scorers = {"A": ScalarScorer([1, 0]), "C": ScalarScorer([0, 1])}
        calibration = Calibration(references={"peaks": summarize_predictions(
            np.stack([np.arange(101.), np.arange(101.) * 2], axis=1), ["A", "C"]
        )})
        entries = [{"name": "nested", "objectives": [
            {"scorer": "A", "mode": "max", "weight": 2.},
            {"weight": 3., "objectives": [
                {"scorer": "A", "mode": "match", "target": "template", "weight": 0.25},
                {"scorer": "C", "mode": "match", "target": "p50", "weight": 0.5},
            ]},
        ]}]
        resolved = resolve_percentile_targets(entries, calibration, "peaks")
        goal = build_objectives(resolved, scorers, template_scores={"A": 3.})["nested"]
        self.assertEqual(len(goal.scorers), 2)
        torch.testing.assert_close(goal(torch.tensor([[4., 102.]])), torch.tensor([-1.25]))
        self.assertEqual(entries[0]["objectives"][1]["objectives"][1]["target"], "p50")
        single = build_objectives(
            [{"name": "scaled", "scorer": "A", "mode": "min", "weight": 0.5}], scorers,
        )["scaled"]
        self.assertIsInstance(single, WeightedObjective)
        self.assertEqual(single.objectives[0].mode, "minimize")
        torch.testing.assert_close(single(torch.tensor([[4.]])), torch.tensor([2.]))
        for entry in (
            {"name": "empty", "objectives": []},
            {"name": "ambiguous", "scorer": "A", "mode": "max", "objectives": []},
            {"name": "old", "scorer": "A", "mode": "match", "value": 1.},
        ):
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                build_objectives([entry], scorers)


class OptimizerValidationTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)

    def test_optimizers_reject_invalid_custom_losses(self):
        scorer = ScalarScorer([1.])

        class Model(torch.nn.Module):
            def forward(self, X):
                return X[:, 0].sum(-1, keepdim=True)

        class CustomObjective:
            scorers = (scorer,)

            def __init__(self, calculate):
                self.calculate = calculate

            def __call__(self, scores):
                return self.calculate(scores)

        template = torch.zeros(1, 4, 2)
        template[:, 3] = 1
        module = ScoreModule(Model(), [scorer])

        def run_ledidi(model, X, y_bar, **kwargs):
            # Exercise the adapter's actual loss callback without a training run.
            kwargs["output_loss"](model(X), y_bar)
            return X

        invalid = (
            (lambda scores: torch.ones(len(scores), 1), "one loss per example"),
            (lambda scores: torch.full((len(scores),), float("nan")), "non-finite"),
            (lambda scores: 1., "one loss per example"),
        )
        for calculate, message in invalid:
            objective = CustomObjective(calculate)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                greedy_substitution(module, template, objective, max_iter=0)
            if importlib.util.find_spec("ledidi"):
                with patch("ledidi.ledidi", side_effect=run_ledidi), self.assertRaisesRegex(ValueError, message):
                    ledidi_design(module, template, objective)
        objective = CustomObjective(lambda scores: -scores[:, 0])
        designed = greedy_substitution(module, template, objective, positions=[1], max_iter=1)
        self.assertEqual(designed.argmax(1).tolist(), [[3, 0]])


class LedidiTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("ledidi"), "requires regseqkit[ledidi]")
    def test_ledidi_sample_count_and_seed(self):
        class Counts(torch.nn.Module):
            def forward(self, X):
                return X[:, 0].sum(-1, keepdim=True)

        template = torch.zeros(1, 4, 12)
        template[:, 3] = 1
        scorer = ScalarScorer(torch.ones(1))
        module = ScoreModule(Counts(), [scorer])
        objective = Objective(scorer, "maximize")
        mask = torch.zeros(12, dtype=torch.bool)
        mask[2:10] = True
        settings = dict(max_iter=2, batch_size=4, n_samples=11, random_state=7)
        first = ledidi_design(module, template, objective, positions=range(2, 10), **settings)
        second = ledidi_design(module, template, objective, positions=range(2, 10), **settings)
        self.assertEqual(first.shape, (11, 4, 12))
        torch.testing.assert_close(first[:, :, ~mask], template[:, :, ~mask].expand(11, -1, -1))
        self.assertTrue(torch.all(first.sum(1) == 1))
        self.assertTrue(torch.all((first == 0) | (first == 1)))
        torch.testing.assert_close(first, second)
        for value in (0, -1, 2.5, True):
            with self.assertRaisesRegex(ValueError, "n_samples"):
                ledidi_design(module, template, objective, positions=range(2, 10), n_samples=value)

    def setUp(self):
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)

    @unittest.skipUnless(importlib.util.find_spec("ledidi"), "requires regseqkit[ledidi]")
    def test_ledidi_position_selection_and_validation(self):
        class Counts(torch.nn.Module):
            def forward(self, X):
                return X[:, 0].sum(-1, keepdim=True)

        template = torch.zeros(1, 4, 4)
        template[:, 3] = 1
        scorer = ScalarScorer(torch.ones(1))
        module = ScoreModule(Counts(), [scorer])
        objective = Objective(scorer, "maximize")
        for positions in ([0, 2], torch.tensor([0, 2]), None, []):
            with self.subTest(positions=positions):
                result = ledidi_design(
                    module, template, objective, positions=positions,
                    max_iter=2, batch_size=2, n_samples=3, random_state=7,
                )
                mask = torch.zeros(4, dtype=torch.bool)
                mask[range(4) if positions is None else positions] = True
                torch.testing.assert_close(result[:, :, ~mask], template[:, :, ~mask].expand(3, -1, -1))
                self.assertTrue(torch.all(result.sum(1) == 1))
                self.assertTrue(torch.all((result == 0) | (result == 1)))
                self.assertEqual(result.shape, (3, 4, 4))
        for positions in ([2, 0], [1, 1], [-1], [4], [1.5], [[1]], [False, True]):
            with self.subTest(positions=positions), self.assertRaisesRegex(ValueError, "positions"):
                ledidi_design(module, template, objective, positions=positions)


if __name__ == "__main__":
    unittest.main()
