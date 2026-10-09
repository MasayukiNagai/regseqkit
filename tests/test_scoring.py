"""Scoring, output conversion, and objective values and gradients."""

import unittest
import torch

from regseqkit.design import Objective, WeightedObjective
from regseqkit.scoring import ScalarScorer, ScoreModule
from regseqkit.inference import log1p_to_counts

class ScoringTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)

    def test_conversion_and_gradients(self):
        values = torch.log1p(torch.tensor([[10., 4.], [3., 7.]])).requires_grad_()
        scorer = ScalarScorer([1., -1.])
        module = ScoreModule(torch.nn.Identity(), [scorer])
        actual = module(log1p_to_counts(values))
        torch.testing.assert_close(actual, torch.tensor([[6.], [-4.]]))
        actual.sum().backward()
        torch.testing.assert_close(values.grad, values.detach().exp() * torch.tensor([1., -1.]))
        self.assertEqual(actual.dtype, values.dtype)
        self.assertEqual(actual.device, values.device)

    def test_conversion_clips_invalid_counts_and_overflow(self):
        values = torch.tensor([-2., 0., 1., 100.], requires_grad=True)
        counts = log1p_to_counts(values)
        self.assertTrue(torch.isfinite(counts).all())
        self.assertEqual(counts[0].item(), 0.)
        self.assertEqual(counts[1].item(), 0.)
        torch.testing.assert_close(counts[2], torch.expm1(values[2]))
        torch.testing.assert_close(counts[3], torch.expm1(torch.tensor(88.)))
        counts.sum().backward()
        self.assertEqual(values.grad[0].item(), 0.)
        self.assertEqual(values.grad[3].item(), 0.)

    def test_scalar_scorer_standardize(self):
        """Standardization is affine, per output, and the identity when absent."""
        weights = torch.tensor([1.0, -1.0])
        values = torch.tensor([[4.0, 6.0], [2.0, 2.0]])
        plain = ScalarScorer(weights)
        torch.testing.assert_close(plain(values), values @ weights)

        center, scale = torch.tensor([1.0, 2.0]), torch.tensor([2.0, 4.0])
        scaled = ScalarScorer(weights, standardize=(center, scale))
        torch.testing.assert_close(scaled(values), ((values - center) / scale) @ weights)

        with self.assertRaises(ValueError):  # a zero scale would divide by zero
            ScalarScorer(weights, standardize=(center, torch.tensor([1.0, 0.0])))
        with self.assertRaises(ValueError):  # one per output, not one per anything
            ScalarScorer(weights, standardize=(torch.zeros(3), torch.ones(3)))

    def test_scorers_validate_shapes(self):
        scalar = ScalarScorer([1., -1.])
        for values in (torch.ones(3, 2, 4), torch.ones(3, 1), torch.ones(2)):
            with self.subTest(shape=values.shape), self.assertRaisesRegex(ValueError, "scalar predictions"):
                scalar(values)

    def test_stack_evaluates_the_shared_model_once(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, X):
                self.calls += 1
                return X

        model = Model()
        scorer = ScalarScorer([1., -1.])
        stack = ScoreModule(model=model, scorers=[scorer, scorer])
        self.assertIs(stack.model, model)
        torch.testing.assert_close(stack(torch.tensor([[4., 1.]])), torch.tensor([[3., 3.]]))
        self.assertEqual(model.calls, 1)


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)

    def test_objective_modes_and_targets(self):
        a = ScalarScorer([1.0, 0.0])
        scores = torch.tensor([[3.0], [1.0]])
        for mode in ("maximize", "max"):
            goal = Objective(a, mode)
            self.assertEqual(goal.mode, "maximize")
            torch.testing.assert_close(goal(scores), torch.tensor([-3.0, -1.0]))
        for mode in ("minimize", "min"):
            goal = Objective(a, mode)
            self.assertEqual(goal.mode, "minimize")
            torch.testing.assert_close(goal(scores), scores[:, 0])
        torch.testing.assert_close(
            Objective(a, "match", 2.0)(scores), torch.tensor([1.0, 1.0])
        )
        # A starting score is an ordinary explicit match target.
        torch.testing.assert_close(
            Objective(a, "match", 3.0)(scores), torch.tensor([0.0, 4.0])
        )
        with self.assertRaises(ValueError):
            Objective(a, "hold")
        with self.assertRaises(ValueError):
            Objective(a, "match")
        with self.assertRaises(ValueError):
            Objective(a, "max", 1.0)
        for target in (float("nan"), float("inf")):
            with self.subTest(target=target), self.assertRaises(ValueError):
                Objective(a, "match", target)
        for shape in ((2,), (2, 2), (2, 1, 1)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                Objective(a, "maximize")(torch.zeros(shape))

    def test_weighted_objective_sums_sub_objective_losses(self):
        a, c = ScalarScorer([1, 0]), ScalarScorer([0, 1])
        objective = WeightedObjective((Objective(a, "max"), Objective(c, "match", 2.0)), weights=(1.0, 0.5))
        scores = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
        # -a + 0.5*(c-2)^2
        torch.testing.assert_close(objective(scores), torch.tensor([-3.0 + 2.0, -1.0 + 2.0]))
        self.assertEqual(objective.scorers, (a, c))
        with self.assertRaises(ValueError):  # wrong number of columns
            objective(torch.zeros(2, 1))
        shared = WeightedObjective((Objective(a, "max"), Objective(a, "min")))
        self.assertEqual(len(shared.scorers), 1)
        torch.testing.assert_close(shared(scores[:, :1]), torch.zeros(2))
        with self.assertRaises(ValueError):
            WeightedObjective(())

    def test_objective_weights_default_validate_and_preserve_gradients(self):
        first, second = ScalarScorer([1., 0.]), ScalarScorer([0., 1.])
        objectives = (Objective(first, "max"), Objective(second, "match", target=2.))
        equal = WeightedObjective(objectives)
        self.assertEqual(equal.weights, (1., 1.))
        values = torch.tensor([[3., 4.]], requires_grad=True)
        torch.testing.assert_close(equal(values), torch.tensor([1.]))
        weighted = WeightedObjective(objectives, weights=[2., 0.5])
        torch.testing.assert_close(weighted(values), torch.tensor([-4.]))
        weighted(values).sum().backward()
        torch.testing.assert_close(values.grad, torch.tensor([[-2., 2.]]))
        self.assertEqual(equal.weights, (1., 1.))
        for weights in ([], [1.], [1., 1., 1.], [1., float("nan")],
                        [float("inf"), 1.], [-1., 1.], [0., 0.]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                WeightedObjective(objectives, weights=weights)

    def test_weighted_objective_shares_model_and_scorer_evaluation(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, X):
                self.calls += 1
                return X

        class CountingScorer:
            def __init__(self):
                self.calls = 0

            def __call__(self, outputs):
                self.calls += 1
                return outputs[:, 0]

        model, scorer = Model(), CountingScorer()
        objective = WeightedObjective(
            objectives=(Objective(scorer, "maximize"), Objective(scorer, "match", 2.)),
            weights=(1., 0.5),
        )
        module = ScoreModule(model, objective.scorers)
        X = torch.tensor([[3.], [1.]], requires_grad=True)
        losses = objective(module(X))
        torch.testing.assert_close(losses, torch.tensor([-2.5, -0.5]))
        losses.sum().backward()
        torch.testing.assert_close(X.grad, torch.tensor([[0.], [-2.]]))
        self.assertEqual(model.calls, 1)
        self.assertEqual(scorer.calls, 1)

    def test_nested_and_custom_objectives_route_scorer_indices_and_gradients(self):
        first, second = ScalarScorer([1., 0.]), ScalarScorer([0., 1.])

        class CustomObjective:
            # Deliberately reverse the root's score order.
            scorers = (second, first)

            def __call__(self, scores):
                return (scores[:, 0] - 2 * scores[:, 1]) ** 2

        nested = WeightedObjective((Objective(second, "match", 2.), CustomObjective()),
                                   weights=(0.5, 0.25))
        objective = WeightedObjective((Objective(first, "maximize"), nested), weights=(1., 2.))
        self.assertIs(objective.scorers[0], first)
        self.assertIs(objective.scorers[1], second)
        values = torch.tensor([[3., 4.]], requires_grad=True)
        torch.testing.assert_close(objective(values), torch.tensor([3.]))
        objective(values).sum().backward()
        torch.testing.assert_close(values.grad, torch.tensor([[3., 2.]]))


if __name__ == "__main__":
    unittest.main()
