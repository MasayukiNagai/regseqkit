import torch

from regseqkit.design import Objective, WeightedObjective, greedy_substitution
from regseqkit.scoring import ScalarScorer, ScoreModule
from regseqkit.sequences import OneHotEncoder


# 1. Define a model wrapper
class TwoModels(torch.nn.Module):
    def __init__(self, modelA, modelB):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB

    def forward(self, X):
        activity_a = self.modelA(X)  # (N, 1)
        activity_b = self.modelB(X)  # (N, 1)
        return torch.cat([activity_a, activity_b], dim=1)  # (N, 2)


device = "cpu"  # Or "cuda:0".
modelA = None  # Replace with your modelA instance
modelB = None  # Replace with your modelB instance
wrapped_model = TwoModels(modelA, modelB).to(device).float().eval()

# 2. Define scorers for each output from the model
scorer_a = ScalarScorer(weights=[1.0, 0.0])  # modelA's output
scorer_b = ScalarScorer(weights=[0.0, 1.0])  # modelB's output

# 3. Define an objective
objective = WeightedObjective(
    objectives=[
        Objective(scorer_a, mode="maximize"),
        Objective(scorer_b, mode="minimize"),
    ],
    weights=[1.0, 1.0],
)
scorers = ScoreModule(wrapped_model, objective.scorers)

# 4. Prepare template sequence(s) to apply edit
onehot_encoder = OneHotEncoder()
template_sequence = "ACGT" * 50  # Replace with your actual template sequence
template_onehot = onehot_encoder.to_onehot([template_sequence]).float()

# 5. Run design algorithm
designed = greedy_substitution(
    scorers,
    template_onehot,
    objective,
    positions=range(template_onehot.shape[-1]),
    max_iter=20,
    batch_size=128,
    device=device,
)

# 6. Evaluate the design
with torch.no_grad():
    print("Before [A, B]:", scorers(template_onehot.to(device)).cpu())
    print("After  [A, B]:", scorers(designed.to(device)).cpu())

designed_sequence = onehot_encoder.from_onehot(designed[0])
print("Designed sequence:", designed_sequence)
