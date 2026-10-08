import torch

from regseqkit.design import Objective, greedy_substitution
from regseqkit.scoring import ScalarScorer, ScoreModule
from regseqkit.sequences import OneHotEncoder


# 1. Define a model wrapper.
class CountOutput(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X):
        # X: (N, 4, length)
        # profiles: (N, channels, positions); counts: (N, outputs)
        profiles, counts = self.model(X)
        return counts


device = "cpu"  # Or "cuda:0".
model = None  # Replace with your model returning (profiles, counts).
wrapped_model = CountOutput(model).to(device).float().eval()

# If the count output is log1p-transformed, convert it before scoring:
# from regseqkit.wrappers import Log1pToCounts
# wrapped_model = Log1pToCounts(wrapped_model)

# 2. Define a scorer for the count output.
# Use one weight per output: [1.0] for one output, or [1.0, 0.0] to
# select the first of two outputs.
count_scorer = ScalarScorer(weights=[1.0])

# 3. Define an objective.
objective = Objective(count_scorer, mode="maximize")
score_module = ScoreModule(wrapped_model, objective.scorers)

# 4. Prepare a template sequence to edit.
onehot_encoder = OneHotEncoder()
template_sequence = "ACGT" * 50  # Replace with a sequence of the model's input length.
template_onehot = torch.from_numpy(onehot_encoder.to_onehot([template_sequence])).float()

# 5. Run the design algorithm.
designed = greedy_substitution(
    score_module,
    template_onehot,
    objective,
    positions=range(template_onehot.shape[-1]),
    max_iter=20,
    batch_size=128,
    device=device,
)

# 6. Evaluate the design.
with torch.no_grad():
    print("Before counts:", score_module(template_onehot.to(device)).cpu())
    print("After counts:", score_module(designed.to(device)).cpu())

designed_sequence = onehot_encoder.from_onehot(designed[0].cpu().numpy())
print("Designed sequence:", designed_sequence)
