import torch

from cherimoya import Cherimoya
from cherimoya.wrappers import ControlWrapper, LogCountWrapper

from regseqkit.design import Objective, greedy_substitution
from regseqkit.scoring import ScalarScorer, ScoreModule
from regseqkit.sequences import OneHotEncoder


# 1. Load a model and select its count output.
device = "cpu"  # Or "cuda:0".
checkpoint = "path/to/model.pt"  # Replace with your Cherimoya checkpoint.
model = Cherimoya.load(checkpoint, device=device, compile=False).float().eval()

# Cherimoya returns (profile_logits, log1p_counts).
# ControlWrapper supplies zero control tracks when the model requires them.
wrapped_model = LogCountWrapper(ControlWrapper(model))

# 2. Define a scorer for the count output.
# For a single-output model, use [1.0]. For multiple outputs, supply one
# weight per output, e.g. [1.0, 0.0] to maximize only the first output.
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

# 6. Evaluate the design in count units.
with torch.no_grad():
    print("Before counts:", score_module(template_onehot.to(device)).cpu())
    print("After counts:", score_module(designed.to(device)).cpu())

designed_sequence = onehot_encoder.from_onehot(designed[0].cpu().numpy())
print("Designed sequence:", designed_sequence)
