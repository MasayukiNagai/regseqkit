"""The library runs independently of the parent project's package."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from regseqkit.io import load_arrays

LIBRARY_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)


class PortabilityTests(unittest.TestCase):
    def test_copied_core_runs_without_a_project_layer(self):
        """`regseqkit` alone, in a fresh project, with a plain PyTorch model."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "nested" / "project"
            (root / "src").mkdir(parents=True)
            (root / "pyproject.toml").write_text('[project]\nname="portable-test"\nversion="0.1.0"\n')
            shutil.copytree(LIBRARY_ROOT / "src/regseqkit", root / "src/regseqkit",
                            ignore=shutil.ignore_patterns("__pycache__"))
            (root / "run.py").write_text("""
import numpy as np, torch
from tangermeme.predict import predict
from regseqkit.mutagenesis import single_site_saturation_mutagenesis, ssm_attribution
from regseqkit.design import Objective, WeightedObjective, greedy_design
from regseqkit.io import save_arrays, validate_examples
from regseqkit.scoring import ScalarScorer, ScoreModule
from regseqkit.scoring import identity_scorers
from regseqkit.config import build_objectives, build_scorers

class Toy(torch.nn.Module):
    def forward(self, X):
        return torch.stack((X[:, 0].sum(-1), X[:, 1].sum(-1)), dim=1)

model = Toy().eval()
outputs = ("A", "C")
X = torch.zeros(2, 4, 8)
X[:, 3] = 1
ids = np.array(["a", "b"])

scalars = predict(model, X, device="cpu")
arrays = dict(ids=ids, observed=np.zeros((2, 2)), predicted=scalars.numpy())
validate_examples(arrays, outputs)
save_arrays("out/predictions.test", arrays, dict(kind="predictions"))

stack = ScoreModule(model, list(identity_scorers(outputs).values()))
reference, mutants = single_site_saturation_mutagenesis(
    stack, X, positions=range(2, 6),
    batch_size=8,
)
attr = ssm_attribution(reference, mutants)
np.savez_compressed("out/scorers.test.attr.npz", attr[:, 0].numpy())

scorer = build_scorers([dict(name="A", weights={"A": 1.0})], outputs)["A"]
single = ScoreModule(model, [scorer])
objective = build_objectives(
    [dict(name="increase_A", scorer="A", mode="maximize")], {"A": scorer}
)["increase_A"]
designed = greedy_design(single, X[:1], objective, positions=range(2, 6),
                         max_iter=3)
np.savez_compressed("out/designs.npz", onehot=designed.numpy())
print("ok")
""")
            env = dict(os.environ, PYTHONPATH=str(root / "src"),
                       MPLCONFIGDIR=str(root / "mpl"))
            result = subprocess.run([sys.executable, "run.py"], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            arrays, metadata = load_arrays(root / "out/predictions.test")
            self.assertEqual(metadata["kind"], "predictions")
            with np.load(root / "out/scorers.test.attr.npz") as handle:
                self.assertEqual(handle["arr_0"].shape, (2, 4, 4))
            with np.load(root / "out/designs.npz") as handle:
                self.assertEqual(int(handle["onehot"][:, 0].sum()), 3)
            # The core must not import the project package, even transitively.
            self.assertNotIn("helpers", result.stdout + result.stderr)

