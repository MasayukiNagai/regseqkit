"""Core features work when optional libraries cannot be imported."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


class OptionalDependencyTests(unittest.TestCase):
    def test_core_without_ledidi_cherimoya_or_modisco(self):
        code = '''
import importlib.abc
import sys

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"ledidi", "cherimoya", "modiscolite"}:
            raise ModuleNotFoundError(f"blocked optional dependency: {fullname}", name=fullname)

sys.meta_path.insert(0, BlockOptional())
import torch
import regseqkit.figures
import regseqkit.greedy_analysis
import regseqkit.motifs
from regseqkit.design import Objective, greedy_substitution, ledidi_design
from regseqkit.config import build_objectives, build_scorers
from tangermeme.predict import predict
from regseqkit.mutagenesis import single_site_saturation_mutagenesis
from regseqkit.scoring import ScalarScorer, ScoreModule, identity_scorers

class Model(torch.nn.Module):
    def forward(self, X):
        return X[:, 0].sum(-1, keepdim=True)

X = torch.zeros(1, 4, 3)
X[:, 3] = 1
scorer = ScalarScorer([1.0])
module = ScoreModule(Model(), [scorer])
objective = Objective(scorer, "maximize")
reference, mutants = single_site_saturation_mutagenesis(module, X, device="cpu")
assert mutants.shape == (1, 3, 4, 1)
designed = greedy_substitution(module, X, objective, max_iter=2, device="cpu")
assert predict(module, designed, device="cpu").item() == 2
try:
    ledidi_design(module, X, objective)
except RuntimeError as exc:
    assert "regseqkit[ledidi]" in str(exc)
else:
    raise AssertionError("missing Ledidi was not reported")
try:
    import regseqkit.cherimoya
except ImportError as exc:
    assert "regseqkit[cherimoya]" in str(exc)
else:
    raise AssertionError("missing Cherimoya was not reported")
try:
    regseqkit.motifs.export_matrices("unused.h5", "unused")
except ImportError as exc:
    assert "regseqkit[motifs]" in str(exc)
else:
    raise AssertionError("missing motif exporter was not reported")
'''
        library = next(
            parent for parent in Path(__file__).resolve().parents
            if (parent / "pyproject.toml").is_file()
        )
        env = dict(os.environ, PYTHONPATH=str(library / "src"))
        result = subprocess.run([sys.executable, "-c", code], env=env,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
