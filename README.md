# regseqkit

Tools for regulatory sequence analysis: sequence and signal extraction, model scoring and evaluation, calibration, mutagenesis, motif export, plotting, and sequence optimization.

## Quick Start

```text
DNA sequence → model outputs → scores → objective → losses → optimizer → designed DNA sequence
```

Start with a PyTorch `model` and a DNA string `sequence` of the model's required input length. The model must accept one-hot DNA shaped `(N, 4, length)` in ACGT order and return scalar predictions shaped `(N, outputs)`.

For example, suppose its two outputs are a reference condition and a target condition, in that order, and you want to increase the target relative to the reference:

```python
import torch

from regseqkit.design import Objective, WeightedObjective, greedy_substitution
from regseqkit.scoring import ScalarScorer, ScoreModule
from regseqkit.sequences import OneHotEncoder

# 1. Prepare the model and starting sequence.
device = "cpu"  # Use "cuda:0" for a GPU.
model = model.to(device).float().eval()
encoder = OneHotEncoder()
template = torch.from_numpy(encoder.to_onehot([sequence])).float()

# 2. Score target minus reference, using weights in model output order.
scorer = ScalarScorer(weights=[-1.0, 1.0])
scores = ScoreModule(model, [scorer])

# 3. Maximize that score. Objectives return a loss to minimize.
objective = Objective(scorer, mode="maximize")

# 4. Allow edits and run the search.
positions = range(template.shape[-1])
designed = greedy_substitution(
    scores, template, objective, positions=positions,
    max_iter=20, batch_size=128, device=device,
)
designed_sequence = encoder.from_onehot(designed[0].numpy())

with torch.no_grad():
    before = scores(template.to(device)).item()
    after = scores(designed.to(device)).item()
print(f"contrast: {before:.3f} → {after:.3f}")
print(designed_sequence)
```

The scorer combines predictions as returned by `model`. If those are log1p counts, the contrast is a difference of logged values. To compare counts, wrap the model before constructing `ScoreModule`:

```python
from regseqkit.wrappers import Log1pToCounts

model = Log1pToCounts(model)
```

This wrapper expects log1p predictions and returns nonnegative counts. If the model returns a tuple or profiles, select the scalar tensor first. For Cherimoya, `cherimoya.wrappers.LogCountWrapper` selects its scalar output.

An `Objective` references one scorer and converts its scores into losses to minimize. Choose its optimization mode:

| Goal | Objective |
| --- | --- |
| Increase a score | `Objective(scorer, mode="maximize")` |
| Decrease a score | `Objective(scorer, mode="minimize")` |
| Reach a target | `Objective(scorer, mode="match", target=2.0)` |
| Stay near the starting value | `Objective(scorer, mode="match", target=reference_value)` |

To increase the contrast while keeping the reference condition near its starting value, replace steps 2–3 with:

```python
scorer = ScalarScorer(weights=[-1.0, 1.0])
reference_scorer = ScalarScorer(weights=[1.0, 0.0])
scores = ScoreModule(model, [scorer, reference_scorer])
increase = Objective(scorer, mode="maximize")
with torch.no_grad():
    reference_value = scores(template.to(device))[0, 1].item()
preserve = Objective(reference_scorer, mode="match", target=reference_value)
objective = WeightedObjective(objectives=(increase, preserve), weights=(1.0, 0.5))
scores = ScoreModule(model, objective.scorers)
```

Pass score columns in `objective.scorers` order. Weighted composition collects the scorers its sub-objectives require, deduplicating shared scorer objects in first-use order. Omitted weights default to 1 for every sub-objective. `match` adds a squared penalty for deviation from its target; its objective weight controls the trade-off and depends on the score scales. Compute a new reference value for each starting sequence. With multiple scores, print the before/after tensors directly instead of calling `.item()`.

Mutagenesis, greedy substitution, and Ledidi accept `positions` as sorted, unique, zero-based indices. Use `positions=range(start, end)` for an interval or `positions=[2, 5, 8]` for selected sites. Omitting `positions` selects all sites; `positions=[]` allows no edits. Positions outside the selection stay fixed during design. For gradient-based design, use `ledidi_design(scores, template, objective, positions=positions, device=device)` from `regseqkit.design` with a differentiable model.

The Python workflow needs no JSON configuration when `model` already returns the required predictions. For checkpoint loading, recipes, and saved artifacts, see the [design stage workflow](../scripts/4_design/README.md).

<details>
<summary>API and configuration reference</summary>

Install this package in editable mode while developing it alongside the parent project. The parent `pyproject.toml` declares the local editable dependency, so run `uv sync` from the parent project. Library changes then take effect without reinstalling. Project scripts and configuration stay in the parent project; standalone library tests live in `regseqkit/tests`. The portability test copies the library into a bare project and runs inference, ISM, and greedy design with a plain toy PyTorch model and no Cherimoya.

## The chain

The model, scoring, and design layers provide predictions, scores, and losses respectively.

```
io  calibrate  metrics  sequences     numpy only, no torch
PyTorch model  ->  scoring  ->  design   torch
mutagenesis   motifs   figures
config                                reads project.json into arguments for the rest
cherimoya/                            one implementation of the model contract
```

`cherimoya/contrast_loss.py` is the one place that reaches into the library rather than
wrapping it: it rebinds the loss name cherimoya's fit loop calls so a within-experiment
contrast term rides inside the count loss, then runs the stock fit command. It is written
against the pinned cherimoya version and nothing else imports it.

The division that matters most:

- **A model** is a PyTorch module returning the tensor to score.
- **A scorer** computes scores from those predictions. `scoring.py`.
- **An objective** converts scores into losses for an optimization goal. `design/objectives.py`.
- **A calibration** supplies reference scales and targets. `calibrate.py`.

Scorers combine model outputs into scores. Objectives apply optimization goals to scores and can combine the resulting losses.

`io`, `metrics`, `calibrate` and `sequences` import neither torch nor tangermeme, so a stage
that only reads coordinates or summarizes saved arrays stays cheap.
`test_light_modules_import_without_torch` pins that.

## project.json

One file. Blocks are independent and a stage reads only what it needs. Paths resolve
relative to the file.

| Block | Holds |
| --- | --- |
| `data` | `fasta`, `loci`, `negatives`, `exclusion_lists`; or `examples`, a prepared NPZ, for a model with no genome |
| `splits` | split name to its chromosomes |
| `tracks` | `name` and `path` per observed track; `path` is a list when a track has several channels |
| `models` | one entry per checkpoint with the `outputs` it emits in order; several entries concatenate |
| `model_output` | `head`: `scalar` (default) or `profile`; `space`: `native` (default) or `counts` |
| `scorers` | `name`, `weights` mapping output name to weight; names belong to configuration, not numerical scorer objects |
| `calibration` | `path` to a calibration JSON and the `reference` set in it; both required |
| `design` | `start`, `end`, `templates`, `objectives`, and `greedy` / `ledidi` settings |

The optimizer settings are `greedy`: `max_iter`, `batch_size`, and `tol`; `ledidi`: `max_iter`, `batch_size`, `n_samples`, `l`, and `random_state`, which is offset by the template index so each template's run is reproducible on its own. Greedy search supports single-site ACGT substitutions and rejects `motifs` and `motif_file`.

Ledidi's `batch_size` controls samples per optimization step. `n_samples` is an
optional positive integer controlling fresh draws from the learned distribution
after optimization; duplicates are possible. Omit it or use `null` to retain
the default best recorded optimization batch. Changing `n_samples` does not
change the optimization batch size or run extra optimization trajectories.

**The `design` block can live in its own file, called a recipe.** Everything above it
describes the model and what can be measured from it, and changes rarely; a recipe describes
what you are trying to make, and changes every run. Split, a sweep is several small files
rather than several copies of everything:

```bash
for r in config/recipes/*.json; do
  3_optimize.py --config config/experiment_second.json --recipe "$r" \
    --method greedy --output-dir "output/4_design/$(basename "$r" .json)"
done
```

A recipe holds those keys at the top level with no enclosing `design`, resolves its own
paths beside itself, and its stem names the run. That name lands in a `run` column of every
scores table, so concatenated runs group by `run`, `objective` and `template`. Omit
`--recipe` and the project's own block is used, with the run named `inline`. The word
*design* is left to mean a designed sequence.

A track is a name and a path. Any further vocabulary, such as which condition or batch a
track belongs to, is the project's and lives in the project's package.

**Nothing in the file names a Python callable.** A custom objective declares its scorers and implements the score-to-loss callable interface. Project entry points choose their model-loading functions; a configuration file cannot cause an import.

## Sequences

`extract_loci_with_coords` centers a window on each locus of a BED and returns a `Loci`
tuple: `onehot` `(N, 4, in_window)`, `signals` `(N, tracks, out_window)`, and `coords`
carrying `chrom`, `start`, `end` and `source_row`.

Returning the surviving coordinates is the point. Tangermeme's extractor returns a boolean
mask over the rows it *read*, so recovering which locus produced example 7 meant rebuilding
its input frame through a private function.

The drop rules reproduce Tangermeme's exactly, because the numbers a project has already
reported were computed on the loci its filters kept: a window off either chromosome end, an
exclusion-list overlap tested at 100 bp resolution, and, with `drop_ambiguous=True`, any
window containing a base outside ACGT. `test_extraction_matches_tangermeme` checks
sequences, signals and coordinates against it on a real peak set.

Encoding is batched through a 128-entry lookup in `OneHotEncoder`, and everything returns
numpy; the conversion to tensors happens at the model boundary in `config`.

Two other readers: `read_fasta` for sequences already cut to a fixed width, which is how
design templates arrive, and `read_npz` for prepared examples.

`config` wraps the extractor twice. `load_inputs` reads one configured split with the
exclusion lists applied, which is what a scored number is computed on. `load_inputs_at`
reads explicit coordinates with no split and no exclusion list, for a figure that names
what it draws: a locus chosen for what the experiment measured there is drawn even when
it overlaps an exclusion list, and the only rows dropped are those whose windows run off
their chromosome. Both return the same arrays.

## Models

The scoring and design APIs accept ordinary PyTorch modules. Wrap a model to select its output tensor or convert units before passing it to `ScoreModule`. A model with multiple outputs can be used directly; multiple models can be combined with a project-defined module, as in [example_two_models.py](../examples/example_two_models.py).

`config.ModelMetadata` is a validated workflow data record containing output names, input width, optional profile width, channel-group sizes and native scalar encoding. Configuration helpers use it for sequence extraction, observed-signal alignment, scorer construction and artifact metadata. Scorers and optimizers do not depend on it. Saved artifacts contain dictionaries so later scoring stages can read predictions without loading Torch or model libraries.

This project's `us_responsive.inference.load_models(config, device)` returns a list of loaded PyTorch models, combined `ModelMetadata`, and resolved checkpoint entries. `load_output_module(config, models, metadata)` selects the configured output tensor and space, combining outputs privately when multiple checkpoints are used. `predict_models(models, X, device=...)` supplies scalar and profile predictions for evaluation.

Cherimoya's `load_checkpoint`, `output_module` and `predict_outputs` functions retain checkpoint validation, batch-size limits, expected-profile conversion and reverse-complement handling. They accept explicit arguments and do not import the configuration module. `cherimoya.jobs.project_blocks` produces track and checkpoint definitions from a training job for the project workflow.

Pass the desired device directly to loading, inference and design. The scripts default to `"cpu"`; use an explicit `--device cuda:N` for GPU execution.

## Scorers

One class per head, each owning the vocabulary it validates.

| Class | `score(outputs)` | Fields |
| --- | --- | --- |
| `ScalarScorer` | weighted sum of scalar outputs | `standardize` |
| `ProfileScorer` | weighted channels per position, then `\|.\|` and a reduction | `absolute`, `reduction` in `mean`/`max`/`sum` |

Call a scorer directly with `scorer(predictions)` to get one score per sequence. `ScoreModule(model, scorers)` calls the model once and stacks the scorers' results into columns.

Output conversion belongs to the model wrapper. `Log1pToCounts(model)` converts log1p predictions before any scorer reads them; without it, scorers combine the model's values directly. A contrast of log1p predictions is a log ratio of counts plus one. A contrast of converted counts is a difference of counts.

Configured workflows choose this once with `"model_output": {"head": "scalar", "space": "native"}` or `"model_output": {"head": "scalar", "space": "counts"}`. The project’s `load_output_module(config, models, metadata)` prepares the module passed to `ScoreModule`. The model-independent `config.prepare_output_module(config, module, metadata)` applies the configured conversion to an already selected output tensor. Profile heads already return expected signal. Scorers validate tensor shapes and contain no output-selection metadata. All scalar scorers in one stack read the same space; use separate output views when both spaces are needed. Existing artifacts with uniform legacy scorer units and per-scorer head settings remain readable.

`config.build_scorers(entries, outputs)` returns a dictionary mapping configured names to numerical scorers. When the scorer block is absent, `config.identity_scorers(outputs)` supplies one scorer per output, selecting each output unchanged. For outputs `["a", "b"]`, the mapping's keys are `"a"` and `"b"`, and its scorers have weights `[1, 0]` and `[0, 1]`. Pass `list(scorers.values())` to `ScoreModule`; retain the keys for report columns and artifact names.

Supply numeric weights in model output order. For `a*y1 - b*y2`, use
`ScalarScorer(weights=[a, -b])`. For the mean of two treatment replicates minus
the mean of two control replicates, ordered as treatment_1, treatment_2,
control_1, control_2, use `weights=[0.5, 0.5, -0.5, -0.5]`.

Reduction belongs only to `ProfileScorer`, because only the profile head has a position axis
to collapse. A scalar scorer never touches the profile.

### ScoreModule

`ScoreModule(model, scorers)` evaluates several scorers over one model forward per input batch. Two consumers:

- **Attribution.** ISM evaluates roughly 3000 alternative sequences for a 1000 bp window, batched for inference. Every scorer uses the same model predictions; additional scorers add aggregation work without repeating model forwards.
- **A weighted objective**, which reads several scorers per forward.

Optimizers receive a stack containing the scorers required by their objective, in `objective.scorers` order. The design workflow also builds a stack of every configured scorer to report the finished sequence's scores.

Construct `ScoreModule(model, objective.scorers)` directly. Objectives do not assemble model modules. Use `str(objective)` for a readable description; configured identifiers remain separate. Both optimizers compute `loss = objective(scores)` and then validate one finite loss per candidate with the internal `_validate_loss` helper. Custom objectives only need to implement their calculation and declare their scorers; there is no public `checked()` method.

## Objectives

`Objective(scorer, mode, target=None)` applies one goal to one scorer. It exposes that scorer through `objective.scorers`, consumes scores shaped `(N, 1)`, and returns losses shaped `(N,)`. Objectives compute losses per sequence; optimizer adapters perform batch reductions when needed.

| `mode` | Loss | `target` |
| --- | --- | --- |
| `maximize` | `-s` | rejected |
| `minimize` | `s` | rejected |
| `match` | `(s - target)**2` | required and finite |

`max` and `min` are accepted aliases and are stored and described as `maximize` and `minimize`. These goals are unbounded, so greedy search stops on its improvement tolerance or iteration budget.

`WeightedObjective(objectives, weights=None)` combines sub-objectives' losses. Weights default to 1, must be finite and nonnegative, and must include at least one positive value. Weights are preserved without normalization, including when there is only one sub-objective. Their absolute scale affects greedy's loss improvement tolerance and Ledidi's balance against its edit penalty.

Weighted composition collects required scorers in first-use order, deduplicating shared scorer objects by identity. Each sub-objective receives the columns its own scorers require, in its declared order. Sub-objectives can be standalone, weighted, or custom objectives. All objectives consume `(N, len(objective.scorers))` scores and return `(N,)` losses.

To keep a score near its starting value, compute that score first and pass it as the match target. Recipes request this with `"target": "template"`; the workflow resolves a numeric target separately for each template before optimization:

```json
{"name": "open_more_hold_control",
 "objectives": [
   {"scorer": "delta_high", "mode": "maximize"},
   {"scorer": "CONTROL", "mode": "match", "target": "template", "weight": 0.5}
 ]}
```

Matching adds a soft squared penalty. Increasing another score can compensate for a target deviation; this does not enforce a fixed tolerance. Biological contrasts belong in scorers, while independently targeted goals belong in objective composition.

A custom objective needs an ordered `scorers` tuple and a `__call__(scores)` implementation returning one finite loss per sequence. Inheritance from a built-in class is optional. Optimizers validate loss shape and finiteness; weighted composition calls its sub-objectives directly.

`config.build_objectives(entries, scorers)` builds `Objective` for standalone definitions and `WeightedObjective` for definitions with sub-objectives. Recipe weights appear beside their sub-objective definitions and are collected into `WeightedObjective.weights`. A top-level weight explicitly scales an objective through a weighted wrapper.

## Calibration

Independently trained models or tracks from different batches can predict on different scales. Weights between sub-objective losses depend on those scales as well as the intended trade-off.

`2_calibrate.py` summarizes predictions over **any named reference set**. Which set you point
it at *is* the method, and nothing in the module prefers one:

- **GC-matched negatives** give a null where zero means background. This is usually the
  right choice for a model trained on peaks plus negatives, because a null built from
  out-of-distribution input such as shuffled sequence puts every design in an unreachable
  part of the scale.
- **The peak set** gives a reporting grid, so a score reads as "the 92nd percentile of real
  peaks".
- **A held-out split** gives the model's working range.

The calibration's center and scale standardize model outputs before scoring. This changes the relative contributions of outputs to a contrast and the scales of objective losses. The quantile grid supports reporting and resolves targets such as `p90` into numbers before optimization. Prediction artifacts record raw model units.

Building one is inference over a locus set plus a summary:

```bash
infer.py  --config project.json --loci .../negatives.bed --output predictions/negatives
2_calibrate.py --reference negatives:predictions/negatives --output calibration.json
```

Then point the config at it, and every scalar scorer is standardized. A scorer opts out with `"standardize": false`. Calibration must describe the prepared output space. For count-space scoring, pass `--space counts` to `2_calibrate.py`; inference artifacts remain in native model space, and calibration converts their values before computing summaries.

```json
"calibration": {"path": "calibration.json", "reference": "negatives"}
```

Match targets must use the same scale as their scorer's scores. Weights between sub-objective losses depend on the score scales, with matching producing squared deviations. A positive rescaling of one score preserves its ranking under maximization, but can change greedy stopping and Ledidi's balance against its edit penalty.

## Artifacts

`io.save_arrays(stem, arrays, metadata)` writes `<stem>.npz` beside `<stem>.json`;
`load_arrays` accepts the stem or either member. Extensions are **appended, not
substituted**, because a stem like `multitask.seed0.test` already contains dots and
`Path.with_suffix` would rewrite it.

The sidecar carries the model metadata, the resolved model entries, the resolved
configuration, and the scorer definitions under `definitions`, so a later stage can tell
whether a scorer changed between two artifacts. `validate_examples` enforces the one invariant everything
assumes: `ids` is unique and every array shares its row count, so anything not
example-indexed belongs outside that dict.

## Attribution, motifs, design

`mutagenesis.single_site_saturation_mutagenesis(model, X, positions=...)` returns `reference_predictions` `(N, scorers)` and absolute `mutant_predictions` `(N, positions, 4, scorers)` in ACGT order. Positions are sorted, unique full-input coordinates; omitting them selects all positions. The core computes reference predictions on each call, evaluates the three alternative bases per position, and fills the unchanged base's entry from the reference predictions. All mutants for one input sequence are allocated on CPU and passed to `tangermeme.predict` in one call; `batch_size` controls its internal inference batching. Set `verbose=True` to show progress over input sequences. Returned tensors are detached on CPU.

`mutagenesis.ssm_attribution(reference_predictions, mutant_predictions)` converts the core's predictions into effects shaped `(N, scorers, 4, positions)`. By default, effects are mean-centered over ACGT and hypothetical: every base at every position is retained. Use `center=False` for uncentered mutant-minus-reference effects. Use `hypothetical=False, X=X[:, :, positions]` to retain only observed-base effects. Precision settings are controlled by the caller; neither function changes them.

TF-MoDISco receives arrays transposed to `(N, length, 4)` and forms the projection itself;
TF-MInDi wants projected contributions, which are `hypothetical * one_hot`. Motif exports
carry both pattern signs, and negative CWMs are never written as probability matrices.

`design.greedy_substitution` applies an objective with numeric targets to absolute predictions from the shared core. It accepts an edit only when loss improvement exceeds `tol`, resolving ties by ascending position then ACGT order. Only objective scorers are evaluated, and previously edited positions remain eligible. `max_iter` limits accepted substitutions rather than final Hamming distance. The optional `on_iteration` callback receives a `GreedyIteration` containing `current_sequence`, `current_scores`, `current_loss`, all candidate scores/losses, `mutation`, `accepted`, `selected_sequence`, `selected_scores`, `selected_loss`, and `stop_reason`. Candidate rows follow ascending editable-position order. Accepted mutations use strings such as `"4A>C"` with zero-based full-input coordinates; rejected rounds have `mutation=None` and retain the current state as the selected state. `greedy_design` remains an alias. Both names return the final sequence tensor.

Both optimizers accept the same `positions` selection as mutagenesis and construct edit masks internally. Ledidi converts the selection to its complementary input mask, where true means locked, and reduces the objective's per-example loss to a scalar.

`design.aggregate_greedy_history(history_dir, destination)` collects one objective/template search's completed per-round artifacts into an NPZ/JSON pair. Candidate arrays gain a round axis; positions and provenance are stored once. Baseline/result scores, sequences, losses, proposed edits, acceptance, and stop reasons remain aligned by round. The aggregate's JSON records completion and the final stop reason. This function can collect interrupted histories independently of optimization; the stage script calls it automatically after successful greedy runs and writes an index under `trajectories.greedy/`. See the [stage README](../scripts/4_design/README.md) for loading and aggregation examples.

</details>
