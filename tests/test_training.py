"""Training job construction from resolved paths and disjoint splits."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("cherimoya"), "requires regseqkit[cherimoya]")
class TrainingJobTests(unittest.TestCase):
    def test_grouped_jobs_roundtrip_and_preserve_output_metadata(self):
        from regseqkit.cherimoya.training import build_jobs
        from regseqkit.cherimoya.jobs import load_job, project_blocks

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = build_jobs(
                root, fasta="/g.fa", loci="/p.bed",
                tracks=[{"track": "pair", "signal": ["/plus.bw", "/minus.bw"], "tag": "x"},
                        {"track": "atac", "signal": "/atac.bw"}],
                splits={"train": ["chr1"], "valid": ["chr2"], "test": ["chr3"]},
                parameters={"in_window": 32, "out_window": 26}, mode="both",
            )
            for path in paths:
                spec = json.loads(path.read_text())
                run = root / "runs" / spec["name"]
                run.mkdir(parents=True)
                (run / f"{spec['name']}.torch").touch()
                signals = [r["signal"] for r in spec["tracks"]]
                (run / f"{spec['name']}.fit.json").write_text(json.dumps({
                    "signals": signals, "in_window": 32, "out_window": 26,
                }))
                job = load_job(path, None)
                self.assertEqual(job.signal_groups, spec["signal_groups"])
                self.assertEqual([r["path"] for r in project_blocks(job)["tracks"]], signals)
                if job.tracks[0] == "pair":
                    self.assertEqual(job.track_records[0]["tag"], "x")

    def test_build_jobs_takes_resolved_arguments(self):
        from regseqkit.cherimoya.training import build_jobs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = build_jobs(
                root / "collection", fasta="/g.fa", loci="/p.bed",
                tracks=[{"track": "a", "signal": "/a.bw"}],
                splits={"train": ["chr1"], "valid": ["chr2"], "test": ["chr3"]},
                parameters={"in_window": 20, "out_window": 10}, mode="multitask",
            )
            self.assertEqual(json.loads(paths[0].read_text())["sequences"], "/g.fa")
            both = build_jobs(
                root / "both", fasta="/g.fa", loci="/p.bed",
                tracks=[{"track": "a", "signal": "/a.bw"}, {"track": "b", "signal": "/b.bw"}],
                splits={"train": ["chr1"], "valid": ["chr2"], "test": ["chr3"]},
                parameters={"in_window": 20, "out_window": 10}, mode="both",
            )
            self.assertEqual(len(both), 3)
            self.assertEqual(json.loads(both[0].read_text())["fit_parameters"]["negative_ratio"], 0)
            self.assertTrue((root / "both/jobs.txt").read_text().startswith("configs/"))
            with self.assertRaises(ValueError):
                build_jobs(
                    root / "bad", fasta="/g.fa", loci="/p.bed",
                    tracks=[{"track": "a", "signal": "/a.bw"}],
                    splits={"train": ["chr1"], "valid": ["chr1"], "test": ["chr3"]},
                    parameters={"in_window": 20, "out_window": 10},
                )


if __name__ == "__main__":
    unittest.main()
