"""Seqlet export preserves example order and centered genomic coordinates."""

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from regseqkit.motifs import export_seqlets


class MotifCoordinateTests(unittest.TestCase):
    def test_odd_window_multiple_loci_signs_and_strands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bed = root / "loci.bed"
            bed.write_text("chr1\t10\t15\nchr2\t101\t108\n")
            patterns = root / "patterns.h5"
            with h5py.File(patterns, "w") as handle:
                for category, indices, starts, ends, reverse in (
                    ("pos_patterns", [1, 0], [0, 1], [2, 4], [False, True]),
                    ("neg_patterns", [1], [3], [7], [True]),
                ):
                    seqlets = handle.create_group(f"{category}/pattern_0/seqlets")
                    for name, values in (("example_idx", indices), ("start", starts),
                                         ("end", ends), ("is_revcomp", reverse)):
                        seqlets[name] = np.asarray(values)
            output = root / "seqlets.bed"
            export_seqlets(patterns, bed, output, window_size=7)
            self.assertEqual(output.read_text().splitlines(), [
                "chr2\t101\t103\tpos_patterns.pattern_0\t0\t+",
                "chr1\t10\t13\tpos_patterns.pattern_0\t0\t-",
                "chr2\t104\t108\tneg_patterns.pattern_0\t0\t-",
            ])

    def test_invalid_seqlet_coordinates_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bed = root / "loci.bed"
            bed.write_text("chr1\t10\t16\n")
            patterns = root / "patterns.h5"
            for index, start, end in ((1, 0, 2), (0, -1, 2), (0, 0, 7), (0, 2, 2)):
                with self.subTest(index=index, start=start, end=end):
                    with h5py.File(patterns, "w") as handle:
                        seqlets = handle.create_group("pos_patterns/pattern_0/seqlets")
                        for name, value in (("example_idx", index), ("start", start),
                                            ("end", end), ("is_revcomp", False)):
                            seqlets[name] = np.asarray([value])
                    with self.assertRaises(ValueError):
                        export_seqlets(patterns, bed, root / "seqlets.bed", window_size=6)
