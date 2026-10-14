"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import os
import re
import shutil
import sys
import tempfile
import unittest
import servalcat # must be imported before gemmi (registers the bundled gemmi)
import gemmi
from servalcat.utils import model
from servalcat.__main__ import main

root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, root)
import make_seq_data as msd

class Log:
    def __init__(self): self.lines = []
    def writeln(self, s=""): self.lines.append(s)
    def write(self, s): self.lines.append(s)
    def text(self): return "\n".join(self.lines)

def run(chains, records, min_identity=0.8):
    st = msd.build(chains)
    rep = model.setup_entities(st, clear=True, force_subchain_names=True, overwrite_entity_type=True,
                               fix_sequences=True, sequences=records, min_seq_identity=min_identity)
    log = Log()
    model.log_sequence_report(rep, log)
    return st, rep, log

def entry(rep, chain):
    return [x for x in rep["entries"] if chain in x["chains"]][0]

def labels_correct(st, chain):
    return all(r.label_seq == r.seqid.num for r in st[0][chain])

class TestAssignment(unittest.TestCase):
    def test_four_chains_with_gaps(self):
        chains, recs = msd.four_chains()
        st, rep, log = run(chains, recs)
        for ch, name, n in (("A", "P1", 30), ("B", "P2", 26), ("C", "DNA", 12), ("D", "RNA", 12)):
            x = entry(rep, ch)
            self.assertEqual(x["name"], name)
            self.assertAlmostEqual(x["identity"], 1.0)
            self.assertEqual(x["n_seq"], n)
            self.assertTrue(labels_correct(st, ch), ch) # numbering follows the sequence across the gaps
        self.assertEqual(entry(rep, "A")["unmodelled"], [[1, 2], [11, 14]])
        self.assertEqual(entry(rep, "C")["unmodelled"], [[11, 12]])
        self.assertEqual([len(e.full_sequence) for e in st.entities], [30, 26, 12, 12])
        self.assertFalse(any(len(x[k]) for x in rep["entries"] for k in ("mismatches", "skipped", "extra", "shifts")))
        self.assertNotIn("WARNING", log.text())

    def test_mismatches_are_reported_and_numbering_is_kept(self):
        chains, recs = msd.four_chains()
        # Met5 -> MSE is not a mismatch; positions 20, 24 and 28 really differ
        chains[0]["names"] = {5: "MSE", 20: "GLY", 28: "TRP", 24: "LEU"}
        st, rep, log = run(chains, recs)
        x = entry(rep, "A")
        self.assertEqual(x["name"], "P1")
        self.assertEqual(sorted(int(m[1]) for m in x["mismatches"]), [20, 24, 28])
        self.assertEqual([m[2] for m in sorted(x["mismatches"], key=lambda m: int(m[1]))], ["GLY", "LEU", "TRP"])
        self.assertTrue(labels_correct(st, "A")) # gemmi's own numbering would leave residues without a label here
        self.assertIn("A/20", log.text())
        self.assertIn("differ from the assigned sequence (3)", log.text())

    def test_sequence_choice_by_alphabet(self):
        chains, recs = msd.four_chains()
        # the DNA sequence is missing: the DNA chain must not take the RNA sequence (same base order, U <-> T)
        st, rep, log = run(chains, [r for r in recs if r[0] != "DNA"])
        self.assertIsNone(entry(rep, "C")["name"])
        self.assertEqual(entry(rep, "D")["name"], "RNA")
        st, rep, log = run(chains, [r for r in recs if r[0] != "RNA"])
        self.assertIsNone(entry(rep, "D")["name"])
        self.assertEqual(entry(rep, "C")["name"], "DNA")

    def test_unassigned_chain_falls_back_to_modelled_sequence(self):
        chains, recs = msd.four_chains()
        st, rep, log = run(chains, [r for r in recs if r[0] != "P2"])
        x = entry(rep, "B")
        self.assertIsNone(x["name"])
        self.assertLess(x["identity"], 0.8)
        self.assertEqual(len(st.entities[1].full_sequence), 21) # modelled residues only
        self.assertIn("no sequence assigned", log.text())
        self.assertIn("closest is", log.text())
        self.assertTrue(all(r.label_seq for r in st[0]["B"])) # still numbered

    def test_min_identity(self):
        chains, recs = msd.four_chains()
        chains[0]["names"] = {20: "GLY", 28: "TRP"} # 22/24 identical = 91.7%
        self.assertEqual(entry(run(chains, recs, 0.9)[1], "A")["name"], "P1")
        self.assertIsNone(entry(run(chains, recs, 0.95)[1], "A")["name"])

    def test_unknown_residues_do_not_count_against_the_sequence(self):
        chains, recs = msd.four_chains()
        chains[1]["names"] = {3: "UNK", 4: "UNK", 5: "UNK", 6: "UNK"}
        st, rep, log = run(chains, recs)
        x = entry(rep, "B")
        self.assertEqual(x["name"], "P2")
        self.assertAlmostEqual(x["identity"], 1.0)
        self.assertEqual(len(x["unknown"]), 4)
        self.assertEqual(len(x["mismatches"]), 0)
        self.assertIn("4 unknown (UNK)", log.text())

    def test_register_shift_is_flagged(self):
        chains, recs = msd.four_chains()
        P = recs[0][1]
        # A: residues 17-27 carry the identities of residues 19-29 (a register error, chain continuous)
        chains[0] = msd.chain("A", P, "AA", [1, 2, 11, 12, 13, 14], shift=(17, 27, 2))
        st, rep, log = run(chains, recs)
        x = entry(rep, "A")
        self.assertEqual(x["name"], "P1")
        self.assertTrue(x["skipped"] or x["extra"] or x["shifts"], "a register error must be flagged")
        self.assertIn("WARNING", log.text())
        self.assertIn("register error", log.text())

    def test_identical_chains_each_get_the_sequence(self):
        chains, recs = msd.four_chains()
        P = recs[0][1]
        chains.append(msd.chain("E", P, "AA", [1, 2, 11, 12, 13, 14]))
        st, rep, log = run(chains, recs)
        for ch in "AE":
            self.assertEqual(entry(rep, ch)["name"], "P1")
            self.assertTrue(labels_correct(st, ch), ch)

    def test_fasta_with_headers(self):
        wd = tempfile.mkdtemp(prefix="servaltest_")
        try:
            chains, recs = msd.four_chains()
            f = os.path.join(wd, "seqs.fa")
            msd.write_fasta(f, recs, description="a long description of the construct")
            got = model.read_sequence_records([f])
            self.assertEqual([n for n, _ in got], [n for n, _ in recs])
            self.assertEqual([s for _, s in got], [s for _, s in recs])
        finally:
            shutil.rmtree(wd)

class TestRefineWithSeq(unittest.TestCase):
    def setUp(self):
        self.wd = tempfile.mkdtemp(prefix="servaltest_")
        os.chdir(self.wd)
        import make_tls_data
        make_tls_data.make(".") # 1l2h (start.pdb has the full SEQRES; residues 4-152 are modelled)

    def tearDown(self):
        os.chdir(root)
        shutil.rmtree(self.wd)

    def test_refine_xtal_seq(self):
        st = gemmi.read_structure("start.pdb")
        st.setup_entities()
        full = gemmi.one_letter_code(st.entities[0].full_sequence)
        self.assertEqual(len(full), 153)
        # construct with a 3-residue tag in front, and residue 50 substituted relative to the model
        pos = 50
        native = full[pos - 1]
        alt = "A" if native != "A" else "G"
        seq = "GSH" + full[:pos - 1] + alt + full[pos:]
        with open("construct.fa", "w") as f:
            f.write(">construct tagged and mutated\n" + seq + "\n>other\nMKKKLLLLAAAA\n")
        sys.argv = ["", "refine_xtal", "--model", "start.pdb", "--hklin", "data.mtz", "-s", "xray",
                    "--labin", "FP,SIGFP,FreeR_flag", "--no_solvent", "--hydrogen", "no", "--ncycle", "1",
                    "--seq", "construct.fa", "-o", "sq"]
        main()
        log = open("sq.log").read()
        self.assertIn("Sequence assignment from construct.fa", log)
        self.assertRegex(log, r"A\s+protein\s+construct\s+156")
        self.assertRegex(log, r"A/{}\s+model\s+\w+\s+sequence\s+\w+\s+\(construct\)".format(pos))
        self.assertIn("differ from the assigned sequence (1)", log)
        # the summary at the end of the run repeats the sequence alignment and shows the progress of the refinement
        tail = log[log.index("Refinement summary"):]
        self.assertIn("= ADP analysis =", tail)
        self.assertIn("Sequence alignment (--seq)", tail)
        self.assertIn("A/{}".format(pos), tail)
        for name in ("R-work", "R-free", "zBond", "zAngle"):
            self.assertIn(name, tail)
        self.assertNotIn("\x1b", log)
        out = gemmi.read_structure("sq.pdb")
        out.setup_entities()
        self.assertEqual(len(out.entities[0].full_sequence), 156) # SEQRES now has the tag and the unmodelled ends
        cif = gemmi.read_structure("sq.mmcif")
        r4 = [r for r in cif[0]["A"] if r.seqid.num == 4][0]
        self.assertEqual(r4.label_seq, 4 + 3) # numbering follows the construct

if __name__ == '__main__':
    unittest.main()
