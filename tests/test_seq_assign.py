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

class TestMultiSequenceFasta(unittest.TestCase):
    """Four protein chains, A and B with the same sequence (B has other residues unmodelled), C and D different, and a
    FASTA file with these three sequences and a decoy"""
    def setUp(self):
        self.wd = tempfile.mkdtemp(prefix="servaltest_")
        os.chdir(self.wd)
        self.X, self.Y, self.Z = msd.random_protein(30, 11), msd.random_protein(24, 12), msd.random_protein(18, 13)
        self.chains = [msd.chain("A", self.X, "AA", [1, 2]), msd.chain("B", self.X, "AA", [29, 30]),
                       msd.chain("C", self.Y, "AA", [10]), msd.chain("D", self.Z, "AA")]
        self.records = [("seqX", self.X), ("seqY", self.Y), ("seqZ", self.Z), ("decoy", msd.random_protein(28, 14))]

    def tearDown(self):
        os.chdir(root)
        shutil.rmtree(self.wd)

    def check_entity_ids(self, st):
        # the entity of every residue exists (chains merged into one entity must not keep the id of the removed one)
        for ch in st[0]:
            for r in ch:
                self.assertIsNotNone(st.get_entity(r.entity_id), (ch.name, r.seqid.num, r.entity_id))
        self.assertEqual({r.entity_id for r in st[0]["A"]}, {r.entity_id for r in st[0]["B"]})

    def test_homodimer_with_different_modelled_ranges(self):
        # regression: A and B get the same sequence and are merged into one entity; chain B kept the removed entity's id,
        # so that st.get_entity(residue.entity_id) was None (crash in find_and_fix_links)
        X = msd.random_protein(40, 21)
        st = msd.build([msd.chain("A", X, "AA"), msd.chain("B", X, "AA", [1, 2, 3, 4, 5, 6, 7])])
        rep = model.setup_entities(st, clear=True, force_subchain_names=True, overwrite_entity_type=True,
                                   fix_sequences=True, sequences=[("X", X)])
        self.assertEqual([(entry(rep, c)["name"], entry(rep, c)["entity"]) for c in "AB"], [("X", "A"), ("X", "A")])
        self.assertEqual(len([e for e in st.entities if e.entity_type == gemmi.EntityType.Polymer]), 1)
        self.check_entity_ids(st)
        from servalcat.utils import fileio
        fileio.write_model(st, "hd", pdb=True, cif=True)
        self.assertEqual(self.seqres("hd.pdb"), dict(A=X, B=X))
        ids = set(gemmi.cif.read("hd.mmcif").sole_block().find_values("_atom_site.label_entity_id"))
        self.assertEqual(len(ids), 1)

    def seqres(self, path):
        out = {}
        for l in open(path):
            if l.startswith("SEQRES"):
                out.setdefault(l[11], []).extend(l[19:].split())
        return {k: gemmi.one_letter_code(v) for k, v in out.items()}

    def test_every_chain_gets_its_own_sequence(self):
        st = msd.build(self.chains)
        autoseq = []
        rep = model.setup_entities(st, clear=True, force_subchain_names=True, overwrite_entity_type=True,
                                   fix_sequences=True, sequences=self.records, autoseq=autoseq)
        self.assertEqual(autoseq, [])
        names = {ch: entry(rep, ch)["name"] for ch in "ABCD"}
        self.assertEqual(names, dict(A="seqX", B="seqX", C="seqY", D="seqZ"))
        for ch in "ABCD":
            self.assertAlmostEqual(entry(rep, ch)["identity"], 1.0)
            self.assertTrue(labels_correct(st, ch), ch)
        self.assertEqual(entry(rep, "A")["unmodelled"], [[1, 2]])
        self.assertEqual(entry(rep, "B")["unmodelled"], [[29, 30]])
        self.check_entity_ids(st)
        from servalcat.utils import fileio
        fileio.write_model(st, "mc", pdb=True, cif=True, drop_seq=autoseq)
        self.assertEqual(self.seqres("mc.pdb"), dict(A=self.X, B=self.X, C=self.Y, D=self.Z))
        cif = gemmi.cif.read("mc.mmcif").sole_block()
        eps = cif.find("_entity_poly_seq.", ["entity_id", "num", "mon_id"])
        by_entity = {}
        for row in eps:
            by_entity.setdefault(row[0], []).append(row[2])
        self.assertEqual(sorted(len(v) for v in by_entity.values()), [18, 24, 30])
        self.assertEqual(sorted(gemmi.one_letter_code(v) for v in by_entity.values()),
                         sorted([self.X, self.Y, self.Z]))

    def test_chain_without_a_sequence_has_no_seqres(self):
        st = msd.build(self.chains)
        autoseq = []
        rep = model.setup_entities(st, clear=True, force_subchain_names=True, overwrite_entity_type=True,
                                   fix_sequences=True, sequences=[r for r in self.records if r[0] != "seqY"],
                                   autoseq=autoseq)
        self.assertIsNone(entry(rep, "C")["name"])
        self.assertEqual(len(autoseq), 1)
        from servalcat.utils import fileio
        fileio.write_model(st, "mc", pdb=True, cif=True, drop_seq=autoseq)
        self.assertEqual(self.seqres("mc.pdb"), dict(A=self.X, B=self.X, D=self.Z)) # nothing for chain C
        self.assertTrue(all(r.label_seq for r in st[0]["C"])) # still numbered, along the modelled residues

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

    def run_plain(self, prefix, *extra):
        sys.argv = ["", "refine_xtal", "--model", "start.pdb", "--hklin", "data.mtz", "-s", "xray",
                    "--labin", "FP,SIGFP,FreeR_flag", "--no_solvent", "--hydrogen", "no", "--ncycle", "1",
                    "-o", prefix] + list(extra)
        main()

    def test_no_seqres_without_a_sequence(self):
        # no input sequence: the output has neither SEQRES nor _entity_poly_seq, but still label_seq_id
        self.run_plain("ns")
        lines = open("ns.pdb").read().splitlines()
        self.assertFalse(any(l.startswith("SEQRES") for l in lines))
        self.assertIn("_entity_poly.entity_id", open("ns.mmcif").read())
        self.assertNotIn("_entity_poly_seq", open("ns.mmcif").read())
        cif = gemmi.read_structure("ns.mmcif")
        self.assertEqual([r.label_seq for r in cif[0]["A"]][:3], [1, 2, 3])

    def test_keep_entities_keeps_the_seqres_of_the_input(self):
        self.run_plain("ke", "--keep_entities")
        lines = open("ke.pdb").read().splitlines()
        self.assertTrue(any(l.startswith("SEQRES") for l in lines))
        self.assertIn("_entity_poly_seq", open("ke.mmcif").read())

    def test_seqres_with_a_sequence(self):
        st = gemmi.read_structure("start.pdb")
        st.setup_entities()
        full = gemmi.one_letter_code(st.entities[0].full_sequence)
        with open("c.fa", "w") as f:
            f.write(">c\n" + full + "\n")
        self.run_plain("ws", "--seq", "c.fa")
        self.assertTrue(any(l.startswith("SEQRES") for l in open("ws.pdb").read().splitlines()))
        self.assertIn("_entity_poly_seq", open("ws.mmcif").read())

if __name__ == '__main__':
    unittest.main()
