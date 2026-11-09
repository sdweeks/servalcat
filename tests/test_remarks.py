"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.

REMARK 3 and the _refine items of the refined models
"""
from __future__ import absolute_import, division, print_function, generators
import json
import math
import os
import re
import shutil
import sys
import tempfile
import unittest
import servalcat # must be imported before gemmi (registers the bundled gemmi)
import gemmi
from servalcat import utils
from servalcat.refine import remarks
from servalcat.__main__ import main

root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, root)

def remark3(path):
    return [l.rstrip() for l in open(path) if l.startswith("REMARK   3")]

def value(lines, label):
    """value after the colon of the first line that contains label"""
    for l in lines:
        if label in l:
            return l.split(" : ", 1)[1].strip()
    raise KeyError(label)

class TestHelpers(unittest.TestCase):
    def test_null(self):
        self.assertEqual(remarks.null(None), "NULL")
        self.assertEqual(remarks.null(float("nan")), "NULL")
        self.assertEqual(remarks.null(0.21, "{:.3f}"), "0.210")

    def test_squeeze(self):
        self.assertEqual(remarks.squeeze(["REMARK   3", "REMARK   3", "REMARK   3 A", "REMARK   3", "REMARK   3"]),
                         ["REMARK   3", "REMARK   3 A", "REMARK   3"])

    def test_cruickshank_dpi(self):
        # sqrt(N / (Nobs - 4N)) * C**(-1/3) * R * dmin
        v = remarks.cruickshank_dpi(0.2, 2.0, 1000, 20000, 0.9)
        self.assertAlmostEqual(v, math.sqrt(1000 / 16000.) * 0.9 ** (-1. / 3) * 0.2 * 2.0)
        self.assertIsNone(remarks.cruickshank_dpi(0.2, 2.0, 1000, 3000, 0.9)) # more parameters than data
        self.assertIsNone(remarks.cruickshank_dpi(None, 2.0, 1000, 20000, 0.9))
        self.assertIsNone(remarks.cruickshank_dpi(0.2, 2.0, 1000, 20000, 0))

    def test_atom_counts(self):
        st = utils.fileio.read_structure(os.path.join(root, "1l2h", "1l2h.cif.gz"))
        st.setup_entities()
        n = remarks.atom_counts(st)
        heavy = sum(1 for cra in st[0].all() if not cra.atom.is_hydrogen())
        self.assertEqual(sum(n.values()), heavy)
        self.assertEqual(n["nucleic"], 0)
        self.assertEqual(n["protein"], sum(1 for cra in st[0].all() if cra.residue.name != "HOH" and not cra.atom.is_hydrogen()
                                           and gemmi.find_tabulated_residue(cra.residue.name).is_amino_acid()))
        self.assertEqual(n["solvent"], sum(1 for cra in st[0].all() if cra.residue.name == "HOH"))

    def test_atom_counts_kinds(self):
        st = gemmi.Structure()
        model = gemmi.Model(1)
        ch = gemmi.Chain("A")
        for name, atoms in (("ALA", ["N", "CA", "C", "O", "CB"]), ("DA", ["P", "OP1"]), ("HOH", ["O"]),
                            ("ZN", ["ZN"]), ("GOL", ["C1", "O1", "H1"])):
            r = gemmi.Residue()
            r.name = name
            r.seqid = gemmi.SeqId(len(ch) + 1, " ")
            for an in atoms:
                a = gemmi.Atom()
                a.name = an
                a.element = gemmi.Element(an[0] if an != "ZN" else "Zn")
                r.add_atom(a)
            ch.add_residue(r)
        model.add_chain(ch)
        st.add_model(model)
        self.assertEqual(remarks.atom_counts(st), dict(protein=5, nucleic=2, hetero=1 + 2, solvent=1))

class TestRefinedModelHeader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wd = tempfile.mkdtemp(prefix="servaltest_")
        cls.cwd = os.getcwd()
        os.chdir(cls.wd)
        import make_tls_data
        make_tls_data.make(".") # 1l2h: start.pdb, data.mtz (FP, SIGFP, FreeR_flag)
        st = gemmi.read_structure("start.pdb")
        st.setup_entities()
        full = gemmi.one_letter_code(st.entities[0].full_sequence)
        with open("c.fa", "w") as f:
            f.write(">c\n" + full + "\n")

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.cwd)
        shutil.rmtree(cls.wd)

    def run_xtal(self, prefix, *extra):
        sys.argv = ["", "refine_xtal", "--model", "start.pdb", "--hklin", "data.mtz", "-s", "xray",
                    "--labin", "FP,SIGFP,FreeR_flag", "--hydrogen", "no", "--ncycle", "2", "-o", prefix] + list(extra)
        main()

    def test_remark3_content(self):
        self.run_xtal("hd", "--tls_per_chain")
        lines = remark3("hd.pdb")
        stats = json.load(open("hd_stats.json"))[-1]
        pdb = [l.rstrip() for l in open("hd.pdb")]
        self.assertTrue(any(l.startswith("REMARK   2 RESOLUTION.") for l in pdb))
        self.assertTrue(all(len(l) <= 80 for l in pdb))
        text = "\n".join(lines)
        for title in ("REFINEMENT.", "DATA USED IN REFINEMENT.", "FIT TO DATA USED IN REFINEMENT.",
                      "FIT IN THE HIGHEST RESOLUTION BIN.", "NUMBER OF NON-HYDROGEN ATOMS USED IN REFINEMENT.",
                      "B VALUES.", "ESTIMATED OVERALL COORDINATE ERROR.", "CORRELATION COEFFICIENTS",
                      "RMS DEVIATIONS FROM IDEAL VALUES", "TLS DETAILS", "BULK SOLVENT MODELLING.", "OTHER REFINEMENT REMARKS:"):
            self.assertIn(title, text)
        # order of a REFMAC file: the TLS section comes before the bulk solvent
        self.assertLess(text.index("TLS DETAILS"), text.index("BULK SOLVENT MODELLING."))
        self.assertNotIn("\nREMARK   3\nREMARK   3\n", text)
        self.assertNotIn("nan", text.lower().replace("maximum", ""))
        # values agree with the statistics of the last cycle
        summ = stats["data"]["summary"]
        self.assertAlmostEqual(float(value(lines, "R VALUE            (WORKING SET)")), summ["Rwork"], places=3)
        self.assertAlmostEqual(float(value(lines, "FREE R VALUE   ")), summ["Rfree"], places=3)
        bins = stats["data"]["binned"]
        self.assertEqual(int(value(lines, "TOTAL NUMBER OF BINS USED")), len(bins))
        hi = min(bins, key=lambda b: b["d_min"])
        self.assertEqual(int(value(lines, "REFLECTION IN BIN")), hi["n_work"])
        self.assertEqual(int(value(lines, "BIN FREE R VALUE SET COUNT")), hi["n_free"])
        n_all = sum(b["n_obs"] for b in bins)
        self.assertEqual(int(value(lines, "NUMBER OF REFLECTIONS")), n_all)
        self.assertEqual(int(value(lines, "FREE R VALUE TEST SET COUNT")), sum(b["n_free"] for b in bins))
        st = gemmi.read_structure("hd.pdb")
        n = remarks.atom_counts(st)
        self.assertEqual(int(value(lines, "PROTEIN ATOMS")), n["protein"])
        self.assertEqual(int(value(lines, "SOLVENT ATOMS")), n["solvent"])
        # coordinate error: Cruickshank's formula applied to the printed numbers
        c = n_all / sum(b["n_all"] for b in bins)
        want = remarks.cruickshank_dpi(summ["Rwork"], min(b["d_min"] for b in bins), sum(n.values()), n_all, c)
        self.assertAlmostEqual(float(value(lines, "BASED ON R VALUE")), want, places=2)
        self.assertLess(float(value(lines, "BASED ON R VALUE")), float(value(lines, "BASED ON FREE R VALUE")) + 1e-9)
        self.assertEqual(value(lines, "METHOD USED"), "MASK BULK SOLVENT")
        # restraint rows keep the numbers of the geometry statistics
        gs = stats["geom"]["summary"]
        row = [l for l in lines if "BOND LENGTHS REFINED ATOMS" in l][0]
        self.assertIn(f"{gs['N restraints']['Bond distances, non H']:6d}", row)

    def test_mmcif_items(self):
        self.run_xtal("hm")
        doc = gemmi.cif.read("hm.mmcif")
        block = doc.sole_block()
        hist = block.find("_refine_hist.", ["pdbx_number_atoms_protein", "pdbx_number_atoms_nucleic_acid",
                                            "pdbx_number_atoms_ligand", "number_atoms_solvent", "number_atoms_total"])
        self.assertEqual(len(hist), 1)
        vals = [int(x) for x in hist[0]]
        st = gemmi.read_structure("hm.mmcif")
        n = remarks.atom_counts(st)
        self.assertEqual(vals, [n["protein"], n["nucleic"], n["hetero"], n["solvent"], sum(n.values())])
        dpi = block.find_value("_refine.overall_SU_R_Cruickshank_DPI")
        dpif = block.find_value("_refine.pdbx_overall_SU_R_free_Cruickshank_DPI")
        self.assertIsNotNone(dpi)
        self.assertGreater(float(dpi), 0)
        self.assertGreaterEqual(float(dpif), float(dpi))
        self.assertIsNotNone(block.find_value("_refine.correlation_coeff_Fo_to_Fc"))

    def test_header_does_not_depend_on_seq(self):
        # the same REMARK 3 with and without a sequence file
        self.run_xtal("sa")
        self.run_xtal("sb", "--seq", "c.fa")
        a, b = remark3("sa.pdb"), remark3("sb.pdb")
        self.assertEqual(a, b)
        self.assertTrue(any("DATA USED IN REFINEMENT." in l for l in b))

    def test_refine_geom(self):
        sys.argv = ["", "refine_geom", "--model", "start.pdb", "--ncycle", "1", "-o", "geo"]
        main()
        text = "\n".join(remark3("geo.pdb"))
        self.assertIn("NUMBER OF NON-HYDROGEN ATOMS USED IN REFINEMENT.", text)
        self.assertIn("MEAN B VALUE", text)
        self.assertIn("RMS DEVIATIONS FROM IDEAL VALUES", text)
        self.assertNotIn("DATA USED IN REFINEMENT.", text) # no reflection data in a geometry refinement

if __name__ == '__main__':
    unittest.main()
