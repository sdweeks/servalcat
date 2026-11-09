"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import os
import shutil
import sys
import tempfile
import unittest
import numpy
import omegaconf
import servalcat # must be imported before gemmi (registers the bundled gemmi)
import gemmi
from servalcat.refine import occ_groups, summary
from servalcat.refine.refine import RefineConfig
from servalcat.__main__ import main

root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, root)
import make_occ_data
import make_seq_data as msd

def groups_of(res, kind):
    return [g for g in res.groups if g.kind == kind]

class TestFindGroups(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wd = tempfile.mkdtemp(prefix="servaltest_")
        cls.info = make_occ_data.make(cls.wd)
        cls.st = gemmi.read_structure(os.path.join(cls.wd, "start.pdb"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.wd)

    def test_alternative_conformations_and_partial_occupancies(self):
        res = occ_groups.find_occupancy_groups(self.st)
        alt = groups_of(res, "alt")
        # 1l2h has real altlocs at 11, 43 and 45; the test data add three pairs and a lone conformer
        resis = sorted({int(g.resi) for g in alt})
        self.assertEqual(resis, sorted([11, 43, 45] + self.info["pairs"] + [self.info["lone"]]))
        self.assertEqual(len(alt), 2 * 6 + 1)
        self.assertEqual([int(g.resi) for g in groups_of(res, "partial")], [self.info["partial"]])
        self.assertEqual([int(g.resi) for g in groups_of(res, "water")], [self.info["water"]])
        # constraints: two conformers sum to 1; a lone conformer, partial atoms and a water are <= 1
        by_ids = {tuple(c.ids): c.complete for c in res.constraints}
        for r in [11, 43, 45] + self.info["pairs"]:
            ids = tuple(g.id for g in alt if int(g.resi) == r)
            self.assertEqual(len(ids), 2)
            self.assertTrue(by_ids[ids], r)
        lone = [g for g in alt if int(g.resi) == self.info["lone"]]
        self.assertEqual(len(lone), 1)
        self.assertFalse(by_ids[(lone[0].id,)])
        self.assertEqual(sum(1 for c in res.constraints if not c.complete), 3)   # lone, partial, water
        # every group selects atoms, and no atom is in two groups
        seen = set()
        for g in res.groups:
            n = 0
            for sel_str in g.selections:
                sel = gemmi.Selection(sel_str)
                for m in sel.models(self.st):
                    for c in sel.chains(m):
                        for r in sel.residues(c):
                            for a in sel.atoms(r):
                                key = (c.name, str(r.seqid), a.name, a.altloc)
                                self.assertNotIn(key, seen, g.selections)
                                seen.add(key)
                                n += 1
            self.assertEqual(n, g.natoms, g.selections)
        # the lone-conformer note and the table are produced
        self.assertTrue(any("only conformer A" in n for n in res.notes))
        self.assertIn("HOH", "\n".join(occ_groups.residue_table(res)))

    def test_water_policy(self):
        n_water = sum(1 for r in self.st[0]["A"] if r.is_water())
        self.assertEqual(len(groups_of(occ_groups.find_occupancy_groups(self.st, waters="none"), "water")), 0)
        self.assertEqual(len(groups_of(occ_groups.find_occupancy_groups(self.st, waters="partial"), "water")), 1)
        self.assertEqual(len(groups_of(occ_groups.find_occupancy_groups(self.st, waters="all"), "water")), n_water)

    def test_alternative_conformers_incomplete(self):
        res = occ_groups.find_occupancy_groups(self.st, alts_complete=False)
        self.assertFalse(any(c.complete for c in res.constraints))

    def test_description_is_a_valid_refinement_config(self):
        res = occ_groups.find_occupancy_groups(self.st)
        d = occ_groups.to_config_dict(res, adpr_weight=5.)["refine"]
        cfg = omegaconf.OmegaConf.merge(omegaconf.OmegaConf.structured(RefineConfig), d)   # validates the types
        self.assertEqual(len(cfg.occ_groups), len(res.groups))
        self.assertEqual(len(cfg.occ_group_constraints), len(res.constraints))
        self.assertEqual(list(cfg.atom_selection.occ.include), res.include)
        self.assertTrue(all(x.w == 5. for x in cfg.local_adpr_weights))
        self.assertNotIn("local_adpr_weights", occ_groups.to_config_dict(res, adpr_weight=0)["refine"])

    def test_refuses_to_mix_with_groups_of_the_user(self):
        res = occ_groups.find_occupancy_groups(self.st)
        cfg = omegaconf.OmegaConf.structured(RefineConfig)
        occ_groups.apply_to_config(cfg, res, 5.)
        self.assertEqual(len(cfg.occ_groups), len(res.groups))
        with self.assertRaises(SystemExit):
            occ_groups.apply_to_config(cfg, res, 5.)   # groups exist now

    def test_nothing_found_in_a_model_without_alternatives(self):
        st = msd.build([msd.chain("A", "ACDEFGHIKL", "AA")])
        res = occ_groups.find_occupancy_groups(st)
        self.assertEqual(res.groups, [])

    def test_insertion_codes_negative_numbers_and_microheterogeneity(self):
        st = msd.build([msd.chain("A", "ACDEFGHIKL", "AA")])
        ch = st[0]["A"]
        ch[1].seqid = gemmi.SeqId(-3, " ")
        ch[2].seqid = gemmi.SeqId(20, "B")
        for r, alts in ((ch[1], ("A", "B")), (ch[2], ("A", "B"))):
            for atom in list(r):
                if atom.name == "CA":
                    atom.altloc = "A"
                    atom.occ = 0.6
                    b = atom.clone()
                    b.altloc = "B"
                    b.occ = 0.4
                    r.add_atom(b)
        res = occ_groups.find_occupancy_groups(st)
        sels = [g.selections[0] for g in res.groups]
        self.assertEqual(sorted(sels), sorted(["//A/-3/*:A", "//A/-3/*:B", "//A/20B/*:A", "//A/20B/*:B"]))
        for g in res.groups:
            sel = gemmi.Selection(g.selections[0])
            atoms = [a for m in sel.models(st) for c in sel.chains(m) for r in sel.residues(c) for a in sel.atoms(r)]
            self.assertEqual(len(atoms), 1)
            self.assertEqual(atoms[0].altloc, g.alt)

class TestRefineOcc(unittest.TestCase):
    def setUp(self):
        self.wd = tempfile.mkdtemp(prefix="servaltest_")
        os.chdir(self.wd)
        self.info = make_occ_data.make(".")

    def tearDown(self):
        os.chdir(root)
        shutil.rmtree(self.wd)

    def argv(self, extra, prefix):
        return ["", "refine_xtal", "--model", "start.pdb", "--hklin", "data.mtz", "--labin", "FP,SIGFP,FreeR_flag",
                "--no_solvent", "--hydrogen", "no", "--fix_xyz", "-o", prefix] + extra

    def occ_of(self, st, num, altloc=None, side_chain=True):
        r = [r for r in st[0]["A"] if r.seqid.num == num][0]
        vals = [a.occ for a in r if (altloc is None or a.altloc == altloc) and (not side_chain or a.name not in make_occ_data.MAIN_CHAIN)]
        self.assertTrue(len(vals) > 0)
        self.assertLess(numpy.ptp(vals), 1e-3) # all atoms of a group share the occupancy
        return float(numpy.mean(vals))

    def test_refine_occ_recovers_the_true_occupancies(self):
        sys.argv = self.argv(["--ncycle", "30", "--refine_occ"], "occ")
        main()
        st = gemmi.read_structure("occ.pdb")
        for num in self.info["pairs"]:
            a, b = self.occ_of(st, num, "A"), self.occ_of(st, num, "B")
            self.assertAlmostEqual(a, 0.7, delta=0.06, msg=str(num))
            self.assertAlmostEqual(b, 0.3, delta=0.06, msg=str(num))
            self.assertAlmostEqual(a + b, 1.0, delta=0.02, msg=str(num))
        # a lone conformer must not be pushed to 1 by the constraints of other residues
        self.assertAlmostEqual(self.occ_of(st, self.info["lone"], "A"), self.info["true_lone"], delta=0.06)
        self.assertAlmostEqual(self.occ_of(st, self.info["partial"]), self.info["true_partial"], delta=0.06)
        self.assertAlmostEqual(self.occ_of(st, self.info["water"], side_chain=False), self.info["true_water"], delta=0.06)
        log = open("occ.log").read()
        self.assertIn("Occupancy groups found in the input model", log)
        self.assertNotIn("non-binary solvent mask", log) # only advised, and set, when there is a solvent
        self.assertNotIn("\x1b", log)                 # no colour codes in the log file
        self.assertIn("Refined occupancies", log)
        self.assertIn("Refinement summary", log)
        self.assertTrue(os.path.isfile("occ_occ.yaml"))
        # the utility writes the same description
        sys.argv = ["", "util", "occ_groups", "--model", "start.pdb", "-o", "utilocc"]
        main()
        self.assertEqual(open("utilocc.yaml").read(), open("occ_occ.yaml").read())

    def test_config_from_the_utility_is_equivalent(self):
        sys.argv = ["", "util", "occ_groups", "--model", "start.pdb", "-o", "u"]
        main()
        sys.argv = self.argv(["--ncycle", "6", "--config", "u.yaml"], "viaconfig")
        main()
        sys.argv = self.argv(["--ncycle", "6", "--refine_occ"], "viaflag")
        main()
        a = numpy.array([x.atom.occ for x in gemmi.read_structure("viaconfig.pdb")[0].all()])
        b = numpy.array([x.atom.occ for x in gemmi.read_structure("viaflag.pdb")[0].all()])
        self.assertLess(numpy.max(numpy.abs(a - b)), 1e-6)

    def test_flag_and_user_groups_are_not_mixed(self):
        sys.argv = ["", "util", "occ_groups", "--model", "start.pdb", "-o", "u"]
        main()
        sys.argv = self.argv(["--ncycle", "1", "--refine_occ", "--config", "u.yaml"], "mixed")
        with self.assertRaises(SystemExit):
            main()

class TestSummary(unittest.TestCase):
    def test_sparkline(self):
        self.assertEqual(summary.sparkline([1, 2, 3, 4, 5, 6, 7, 8]), "▁▂▃▄▅▆▇█")
        self.assertEqual(summary.sparkline([0.3, 0.3, 0.3]), "▁▁▁")   # flat
        self.assertEqual(summary.sparkline([5, 1]), "█▁")
        self.assertEqual(summary.sparkline([]), "")

    def test_series_and_rows(self):
        def cyc(rw, rf, zb, za):
            return {"data": {"summary": {"Rwork": rw, "Rfree": rf}},
                    "geom": {"summary": {"r.m.s.Z": {"Bond distances, non H": zb, "Bond angles, non H": za}}}}
        stats = [cyc(0.30, 0.32, 1.5, 1.8), cyc(0.25, 0.29, 1.0, 1.4), cyc(0.20, 0.27, 0.8, 1.1)]
        s = summary.series_from_stats(stats)
        self.assertEqual(list(s), ["R-work", "R-free", "zBond", "zAngle"])
        self.assertEqual(s["R-free"], [0.32, 0.29, 0.27])
        rows = summary.sparkline_rows(s, 2)
        self.assertEqual(len(rows), 4)
        plain = rows[0][0]
        self.assertIn("R-work", plain)
        self.assertIn("0.3000 → 0.2000", plain)
        self.assertIn("↓ 0.1000", plain)
        self.assertNotIn("\x1b", plain)
        self.assertIn("\x1b", rows[0][1])
        # without geometry statistics only the R-values are shown; intensity-based names are handled
        s = summary.series_from_stats([{"data": {"summary": {"R1work": 0.2, "R1free": 0.25}}}, {"data": {"summary": {"R1work": 0.1, "R1free": 0.2}}}])
        self.assertEqual(list(s), ["R-work", "R-free"])
        self.assertEqual(summary.series_from_stats([]), {})

    @unittest.skipUnless(__import__("importlib").util.find_spec("matplotlib"), "matplotlib not installed")
    def test_png(self):
        wd = tempfile.mkdtemp(prefix="servaltest_")
        try:
            path = os.path.join(wd, "s.png")
            self.assertTrue(summary.write_png(path, {"R-work": [0.3, 0.2], "R-free": [0.3, 0.25], "zBond": [1.2, 0.9], "zAngle": [1.5, 1.2]}))
            self.assertGreater(os.path.getsize(path), 1000)
        finally:
            shutil.rmtree(wd)

if __name__ == '__main__':
    unittest.main()
