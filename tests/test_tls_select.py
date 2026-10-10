"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import unittest
import itertools
import json
import os
import re
import shutil
import sys
import tempfile
import numpy
import pandas
import servalcat # must be imported before gemmi (registers the bundled gemmi)
import gemmi
from servalcat import utils
from servalcat.refine import tls_select
from servalcat.__main__ import main

root = os.path.abspath(os.path.dirname(__file__))

class TestSelectionLogic(unittest.TestCase):
    def test_dynamic_programming_matches_brute_force(self):
        rng = numpy.random.default_rng(2)
        for min_res in (1, 3):
            bounds = [0, 2, 3, 6, 8, 11, 13, 14, 17]
            m = len(bounds)
            gain = rng.normal(size=(m, m)) * 5
            res = tls_select.best_partitions(gain, bounds, min_res, 5)
            for k in range(1, 6):
                best = None
                for inner in itertools.combinations(range(1, m - 1), k - 1):
                    idx = (0,) + inner + (m - 1,)
                    if any(bounds[b] - bounds[a] < min_res for a, b in zip(idx[:-1], idx[1:])):
                        continue
                    total = sum(gain[a, b] for a, b in zip(idx[:-1], idx[1:]))
                    if best is None or total > best[0]:
                        best = (total, [bounds[i] for i in idx])
                if best is None:
                    self.assertNotIn(k, res)
                else:
                    self.assertAlmostEqual(res[k][0], best[0])
                    self.assertEqual(res[k][1], best[1])

    def test_candidate_boundaries(self):
        b = tls_select.candidate_boundaries(50, 120)
        self.assertEqual(b, list(range(51))) # all residue boundaries when few residues
        b = tls_select.candidate_boundaries(1000, 120)
        self.assertEqual(b[0], 0)
        self.assertEqual(b[-1], 1000)
        self.assertLessEqual(len(b), 130)

    def test_choose_n(self):
        c = tls_select.choose_n
        # smooth curve: biggest jump after the first group is at 2 groups
        self.assertEqual(c([0.30, 0.295, 0.290, 0.289, 0.288])[0], 2)
        # a clear jump when the 4th group is added
        self.assertEqual(c([0.30, 0.29, 0.289, 0.288, 0.270, 0.269])[0], 4)
        # TLS does not help at all
        self.assertEqual(c([0.30, 0.2995, 0.299, 0.298])[0], 0)
        # one group helps, more do not
        self.assertEqual(c([0.30, 0.28, 0.2795, 0.2792])[0], 1)
        # near ties are resolved towards the lowest number of groups
        self.assertEqual(c([0.30, 0.28, 0.270, 0.2595])[0], 2)
        self.assertEqual(c([0.30, 0.28, 0.270, 0.2500])[0], 3) # 3rd jump is clearly bigger
        # only one number evaluated
        self.assertEqual(c([0.30, 0.28])[0], 1)
        self.assertEqual(c([0.30])[0], 0)

    def test_parse_ngroups(self):
        p = tls_select.parse_ngroups
        self.assertEqual(p(["3"], ["A", "B"]), {"A": 3, "B": 3})
        self.assertEqual(p(["A:4", "B=2"], ["A", "B"]), {"A": 4, "B": 2})
        self.assertEqual(p(["A:4"], ["A", "B"]), {"A": 4, "B": 1}) # chains not mentioned get one group
        self.assertEqual(p(["2", "B:5"], ["A", "B"]), {"A": 2, "B": 5})
        for bad in (["0"], ["abc"], ["C:2"], ["A:"]):
            with self.assertRaises(SystemExit):
                p(bad, ["A", "B"])

class TestTlsSelect(unittest.TestCase):
    def setUp(self):
        self.wd = tempfile.mkdtemp(prefix="servaltest_")
        os.chdir(self.wd)
        print("In", self.wd)
        sys.path.insert(0, root)
        import make_tls_data
        make_tls_data.make(".") # 1l2h with two TLS groups: residues 4-80 and 81-164 (polymer: 4-152)

    def tearDown(self):
        os.chdir(root)
        shutil.rmtree(self.wd)

    def run_select(self, extra, prefix="sel"):
        sys.argv = ["", "util", "tls_select", "--model", "start.pdb", "--hklin", "data.mtz", "-s", "xray",
                    "--labin", "FP,SIGFP,FreeR_flag", "--no_solvent", "--hydrogen", "no", "-o", prefix] + extra
        main()

    def read_ranges(self, path):
        import omegaconf
        cfg = omegaconf.OmegaConf.load(path)
        out = []
        for g in cfg.refine.tls.groups:
            self.assertEqual(len(g.selections), 1)
            m = re.match(r"^//(\w+)/(-?\d+)-(-?\d+)/\*$", g.selections[0])
            self.assertIsNotNone(m, g.selections[0])
            out.append((m.group(1), int(m.group(2)), int(m.group(3))))
        return out

    def test_ngroups_finds_domain_boundary(self):
        # the gradient information alone places the boundary of two groups at the true domain boundary (80|81)
        self.run_select(["--ngroups", "2", "--no_eval"])
        rng = self.read_ranges("sel_tls.yaml")
        self.assertEqual(len(rng), 2)
        self.assertEqual(rng[0][:2], ("A", 4))
        self.assertEqual(rng[1][2], 152)
        self.assertLessEqual(abs(rng[0][2] - 80), 3, msg=str(rng))
        self.assertEqual(rng[1][1], rng[0][2] + 1)

    def test_ngroups_per_chain_ranges_tile_the_chain(self):
        self.run_select(["--ngroups", "A:4", "--min_residues", "15", "--no_eval"])
        rng = self.read_ranges("sel_tls.yaml")
        self.assertEqual(len(rng), 4)
        st = utils.fileio.read_structure("start.pdb")
        st.setup_entities()
        nums = [r.seqid.num for r in st[0]["A"].get_polymer()]
        self.assertGreater(len(nums), 100)
        covered = []
        for _, a, b in rng:
            seg = [n for n in nums if a <= n <= b]
            self.assertGreaterEqual(len(seg), 15)
            covered += seg
        self.assertEqual(covered, nums) # every polymer residue exactly once, in order

    def test_scan_selects_two_groups(self):
        # R-values as a function of the number of groups; the data come from two rigid groups
        self.run_select(["--max_groups", "3", "--tls_cycles", "5", "-d", "2.0"], prefix="scan")
        df = pandas.read_csv("scan_scan.csv")
        self.assertEqual(list(df.N), [0, 1, 2, 3])
        self.assertTrue(numpy.all(numpy.diff(df.Rfree) < 0.0005)) # more groups do not make Rfree worse
        self.assertGreater(df.Rfree[0] - df.Rfree[1], 0.01)     # TLS helps
        self.assertGreater(df.Rfree[1] - df.Rfree[2], 0.005)    # the second group helps a lot
        self.assertLess(df.Rfree[2] - df.Rfree[3], 0.003)       # a third does not
        with open("scan_scan.json") as f:
            js = json.load(f)
        self.assertEqual(js["selected_N"], 2)
        for fn in ("scan_tls.yaml", "scan_tls_N1.yaml", "scan_tls_N2.yaml", "scan_tls_N3.yaml"):
            self.assertTrue(os.path.isfile(fn), fn)
        self.assertEqual(len(self.read_ranges("scan_tls.yaml")), 2)

class TestTlsSelectTwoChains(unittest.TestCase):
    def setUp(self):
        self.wd = tempfile.mkdtemp(prefix="servaltest_")
        os.chdir(self.wd)
        sys.path.insert(0, root)
        import make_tls_data
        self.split = make_tls_data.make_two_chain(".") # chain A: one group, chain B: two groups meeting here

    def tearDown(self):
        os.chdir(root)
        shutil.rmtree(self.wd)

    def test_ngroups_per_chain(self):
        # different numbers of groups for different chains; the boundary in chain B is found
        import omegaconf
        sys.argv = ["", "util", "tls_select", "--model", "start.pdb", "--hklin", "data.mtz", "-s", "xray",
                    "--labin", "FP,SIGFP,FreeR_flag", "--no_solvent", "--hydrogen", "no", "--ngroups", "A:1", "B:2",
                    "--no_eval", "-o", "sel"]
        main()
        sels = [g.selections[0] for g in omegaconf.OmegaConf.load("sel_tls.yaml").refine.tls.groups]
        self.assertEqual(len(sels), 3)
        self.assertTrue(sels[0].startswith("//A/"))
        self.assertTrue(sels[1].startswith("//B/") and sels[2].startswith("//B/"))
        b_end = int(re.match(r"^//B/\d+-(\d+)/\*$", sels[1]).group(1))
        self.assertLessEqual(abs(b_end + 1 - self.split), 3, msg=str(sels))

if __name__ == '__main__':
    unittest.main()
