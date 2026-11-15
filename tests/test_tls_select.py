"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import argparse
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
try:
    import matplotlib
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

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

    def check_domain_files(self, csv, pdb, yaml, expect_groups):
        """the csv and the pdb describe the same groups as the yaml"""
        import omegaconf
        df = pandas.read_csv(csv)
        self.assertEqual(sorted(set(df.domain)), list(range(1, expect_groups + 1)))
        cfg = omegaconf.OmegaConf.load(yaml)
        for gi, g in enumerate(cfg.refine.tls.groups):
            self.assertEqual(g.id, gi + 1)
            for sel_str in g.selections:
                m = re.match(r"^//(\w+)/(-?\d+)-(-?\d+)/\*$", sel_str)
                chain, a, b = m.group(1), int(m.group(2)), int(m.group(3))
                sub = df[(df.chain == chain) & (df.resi.astype(int).between(a, b))]
                self.assertTrue(len(sub) > 0)
                self.assertEqual(set(sub.domain), {gi + 1}, sel_str)       # the csv agrees with the yaml
        if pdb:
            st = gemmi.read_structure(pdb)
            self.assertFalse(st[0].has_hydrogen())
            by_res = {(ch.name, r.seqid.num): {round(a.b_iso) for a in r} for ch in st[0] for r in ch}
            for _, row in df.iterrows():
                self.assertEqual(by_res[(row.chain, int(row.resi))], {int(row.domain)})   # B column = group number
            self.assertEqual({v for vals in by_res.values() for v in vals} - {0}, set(range(1, expect_groups + 1)))

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
        self.check_domain_files("sel_domains.csv", "sel_domains.pdb", "sel_tls.yaml", 2)

    def test_bfit_with_reflections_reports_r_values(self):
        # partitions from the B-factors, every number of groups tested by refinement (the model has no TLS in its B,
        # so only the plumbing is checked here)
        sys.argv = ["", "util", "tls_bfit", "--model", "start.pdb", "--hklin", "data.mtz", "-s", "xray",
                    "--labin", "FP,SIGFP,FreeR_flag", "--no_solvent", "--hydrogen", "no", "--max_groups", "2",
                    "--tls_cycles", "3", "-d", "2.5", "-o", "bfev"]
        main()
        df = pandas.read_csv("bfev_scan.csv")
        self.assertEqual(list(df.N), [0, 1, 2])
        for k in ("Rwork", "Rfree", "B_rms"):
            self.assertIn(k, df.columns)
        self.assertGreater(df.Rfree[0] - df.Rfree[1], 0.005) # TLS helps
        self.assertTrue(os.path.isfile("bfev_tls_N2.yaml"))
        self.assertFalse(os.path.exists("bfev_trials")) # removed after the trial refinements
        if HAVE_MPL:
            self.assertTrue(os.path.isfile("bfev_scan.png"))

    def test_source_defaults_to_xray(self):
        # no -s/--source: X-ray is the default
        sys.argv = ["", "util", "tls_select", "--model", "start.pdb", "--hklin", "data.mtz",
                    "--labin", "FP,SIGFP,FreeR_flag", "--no_solvent", "--hydrogen", "no", "--ngroups", "2", "--no_eval",
                    "-o", "nosrc"]
        main()
        self.assertEqual(len(self.read_ranges("nosrc_tls.yaml")), 2)

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
        self.assertFalse(os.path.exists("scan_trials")) # removed after the trial refinements
        page = open("scan_report.html").read()                # HTML report: groups of every N and the scan
        self.assertEqual(page.count('class="selrow"'), 1)
        self.assertIn("Recommendation: 2 groups per chain (2 TLS groups)", page)
        self.assertIn("R-values of the trial refinements", page)
        self.assertEqual(page.count('class="box"'), 1 + 2 + 3)
        self.assertLess(page.index('href="scan_tls.yaml"'), page.index('href="scan_tls_N1.yaml"'))
        if HAVE_MPL:
            self.assertTrue(os.path.isfile("scan_scan.png"))
        for n in (1, 2, 3):
            self.check_domain_files(f"scan_domains_N{n}.csv", None, f"scan_tls_N{n}.yaml", n)
        self.check_domain_files("scan_domains_N2.csv", "scan_domains.pdb", "scan_tls.yaml", 2)   # the recommendation

    def test_parallel_scan_equals_serial_scan(self):
        # trials are independent, so running them in parallel processes must not change any number
        self.run_select(["--max_groups", "2", "--tls_cycles", "3", "-d", "2.5", "--jobs", "1"], prefix="ser")
        self.run_select(["--max_groups", "2", "--tls_cycles", "3", "-d", "2.5", "--jobs", "3"], prefix="par")
        a, b = pandas.read_csv("ser_scan.csv"), pandas.read_csv("par_scan.csv")
        self.assertEqual(list(a.N), [0, 1, 2])
        self.assertEqual(list(b.N), [0, 1, 2]) # sorted by N although trials finish in any order
        for k in ("Rwork", "Rfree", "-LL"):
            self.assertTrue(numpy.allclose(a[k], b[k], rtol=0, atol=1e-9), k)
        self.assertIn("in parallel on 3 processes", open("par.log").read())
        self.assertIn("one after another", open("ser.log").read())

    def test_ngroups_with_evaluation(self):
        # --ngroups without --no_eval: the definition is compared with no TLS by two trial refinements
        self.run_select(["--ngroups", "2", "--tls_cycles", "2", "-d", "2.5"], prefix="ngev")
        self.assertEqual(len(self.read_ranges("ngev_tls.yaml")), 2)
        self.assertRegex(open("ngev.log").read(), r"2 groups\s+Rwork=")
        self.assertFalse(os.path.exists("ngev_trials"))
        page = open("ngev_report.html").read()
        self.assertIn("groups set by --ngroups 2", page)
        self.assertEqual(page.count('class="box"'), 2)

    def test_failing_trial_is_reported(self):
        args = argparse.ArgumentParser()
        tls_select.add_arguments(args)
        a = args.parse_args(["--model", "start.pdb", "--hklin", "data.mtz", "-s", "xray", "--labin", "FP,SIGFP,FreeR_flag",
                             "--no_solvent", "--hydrogen", "no", "-d", "2.5", "-o", "ft"])
        os.makedirs("ft_trials", exist_ok=True)
        specs = [("bad", "no_such_file.yaml", "ft_trials/bad"), ("ok", None, "ft_trials/ok")]
        with self.assertRaises(FileNotFoundError):
            tls_select.run_trials(a, specs, 1, 2)

class TestBFitMethod(unittest.TestCase):
    """util tls_bfit: TLS fitted to the isotropic B-factors of a model (in the spirit of TLSMD)"""
    @classmethod
    def setUpClass(cls):
        cls.wd = tempfile.mkdtemp(prefix="servaltest_")
        sys.path.insert(0, root)
        import make_tls_data
        cls.boundary = make_tls_data.make_bmodel(cls.wd)
        cls.model = os.path.join(cls.wd, "bmodel.pdb")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.wd)

    def chains(self):
        st = utils.fileio.read_structure(self.model)
        st.setup_entities()
        return tls_select.polymer_chains(st, 10)

    def test_residual_and_boundary(self):
        chains = self.chains()
        a = argparse.Namespace(atoms="all", min_residues=10, max_groups=4)
        parts = tls_select.partitions_by_bfactor(chains, a)["A"]
        c = chains[0]
        res = {k: -v[0] for k, v in parts.items()}
        rms = {k: numpy.sqrt(res[k] / a._bfit_atoms["A"]) for k in res}
        self.assertGreater(rms[1], 4.0)                 # one rigid group cannot explain two domains
        self.assertLess(rms[2], 0.7)                    # two can: down to the noise (0.5)
        self.assertLess(rms[3] / rms[2], 1.0)
        self.assertGreater(rms[3] / rms[2], 0.95)       # a third group gains almost nothing
        bounds = parts[2][1]
        self.assertEqual(c.residues[bounds[1]].seqid.num, self.boundary)   # the boundary between the domains

    def test_atom_sets_and_minimum_size(self):
        chains = self.chains()
        a = argparse.Namespace(atoms="ca", min_residues=10, max_groups=3)
        parts = tls_select.partitions_by_bfactor(chains, a)["A"]
        for k, (gain, bounds) in parts.items():
            sizes = numpy.diff(bounds)
            self.assertGreaterEqual(sizes.min(), 25)    # about 10 parameters from 1 atom per residue: larger groups

    def test_choose_n_elbow(self):
        c = tls_select.choose_n_elbow
        self.assertEqual(c([100., 10., 9.8, 9.7])[0], 2)
        self.assertEqual(c([100., 50., 25., 12.5, 12.4])[0], 4)
        self.assertEqual(c([100., 99.])[0], 1)
        self.assertEqual(c([100., 60., 30.])[0], 3)     # still falling at the end

    def run_bfit(self, extra, prefix):
        sys.argv = ["", "util", "tls_bfit", "--model", self.model] + extra + ["-o", os.path.join(self.wd, prefix)]
        main()

    def test_without_reflection_data(self):
        self.run_bfit(["--max_groups", "4"], "nodata")
        p = os.path.join(self.wd, "nodata")
        df = pandas.read_csv(p + "_scan.csv")
        self.assertEqual(list(df.N), [1, 2, 3, 4])
        self.assertTrue(numpy.all(numpy.diff(df.B_rms) < 0))
        with open(p + "_scan.json") as f:
            js = json.load(f)
        self.assertEqual(js["selected_N"], 2)
        import omegaconf
        cfg = omegaconf.OmegaConf.load(p + "_tls.yaml")
        sels = [g.selections[0] for g in cfg.refine.tls.groups]
        self.assertEqual(len(sels), 2)
        first = int(re.match(r"^//A/(\d+)-(\d+)/\*$", sels[1]).group(1))
        self.assertEqual(first, self.boundary)
        for n in (1, 2, 3, 4):
            self.assertTrue(os.path.isfile(f"{p}_tls_N{n}.yaml"))
            self.assertEqual(sorted(set(pandas.read_csv(f"{p}_domains_N{n}.csv").domain)), list(range(1, n + 1)))
        dom = pandas.read_csv(p + "_domains_N2.csv")
        self.assertEqual(set(dom[dom.resi.astype(int) < self.boundary].domain), {1})
        self.assertEqual(set(dom[dom.resi.astype(int) >= self.boundary].domain), {2})
        self.assertTrue(os.path.isfile(p + "_domains.pdb"))
        self.assertFalse(os.path.exists(p + "_trials"))                # no refinement, no trial directory
        page = open(p + "_report.html").read()
        self.assertIn("TLS groups from tls_bfit", page)
        self.assertIn("Fit of TLS to the B-factors", page)
        self.assertNotIn("R-values of the trial refinements", page)
        log = open(p + ".log").read()
        self.assertIn("no refinement", log)
        self.assertIn("rule of thumb", log)

    def test_ngroups_without_data(self):
        self.run_bfit(["--ngroups", "2"], "ng")
        import omegaconf
        cfg = omegaconf.OmegaConf.load(os.path.join(self.wd, "ng_tls.yaml"))
        self.assertEqual(len(cfg.refine.tls.groups), 2)

class TestSelectionsAroundHetero(unittest.TestCase):
    def test_generated_ranges_do_not_take_in_a_ligand_numbered_inside(self):
        sys.path.insert(0, root)
        import test_tls
        st = test_tls.polymer_with_hetero()
        st.setup_entities()
        chains = tls_select.polymer_chains(st, 5)
        self.assertEqual([c.name for c in chains], ["A"])
        c = chains[0]
        self.assertEqual(c.n, 19)
        self.assertEqual(c.selections(0, c.n), ["//A/1-10/*", "//A/12-20/*"])    # NAG 11 is skipped
        self.assertEqual(c.selections(0, 5), ["//A/1-5/*"])
        self.assertEqual(c.selections(10, 19), ["//A/12-20/*"])


class TestParallelHelpers(unittest.TestCase):
    def test_radiation_source_default(self):
        from servalcat.refine import refine_xtal
        from servalcat.xtal import sigmaa
        for mod, extra in ((refine_xtal, ["--model", "m.pdb", "--hklin", "d.mtz"]), (sigmaa, ["--model", "m.pdb", "--hklin", "d.mtz"])):
            p = argparse.ArgumentParser()
            mod.add_arguments(p)
            self.assertEqual(p.parse_args(extra).source, "xray")
            self.assertEqual(p.parse_args(extra + ["-s", "neutron"]).source, "neutron")
            self.assertEqual(p.parse_args(extra + ["--source", "electron"]).source, "electron")

    def test_trial_directory_is_removed(self):
        wd = tempfile.mkdtemp(prefix="servaltest_")
        try:
            d = os.path.join(wd, "x_trials")
            with tls_select.trial_directory(d):
                self.assertTrue(os.path.isdir(d))
                open(os.path.join(d, "n1.log"), "w").close()
            self.assertFalse(os.path.exists(d))
            with self.assertRaises(RuntimeError): # also after an error in a trial
                with tls_select.trial_directory(d):
                    raise RuntimeError
            self.assertFalse(os.path.exists(d))
            os.makedirs(d)                        # a folder that was there before is left alone
            with tls_select.trial_directory(d):
                pass
            self.assertTrue(os.path.isdir(d))
        finally:
            shutil.rmtree(wd)

    def test_available_cpus(self):
        n = tls_select.available_cpus()
        self.assertGreaterEqual(n, 1)
        self.assertLessEqual(n, os.cpu_count() or n)

    def test_single_threaded_libraries_restores_environment(self):
        os.environ["OMP_NUM_THREADS"] = "7"
        os.environ.pop("MKL_NUM_THREADS", None)
        with tls_select.single_threaded_libraries():
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")
            self.assertEqual(os.environ["MKL_NUM_THREADS"], "1")
        self.assertEqual(os.environ["OMP_NUM_THREADS"], "7")
        self.assertNotIn("MKL_NUM_THREADS", os.environ)
        del os.environ["OMP_NUM_THREADS"]

    def test_help_layout_and_defaults(self):
        from servalcat.refine import refine_xtal, refine_geom
        from servalcat.xtal import sigmaa
        for mod in (refine_xtal, tls_select, sigmaa, refine_geom):
            p = argparse.ArgumentParser(prog="servalcat test")
            mod.add_arguments(p)
            text = p.format_help()
            self.assertIn("Required arguments:", text, mod.__name__)
            self.assertIn("Optional arguments with default values:", text, mod.__name__)
            self.assertIn("(off unless given):", text, mod.__name__)
            required = {a.dest for g in p._action_groups if g.title == "Required arguments" for a in g._group_actions}
            required |= {a.dest for g in p._mutually_exclusive_groups for a in g._group_actions}
            self.assertIn("model", required)
            for act in p._actions:
                if act.dest == "help" or act.help == argparse.SUPPRESS or act.dest in required:
                    continue
                # every other shown option states its default
                self.assertIn("(default:", act.help, "{} {}".format(mod.__name__, act.dest))
            self.assertIn("(default: xray)" if mod is not refine_geom else "(default: all)", text)
        # defaults named in the text are the real ones
        p = argparse.ArgumentParser(prog="servalcat test")
        refine_xtal.add_arguments(p)
        self.assertIn("(default: 10)", p.format_help()) # --ncycle
        self.assertEqual(p.parse_args(["--hklin", "a", "--model", "b"]).ncycle, 10)
        # tls_select does not list options that it sets itself
        p = argparse.ArgumentParser(prog="servalcat test")
        tls_select.add_arguments(p)
        shown = p.format_help()
        for o in ("--ncycle", "--fix_xyz", "--tls ", "--tls_file", "--adp "):
            self.assertNotIn("  " + o, shown)
        self.assertIn("--max_groups", shown)
        self.assertIn("--seq ", shown)
        # tls_bfit: the reflection data are optional
        p = argparse.ArgumentParser(prog="servalcat test")
        tls_select.add_arguments(p, method="bfactor")
        text = p.format_help()
        self.assertIn("Required arguments:", text)
        self.assertIn("--atoms", text)
        self.assertEqual(p.parse_args(["--model", "m.pdb"]).hklin, None)
        for act in p._actions:
            if act.dest not in ("help", "model") and act.help != argparse.SUPPRESS and not act.required:
                self.assertIn("(default:", act.help, act.dest)
        # refine_geom: one of --model / --update_dictionary
        p = argparse.ArgumentParser(prog="servalcat test")
        refine_geom.add_arguments(p)
        with self.assertRaises(SystemExit):
            p.parse_args([])
        self.assertEqual(p.parse_args(["--update_dictionary", "x.cif"]).update_dictionary, "x.cif")

    def test_n_jobs(self):
        ns = argparse.Namespace
        self.assertEqual(tls_select.n_jobs(ns(jobs=4), 9), 4)
        self.assertEqual(tls_select.n_jobs(ns(jobs=4), 2), 2)   # never more processes than trials
        self.assertEqual(tls_select.n_jobs(ns(jobs=0), 100), tls_select.available_cpus())
        self.assertEqual(tls_select.n_jobs(ns(jobs=0), 1), 1)

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
        df = pandas.read_csv("sel_domains.csv")                       # groups are numbered chain by chain: A: 1, B: 2 and 3
        self.assertEqual(sorted(set(df[df.chain == "A"].domain)), [1])
        self.assertEqual(sorted(set(df[df.chain == "B"].domain)), [2, 3])

if __name__ == '__main__':
    unittest.main()
