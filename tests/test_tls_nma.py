"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import argparse
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
import numpy
import pandas
import servalcat # must be imported before gemmi (registers the bundled gemmi)
import gemmi
from servalcat import utils
from servalcat.refine import tls_nma as nma
from servalcat.__main__ import main

root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, root)
import make_nma_data as md

def lysozyme_nodes():
    st = utils.fileio.read_structure(os.path.join(root, "1l2h", "1l2h.cif.gz"))
    st.setup_entities()
    nodes, by_chain = nma.collect_nodes(st)
    return nodes, numpy.array([n.pos for n in nodes])

def agreement(labels, truth):
    """fraction of residues in the right one of two domains (label numbers are arbitrary)"""
    return max(numpy.mean((labels == 0) == (truth == 0)), numpy.mean((labels == 0) == (truth == 1)))

class TestNetworkModels(unittest.TestCase):
    def test_nodes(self):
        nodes, pos = lysozyme_nodes()
        self.assertEqual(len(nodes), 144)                 # polymer residues only: no waters, no ligands
        self.assertTrue(all(n.residue.name != "HOH" for n in nodes))

    def test_matrices_have_the_zero_modes_of_rigid_motions(self):
        nodes, pos = lysozyme_nodes()
        e = nma.contact_edges(pos, 15.0)
        H = nma.anm_hessian(pos, e).toarray()
        self.assertLess(numpy.abs(H - H.T).max(), 1e-12)
        t = numpy.zeros((3 * len(pos), 3))                # translations
        for k in range(3):
            t[k::3, k] = 1.
        self.assertLess(numpy.abs(H @ t).max(), 1e-9)
        rot = numpy.zeros(3 * len(pos))                   # rotation about z
        rot[0::3], rot[1::3] = -pos[:, 1], pos[:, 0]
        self.assertLess(numpy.abs(H @ rot).max(), 1e-8)
        w = numpy.linalg.eigvalsh(H)
        self.assertEqual(int((w < 1e-8).sum()), 6)        # exactly the six rigid-body modes
        self.assertGreater(w[6], 1e-3)
        K = nma.gnm_kirchhoff(len(pos), nma.contact_edges(pos, 10.0)).toarray()
        self.assertLess(numpy.abs(K.sum(1)).max(), 1e-12)
        self.assertEqual(int((numpy.linalg.eigvalsh(K) < 1e-8).sum()), 1)

    def test_sparse_solver_equals_dense(self):
        nodes, pos = lysozyme_nodes()
        big = numpy.vstack([pos + numpy.array([0, 0, 45.0 * k]) for k in range(3)])
        for enm, cut in (("anm", 15.0), ("gnm", 10.0)):
            e = nma.contact_edges(big, cut)
            M = nma.anm_hessian(big, e) if enm == "anm" else nma.gnm_kirchhoff(len(big), e)
            nz = (6 if enm == "anm" else 1) * nma.n_components(len(big), e)
            wd, _ = nma.lowest_modes(M, 6, nz)
            old = nma.DENSE_LIMIT
            nma.DENSE_LIMIT = 10
            try:
                ws, _ = nma.lowest_modes(M, 6, nz)
            finally:
                nma.DENSE_LIMIT = old
            self.assertLess(numpy.abs(wd - ws).max(), 1e-9, enm)

    def test_eigenvalues_agree_with_prody(self):
        prody = __import__("pytest").importorskip("prody")
        prody.confProDy(verbosity="none")
        nodes, pos = lysozyme_nodes()
        for enm, cls, cut in (("anm", prody.ANM, 15.0), ("gnm", prody.GNM, 10.0)):
            edges, w, V = nma.network_modes(pos, enm, cut, 10)
            m = cls("t")
            (m.buildHessian if enm == "anm" else m.buildKirchhoff)(pos, cutoff=cut, gamma=1.0)
            m.calcModes(n_modes=10)
            self.assertLess(numpy.abs(w - m.getEigvals()).max(), 1e-8, enm)
            overlap = numpy.linalg.norm(V[:, :5].T @ m.getEigvecs()[:, :5]) ** 2 / 5
            self.assertAlmostEqual(overlap, 1.0, places=6, msg=enm)

class TestDomains(unittest.TestCase):
    def find(self, enm, layout, seed=0, separation=32.0, k=2, **kw):
        st, truth = md.build(layout, seed=seed, separation=separation)
        nodes, by_chain = nma.collect_nodes(st)
        pos = numpy.array([n.pos for n in nodes])
        args = types.SimpleNamespace(max_groups=k, min_fragment=5, min_residues=20, min_stretch=3, **kw)
        edges, w, V = nma.network_modes(pos, enm, nma.ENM_CUTOFF[enm], 10)
        return nma.partitions(nodes, by_chain, args, edges, enm, w, V), truth

    def test_internal_domain_is_found_with_both_models(self):
        # residues 41-80 form an internal domain, the flanks 1-40 and 81-120 the other one
        for enm in ("anm", "gnm"):
            parts, truth = self.find(enm, [(0, 40), (1, 40), (0, 40)])
            labels, groups = parts[2]
            self.assertEqual(agreement(labels, truth), 1.0, enm)
            self.assertEqual(sorted(map(sorted, groups)), [["//A/1-40/*", "//A/81-120/*"], ["//A/41-80/*"]], enm)

    def test_robust_to_the_contact_between_the_domains(self):
        for enm in ("anm", "gnm"):
            for seed in range(3):
                parts, truth = self.find(enm, [(0, 40), (1, 40), (0, 40)], seed=seed, separation=28.0)
                self.assertGreaterEqual(agreement(parts[2][0], truth), 0.95, f"{enm} seed {seed}")

    def test_three_domains(self):
        for enm in ("anm", "gnm"):
            parts, truth = self.find(enm, [(0, 40), (1, 40), (2, 40)], k=3)
            labels = parts[3][0]
            self.assertEqual(sorted(numpy.bincount(labels).tolist()), [40, 40, 40], enm)
            for d in range(3):
                self.assertEqual(len(set(labels[truth == d])), 1, enm)   # every true domain is one group

    def test_two_chains_in_one_domain(self):
        # a domain of two chains: chain B of 40 residues sits next to the first half of chain A
        st, truth = md.build([(0, 40), (1, 40)], seed=0, separation=34.0)
        ch = st[0]["A"]
        # move the last 40 residues to a second chain: the second domain is formed by chain B only, domain 0 by chain A
        chB = gemmi.Chain("B")
        for i in range(40, 80):
            chB.add_residue(ch[i])
        for i in reversed(range(40, 80)):
            del ch[i]
        st[0].add_chain(chB)
        nodes, by_chain = nma.collect_nodes(st)
        pos = numpy.array([n.pos for n in nodes])
        args = types.SimpleNamespace(max_groups=2, min_fragment=5, min_residues=20, min_stretch=3)
        edges, w, V = nma.network_modes(pos, "anm", 15.0, 10)
        parts = nma.partitions(nodes, by_chain, args, edges, "anm", w, V)
        self.assertEqual(sorted(map(sorted, parts[2][1])), [["//A/1-40/*"], ["//B/41-80/*"]])

    def test_cleaning(self):
        # a chain of 60 nodes on a line; labels with a 2-residue island and a 3-residue island
        pos = numpy.array([[i * 3.8, 0, 0] for i in range(60)])
        edges = nma.contact_edges(pos, 8.0)
        A = nma.graph_affinity(60, edges, numpy.ones(len(edges)))
        labels = numpy.array([0] * 30 + [1] * 30)
        labels[10:12] = 1                       # island inside group 0
        labels[45:48] = 0                       # island inside group 1
        out = nma.clean_labels(labels, A, 5, 20)
        self.assertEqual(out.tolist(), [0] * 30 + [1] * 30)
        nodes = [nma.Node("A", None, pos[i], i) for i in range(60)]
        labels = numpy.array([0] * 30 + [1] * 30)
        labels[10:12] = 1
        out = nma.smooth_stretches(labels, nodes, A, 3)
        self.assertEqual(out.tolist(), [0] * 30 + [1] * 30)
        # a group below the minimum size is merged
        labels = numpy.array([0] * 55 + [1] * 5)
        self.assertEqual(len(set(nma.clean_labels(labels, A, 2, 20))), 1)

    def test_spectral_clustering_of_two_blocks(self):
        pos = numpy.vstack([numpy.random.default_rng(0).normal(size=(30, 3)) * 3, numpy.random.default_rng(1).normal(size=(30, 3)) * 3 + [40, 0, 0]])
        edges = nma.contact_edges(pos, 25.0)
        A = nma.graph_affinity(60, edges, numpy.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1))
        lab = nma.spectral_labels(A, 2)
        self.assertEqual(len(set(lab[:30])), 1)
        self.assertEqual(len(set(lab[30:])), 1)
        self.assertNotEqual(lab[0], lab[-1])

class TestCommand(unittest.TestCase):
    def setUp(self):
        self.wd = tempfile.mkdtemp(prefix="servaltest_")
        os.chdir(self.wd)
        st, self.truth = md.build([(0, 40), (1, 40), (0, 40)], seed=0, separation=32.0)
        utils.fileio.write_model(st, "pseudo", pdb=True)

    def tearDown(self):
        os.chdir(root)
        shutil.rmtree(self.wd)

    def run_cmd(self, extra, prefix):
        sys.argv = ["", "util", "tls_nma", "--model", "pseudo.pdb"] + extra + ["-o", prefix]
        main()

    def groups(self, path):
        import omegaconf
        return [list(g.selections) for g in omegaconf.OmegaConf.load(path).refine.tls.groups]

    def test_command_writes_discontiguous_groups(self):
        for enm in ("anm", "gnm"):
            self.run_cmd(["--enm", enm, "--max_groups", "3"], f"o_{enm}")
            p = f"o_{enm}"
            df = pandas.read_csv(p + "_scan.csv")
            self.assertEqual(list(df.N), [1, 2, 3])
            with open(p + "_scan.json") as f:
                js = json.load(f)
            self.assertEqual(js["enm"], enm)
            n_sel = js["selected_N"]
            self.assertIn(n_sel, (1, 2, 3))
            g2 = self.groups(p + "_tls_N2.yaml")
            self.assertEqual(sorted(map(sorted, g2)), [["//A/1-40/*", "//A/81-120/*"], ["//A/41-80/*"]])
            dom = pandas.read_csv(p + "_domains_N2.csv")
            self.assertEqual(len(dom), 120)
            self.assertEqual(set(dom.domain), {1, 2})
            self.assertEqual(set(dom[dom.resi.between(41, 80)].domain.tolist()), {int(dom.domain[40])})
            self.assertTrue(os.path.isfile(p + "_scan.json") and os.path.isfile(p + "_tls.yaml"))

    def test_domains_pdb_and_ngroups(self):
        self.run_cmd(["--ngroups", "2"], "ng")
        self.assertEqual(len(self.groups("ng_tls.yaml")), 2)
        st = gemmi.read_structure("ng_domains.pdb")
        vals = {round(a.atom.b_iso) for a in st[0].all()}
        self.assertEqual(vals, {1, 2})
        sys.argv = ["", "util", "tls_nma", "--model", "pseudo.pdb", "--ngroups", "9", "-o", "bad"]
        with self.assertRaises(SystemExit):
            main()

    def test_help_layout(self):
        p = argparse.ArgumentParser(prog="servalcat test")
        nma.add_arguments(p)
        text = p.format_help()
        for s in ("--enm {anm,gnm}", "--cutoff", "--nmodes", "--min_stretch", "Required arguments:", "(off unless given):"):
            self.assertIn(s, text)
        self.assertEqual(p.parse_args(["--model", "m.pdb"]).hklin, None)
        self.assertEqual(p.parse_args(["--model", "m.pdb"]).enm, "anm")
        for act in p._actions:
            if act.dest not in ("help", "model") and act.help != argparse.SUPPRESS:
                self.assertIn("(default:", act.help, act.dest)

class TestWithReflections(unittest.TestCase):
    def test_r_values_per_number_of_domains(self):
        wd = tempfile.mkdtemp(prefix="servaltest_")
        os.chdir(wd)
        try:
            import make_tls_data
            make_tls_data.make(".")
            sys.argv = ["", "util", "tls_nma", "--model", "start.pdb", "--hklin", "data.mtz", "-s", "xray",
                        "--labin", "FP,SIGFP,FreeR_flag", "--no_solvent", "--hydrogen", "no", "--max_groups", "2",
                        "--tls_cycles", "3", "-d", "2.5", "-o", "ev"]
            main()
            df = pandas.read_csv("ev_scan.csv")
            self.assertEqual(list(df.N), [0, 1, 2])
            for k in ("Rwork", "Rfree", "B_rms"):
                self.assertIn(k, df.columns)
            self.assertGreater(df.Rfree[0] - df.Rfree[1], 0.005)    # TLS helps
        finally:
            os.chdir(root)
            shutil.rmtree(wd)

if __name__ == '__main__':
    unittest.main()
