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
try:
    import matplotlib
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

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
        args = types.SimpleNamespace(max_groups=k, min_fragment=5, min_residues=20, min_stretch=3, **kw)
        networks = nma.build_networks(nodes, enm, nma.ENM_CUTOFF[enm], 10)
        return nma.partitions(nodes, by_chain, args, networks, enm), truth

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
        # with span_chains a domain may consist of several chains: chain B of 40 residues next to the first half of chain A
        st, truth = md.build([(0, 40), (1, 40)], seed=0, separation=34.0)
        ch = st[0]["A"]
        chB = gemmi.Chain("B")
        for i in range(40, 80):
            chB.add_residue(ch[i])
        for i in reversed(range(40, 80)):
            del ch[i]
        st[0].add_chain(chB)
        nodes, by_chain = nma.collect_nodes(st)
        args = types.SimpleNamespace(max_groups=2, min_fragment=5, min_residues=20, min_stretch=3, span_chains=True)
        networks = nma.build_networks(nodes, "anm", 15.0, 10)
        parts = nma.partitions(nodes, by_chain, args, networks, "anm")
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
            if HAVE_MPL:
                self.assertTrue(os.path.isfile(p + "_scan.png")) # B_rms vs N (no reflection data)
            self.assertFalse(os.path.exists(p + "_trials"))

    def test_domains_pdb_and_ngroups(self):
        self.run_cmd(["--ngroups", "2"], "ng")
        self.assertEqual(len(self.groups("ng_tls.yaml")), 2)
        st = gemmi.read_structure("ng_domains.pdb")
        vals = {round(a.atom.b_iso) for a in st[0].all()}
        self.assertEqual(vals, {1, 2})
        # more domains than the chain can have (120 residues, at least 20 per domain): clipped, with a warning
        sys.argv = ["", "util", "tls_nma", "--model", "pseudo.pdb", "--ngroups", "9", "-o", "many"]
        main()
        self.assertRegex(open("many.log").read(), r"WARNING: chain A: 9 domains requested, \d possible")
        self.assertLessEqual(len(self.groups("many_tls.yaml")), 6)

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
            self.assertFalse(os.path.exists("ev_trials"))           # removed after the trial refinements
            if HAVE_MPL:
                self.assertTrue(os.path.isfile("ev_scan.png"))      # R vs N
        finally:
            os.chdir(root)
            shutil.rmtree(wd)


class TestAssemblies(unittest.TestCase):
    """homo- and hetero-oligomers: the chains are divided on their own (default), the network is that of the assembly"""
    LAYOUT = [(0, 40), (1, 40), (0, 40)]

    def ring(self, copies=3):
        return md.oligomer(self.LAYOUT, copies=copies)

    def run_partitions(self, st, enm="anm", network="complex", k=2, **kw):
        nodes, by_chain = nma.collect_nodes(st)
        args = types.SimpleNamespace(max_groups=k, min_fragment=5, min_residues=20, min_stretch=3, **kw)
        networks = nma.build_networks(nodes, enm, nma.ENM_CUTOFF[enm], 10, network, args.min_residues)
        return nodes, nma.partitions(nodes, by_chain, args, networks, enm), networks

    def chains_in(self, group):
        return {s.split("/")[2] for s in group}

    def test_the_network_of_the_assembly_has_springs_between_the_chains(self):
        st, truth = self.ring()
        nodes, parts, networks = self.run_partitions(st)
        self.assertEqual(len(networks), 1)
        chains = [nodes[g].chain for g in networks[0].members]
        between = sum(1 for a, b in networks[0].edges if chains[a] != chains[b])
        self.assertGreater(between, 20)
        self.assertEqual(nma.n_components(len(nodes), networks[0].edges), 1)       # the ring is one connected assembly
        nodes, parts, networks = self.run_partitions(st, network="chain")
        self.assertEqual(len(networks), 3)                                          # chain by chain

    def test_homo_oligomer_is_divided_chain_by_chain(self):
        for enm in ("anm", "gnm"):
            for scope in ("complex", "chain"):
                st, truth = self.ring()
                nodes, parts, _ = self.run_partitions(st, enm, scope, k=2)
                labels, groups = parts[1]                       # one domain per chain: three groups, never one for all
                self.assertEqual(len(groups), 3, (enm, scope))
                self.assertEqual([self.chains_in(g) for g in groups], [{"A"}, {"B"}, {"C"}])
                labels, groups = parts[2]
                self.assertEqual(len(groups), 6, (enm, scope))
                for g in groups:
                    self.assertEqual(len(self.chains_in(g)), 1, "a group must not contain two chains")
                for c in "ABC":
                    sel = sorted(sorted(g) for g in groups if self.chains_in(g) == {c})
                    self.assertEqual(sel, [[f"//{c}/1-40/*", f"//{c}/81-120/*"], [f"//{c}/41-80/*"]], (enm, scope))

    def test_span_chains_allows_domains_of_several_chains(self):
        st, truth = self.ring()
        nodes, parts, _ = self.run_partitions(st, span_chains=True)
        labels, groups = parts[1]
        self.assertEqual(len(groups), 1)                         # the old behaviour: one group for the whole assembly
        self.assertEqual(self.chains_in(groups[0]), {"A", "B", "C"})

    def test_hetero_oligomer_with_numbers_of_domains_per_chain(self):
        # chain A: three stretches of two domains; chain B: one compact domain of 60 residues next to it; peptide P: 10 residues
        stA, truthA = md.build([(0, 40), (1, 40), (0, 40)], seed=0, separation=34.0)
        stB, _ = md.build([(0, 60)], seed=3)
        chB = stB[0]["A"].clone()
        chB.name = "B"
        for r in chB:
            for a in r:
                a.pos = gemmi.Position(a.pos.x + 30, a.pos.y + 20, a.pos.z)
        stP, _ = md.build([(0, 10)], seed=5)
        chP = stP[0]["A"].clone()
        chP.name = "P"
        for r in chP:
            for a in r:
                a.pos = gemmi.Position(a.pos.x - 25, a.pos.y, a.pos.z + 40)
        stA[0].add_chain(chB)
        stA[0].add_chain(chP)
        nodes, by_chain = nma.collect_nodes(stA)
        args = types.SimpleNamespace(max_groups=3, min_fragment=5, min_residues=20, min_stretch=3)
        networks = nma.build_networks(nodes, "anm", 15.0, 10)
        finder = nma.DomainFinder(nodes, by_chain, args, networks, "anm")
        self.assertEqual(finder.chain_names(), ["A", "B"])        # the 10-residue peptide is not in a group
        counts = {"A": 2, "B": 1}
        labels = finder.labels_for(lambda u: counts[u.chain])
        labels, groups = finder.result(labels)
        self.assertEqual(sorted(sorted(g) for g in groups),
                         [["//A/1-40/*", "//A/81-120/*"], ["//A/41-80/*"], ["//B/1-60/*"]])
        self.assertTrue((labels[-10:] == -1).all())               # the peptide: no group

    def test_command_on_an_assembly_writes_groups_per_chain(self):
        wd = tempfile.mkdtemp(prefix="servaltest_")
        os.chdir(wd)
        try:
            st, truth = self.ring()
            utils.fileio.write_model(st, "ring", pdb=True)
            sys.argv = ["", "util", "tls_nma", "--model", "ring.pdb", "--max_groups", "2", "-o", "r"]
            main()
            import omegaconf
            groups = [list(g.selections) for g in omegaconf.OmegaConf.load("r_tls_N1.yaml").refine.tls.groups]
            self.assertEqual(len(groups), 3)
            dom = pandas.read_csv("r_domains_N2.csv")
            self.assertEqual(sorted(set(dom.domain)), [1, 2, 3, 4, 5, 6])
            for d, rows in dom.groupby("domain"):
                self.assertEqual(len(set(rows.chain)), 1, f"domain {d} spans chains")
            log = open("r.log").read()
            self.assertIn("whole assembly (3 chains)", log)
            self.assertIn("between chains", log)
            self.assertIn("never contains residues of two chains", log)
            # per-chain numbers of domains
            sys.argv = ["", "util", "tls_nma", "--model", "ring.pdb", "--ngroups", "A:2", "B:1", "-o", "pc"]
            main()
            groups = [list(g.selections) for g in omegaconf.OmegaConf.load("pc_tls.yaml").refine.tls.groups]
            self.assertEqual(len(groups), 2 + 1 + 1)                # A: 2, B: 1, C: 1 (chains not given: 1)
            # --span_chains needs the network of the assembly
            sys.argv = ["", "util", "tls_nma", "--model", "ring.pdb", "--network", "chain", "--span_chains", "-o", "bad"]
            with self.assertRaises(SystemExit):
                main()
        finally:
            os.chdir(root)
            shutil.rmtree(wd)

if __name__ == '__main__':
    unittest.main()
