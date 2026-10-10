"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import unittest
import json
import os
import shutil
import sys
import tempfile
import numpy
import servalcat # must be imported before gemmi (registers the bundled gemmi)
import gemmi
from servalcat import utils
from servalcat.refine import tls
from servalcat.__main__ import main

root = os.path.abspath(os.path.dirname(__file__))

class TestTlsMath(unittest.TestCase):
    def setUp(self):
        rng = numpy.random.default_rng(1)
        self.p21 = rng.normal(size=21) * 0.3
        self.origin = numpy.array([1., 2., 3.])
        self.pos = rng.normal(size=(6, 3)) * 8

    def test_forward_model_matches_gemmi(self):
        g = gemmi.TlsGroup()
        g.origin = gemmi.Position(*self.origin)
        g.T = gemmi.SMat33d(*self.p21[0:6])
        g.L = gemmi.SMat33d(*self.p21[6:12])
        g.S = gemmi.Mat33(self.p21[12:21].reshape(3, 3).tolist())
        ref = numpy.array([gemmi.calculate_u_from_tls(g, gemmi.Position(*x)).elements_pdb() for x in self.pos])
        mine = tls.tls_jacobian(self.pos - self.origin) @ self.p21
        numpy.testing.assert_allclose(mine, ref, atol=1e-12)

    def test_s_trace_does_not_change_u(self):
        # adding c*I to S has no effect on U_tls (this is why S has 8 free parameters)
        J = tls.tls_jacobian(self.pos - self.origin)
        p2 = self.p21.copy()
        p2[[12, 16, 20]] += 0.7
        numpy.testing.assert_allclose(J @ p2, J @ self.p21, atol=1e-12)

    def test_parameter_roundtrip(self):
        T, L, S = tls.tls_matrices(tls.p_from_matrices(tls.sym33(self.p21[0:6]), tls.sym33(self.p21[6:12]),
                                                       self.p21[12:21].reshape(3, 3)))
        numpy.testing.assert_allclose(T, tls.sym33(self.p21[0:6]))
        numpy.testing.assert_allclose(L, tls.sym33(self.p21[6:12]))
        numpy.testing.assert_allclose(S, self.p21[12:21].reshape(3, 3) - numpy.trace(self.p21[12:21].reshape(3, 3)) / 3 * numpy.eye(3),
                                      atol=1e-12)
        self.assertAlmostEqual(numpy.trace(S), 0.)

    def test_psd_projection(self):
        p = numpy.zeros(tls.NPAR)
        p[0:6] = [1., -0.5, 2., 0., 0., 0.] # T has a negative eigenvalue
        p[6:12] = [1., 1., 1., 0.1, 0., 0.]  # L is positive definite
        q = tls.project_psd(p)
        self.assertGreaterEqual(numpy.linalg.eigvalsh(tls.sym33(q[0:6])).min(), -1e-12)
        numpy.testing.assert_allclose(q[6:12], p[6:12])

class TestTlsRefine(unittest.TestCase):
    def setUp(self):
        self.wd = tempfile.mkdtemp(prefix="servaltest_")
        os.chdir(self.wd)
        print("In", self.wd)

    def tearDown(self):
        os.chdir(root)
        shutil.rmtree(self.wd)

    def test_groups_from_model(self):
        st = utils.fileio.read_structure(os.path.join(root, "5e5z", "5e5z.pdb.gz"))
        groups = tls.groups_from_model(st)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0].atoms), st[0].count_atom_sites()) # "SELECTION: ALL"
        T, L, S = groups[0].tensors()
        self.assertAlmostEqual(numpy.trace(S), 0.) # the trace of S is not a parameter

    def test_refine_tls_from_model(self):
        mtzin = os.path.join(root, "5e5z", "5e5z.mtz.gz")
        pdbin = os.path.join(root, "5e5z", "5e5z.pdb.gz")
        sys.argv = ["", "refine_xtal", "--model", pdbin, "--hklin", mtzin, "-s", "xray",
                    "--labin", "FP,SIGFP,FREE", "--tls", "--ncycle", "4", "--hydrogen", "no"]
        main()
        with open("5e5z_refined_stats.json") as f:
            stats = json.load(f)
        self.assertEqual(len(stats[-1]["tls"]), 1)
        self.assertLess(stats[-1]["data"]["summary"]["-LL"], stats[0]["data"]["summary"]["-LL"])
        # output: total U in ANISOU, TLS groups in the metadata
        for fname in ("5e5z_refined.mmcif", "5e5z_refined.pdb"):
            st = utils.fileio.read_structure(fname)
            self.assertTrue(all(x.atom.aniso.nonzero() for x in st[0].all()))
            self.assertEqual(len(tls.groups_from_model(st)), 1) # can be read back (metadata or REMARK 3)
        st = utils.fileio.read_structure("5e5z_refined.mmcif")
        self.assertEqual(len(st.meta.refinement[0].tls_groups), 1)
        # B in the output is the isotropic equivalent of the total U
        for cra in st[0].all():
            self.assertAlmostEqual(cra.atom.b_iso, cra.atom.aniso.trace() / 3 * utils.model.u_to_b, places=1)

    def test_tls_errors(self):
        hklin = os.path.join(root, "biotin", "biotin_talos.mtz")
        xyzin = os.path.join(root, "biotin", "biotin_talos.pdb")
        base = ["", "refine_xtal", "--model", xyzin, "--hklin", hklin, "-s", "xray", "--unrestrained",
                "--hydrogen", "no", "--ncycle", "1"]
        sys.argv = base + ["--tls"] # no groups defined anywhere
        with self.assertRaises(SystemExit):
            main()
        sys.argv = base + ["--tls_per_chain", "--adp", "aniso"] # residual B must be isotropic
        with self.assertRaises(SystemExit):
            main()

    def test_tls_gradient(self):
        # the gradient of -LL with respect to T, L, S (projected from the per-atom gradient)
        # must agree with finite differences of -LL. (A system with a realistic cell is needed:
        # the tiny cell of 5e5z is smaller than the extent of the atomic densities.)
        mtzin = os.path.join(root, "1l2h", "1l2h.mtz.gz")
        cifin = os.path.join(root, "1l2h", "1l2h.cif.gz")
        with open("tls.yaml", "w") as f: # starts from T=L=S=0
            f.write("refine:\n  tls:\n    groups:\n"
                    "      - id: 1\n        selections: ['//A/4-80/*']\n"
                    "      - id: 2\n        selections: ['//A/81-164/*']\n")
        result = {}
        def check(self_, ll, ncycle=None):
            if result:
                return []
            ll.update_fc()
            grad, hess = self_.gradient(ll)[0]
            g = self_.groups[0]
            for k, h in ((0, 0.008), (1, 0.008), (2, 0.008), (6, 0.15), (7, 0.15), (8, 0.15)): # T11-T33, L11-L33
                p0 = g.p.copy()
                fs = []
                for sgn in (1, -1):
                    g.p = p0.copy()
                    g.p[k] += sgn * h
                    self_.apply()
                    ll.update_fc()
                    fs.append(ll.calc_target())
                g.p = p0
                result[k] = (grad[k], (fs[0] - fs[1]) / (2 * h))
            self_.apply()
            ll.update_fc()
            return []
        orig = tls.TlsModel.refine
        tls.TlsModel.refine = check
        try:
            sys.argv = ["", "refine_xtal", "--model", cifin, "--hklin", mtzin, "-s", "xray", "--no_solvent",
                        "--labin", "IMEAN,SIGIMEAN,FreeR_flag", "--config", "tls.yaml", "--ncycle", "1",
                        "--hydrogen", "no"]
            main()
        finally:
            tls.TlsModel.refine = orig
        self.assertEqual(len(result), 6)
        for k, (ana, fd) in result.items():
            self.assertAlmostEqual(ana / fd, 1., delta=0.05, msg=f"parameter {k}: analytic {ana} FD {fd}")

    def test_tls_synthetic(self):
        # noise-free amplitudes calculated from a model with two TLS groups; refinement starting without
        # any anisotropy must recover the anisotropic part of the displacement
        sys.path.insert(0, root)
        import make_tls_data
        d = make_tls_data.make(".")
        with open("tls.yaml", "w") as f:
            f.write("refine:\n  tls:\n    groups:\n"
                    "      - id: 1\n        selections: ['//A/4-80/*']\n"
                    "      - id: 2\n        selections: ['//A/81-164/*']\n")
        sys.argv = ["", "refine_xtal", "--model", "start.pdb", "--hklin", "data.mtz", "-s", "xray",
                    "--labin", "FP,SIGFP,FreeR_flag", "--no_solvent", "--hydrogen", "no", "--tls",
                    "--config", "tls.yaml", "--ncycle", "10", "-o", "out"]
        main()
        with open("out_stats.json") as f:
            stats = json.load(f)
        self.assertLess(stats[-1]["data"]["summary"]["Rwork"], 0.04) # isotropic B alone: 0.058
        truth = numpy.load("truth_u.npy")
        st = utils.fileio.read_structure("out.pdb")
        out = numpy.array([cra.atom.aniso.elements_pdb() for cra in st[0].all()])
        sel = numpy.array([numpy.any(t[3:] != 0) for t in truth])
        out_a, tru_a = out.copy(), truth.copy()
        out_a[:, :3] -= out[:, :3].mean(axis=1)[:, None]
        tru_a[:, :3] -= truth[:, :3].mean(axis=1)[:, None]
        for k in range(6):
            cc = numpy.corrcoef(out_a[sel, k], tru_a[sel, k])[0, 1]
            self.assertGreater(cc, 0.9, f"component {k}")

if __name__ == '__main__':
    unittest.main()
