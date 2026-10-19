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
from servalcat import utils, ext
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

    def test_origin_shift_is_exact(self):
        # moving the origin transforms T, L, S such that U_tls at every position is unchanged
        pts = numpy.random.default_rng(5).normal(size=(80, 3)) * 10
        old, new = numpy.array([1., 2., 3.]), numpy.array([6., -2., 9.])
        M = tls.origin_shift_matrix(old, new, pts)
        p = numpy.random.default_rng(6).normal(size=tls.NPAR)
        u_old = tls.tls_jacobian(pts - old) @ tls._P @ p
        u_new = tls.tls_jacobian(pts - new) @ tls._P @ (M @ p)
        numpy.testing.assert_allclose(u_new, u_old, atol=1e-10)

    def test_psd_projection(self):
        p = numpy.zeros(tls.NPAR)
        p[0:6] = [1., -0.5, 2., 0., 0., 0.] # T has a negative eigenvalue
        p[6:12] = [1., 1., 1., 0.1, 0., 0.]  # L is positive definite
        q = tls.project_psd(p)
        self.assertGreaterEqual(numpy.linalg.eigvalsh(tls.sym33(q[0:6])).min(), -1e-12)
        numpy.testing.assert_allclose(q[6:12], p[6:12])

def polymer_with_hetero():
    """chain A: residues 1-10 and 12-20 (a loop is missing at 11), a glycan NAG numbered 11 INSIDE the numbering of the
    polymer, waters 101-103 and a ligand; plus a chain W that only has waters"""
    sys.path.insert(0, root)
    import make_seq_data as msd
    st = msd.build([msd.chain("A", "ACDEFGHIKLMNPQRSTVWY", "AA", missing=[11])])
    ch = st[0]["A"]
    def add(name, num, atoms, entity, het="H"):
        r = gemmi.Residue()
        r.name, r.seqid, r.het_flag, r.entity_type = name, gemmi.SeqId(num, " "), het, entity
        for i, an in enumerate(atoms):
            a = gemmi.Atom()
            a.name, a.element, a.occ, a.b_iso = an, gemmi.Element(an[0]), 1.0, 30.0
            a.pos = gemmi.Position(3.8 * num + i, 5.0, 5.0)
            r.add_atom(a)
        ch.add_residue(r)
    add("NAG", 11, ["C1", "C2", "O5"], gemmi.EntityType.NonPolymer)
    for n in (101, 102, 103):
        add("HOH", n, ["O"], gemmi.EntityType.Water)
    add("GOL", 201, ["C1", "O1"], gemmi.EntityType.NonPolymer)
    w = gemmi.Chain("W")
    r = gemmi.Residue()
    r.name, r.seqid, r.het_flag, r.entity_type = "HOH", gemmi.SeqId(1, " "), "H", gemmi.EntityType.Water
    a = gemmi.Atom()
    a.name, a.element, a.occ = "O", gemmi.Element("O"), 1.0
    r.add_atom(a)
    w.add_residue(r)
    st[0].add_chain(w)
    return st

class TestPolymerOnlyGroups(unittest.TestCase):
    """TLS (anisotropic) treatment is for protein and nucleic-acid chains: waters, ligands and glycans are not in the groups"""
    def test_polymer_residues(self):
        st = polymer_with_hetero()
        kinds = {r.name: tls.is_polymer_residue(r) for r in st[0]["A"]}
        self.assertTrue(kinds["ALA"] and kinds["TYR"])
        self.assertFalse(kinds["HOH"] or kinds["NAG"] or kinds["GOL"])

    def test_range_selections_skip_residues_numbered_inside(self):
        st = polymer_with_hetero()
        ch = st[0]["A"]
        poly = [r for r in ch if tls.is_polymer_residue(r)]
        blockers = [r.seqid.num for r in ch if not tls.is_polymer_residue(r)]
        self.assertEqual(tls.range_selections("A", poly, blockers), ["//A/1-10/*", "//A/12-20/*"])  # NAG 11 is skipped
        self.assertEqual(tls.range_selections("A", poly, []), ["//A/1-20/*"])
        self.assertEqual(tls.range_selections("A", [], blockers), [])

    def test_groups_per_chain(self):
        st = polymer_with_hetero()
        groups = tls.groups_per_chain(st)
        self.assertEqual(len(groups), 1)                       # the chain of waters gets no group
        g = groups[0]
        self.assertEqual(len(g.atoms), sum(len(r) for r in st[0]["A"] if tls.is_polymer_residue(r)))
        self.assertEqual(g.selections, ["//A/1-10/*", "//A/12-20/*"])
        # the selections select exactly the atoms of the group (a restart from the written file gives the same group)
        from servalcat.refine.refine import iterate_selection
        serial = {a.serial for a in g.atoms}
        n = 0
        for sel in g.selections:
            for _, _, res, atom in iterate_selection(sel, st):
                self.assertTrue(tls.is_polymer_residue(res), res.name)
                n += 1
        self.assertEqual(n, len(g.atoms))

    def test_user_groups_with_waters_are_reported(self):
        st = polymer_with_hetero()
        for i, cra in enumerate(st[0].all()):
            cra.atom.serial = i + 1
        item = type("Item", (), {"id": 1, "selections": ["//A/*"]})()
        groups = tls.groups_from_selections(st, [item])
        lines = []
        old = tls.logger.writeln
        tls.logger.writeln = lines.append
        try:
            tls.report_non_polymer(st, groups)
        finally:
            tls.logger.writeln = old
        self.assertEqual(len(lines), 1)
        self.assertIn("TLS group 1 includes 8 atoms of non-polymer residues", lines[0])    # NAG 3 + 3 waters + GOL 2

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

    def test_sidecar_roundtrip(self):
        # groups written to the yaml come back with the same residues, origin, T, L and S
        import omegaconf
        from servalcat.refine.refine import RefineConfig
        st = utils.fileio.read_structure(os.path.join(root, "1l2h", "1l2h.cif.gz"))
        for i, cra in enumerate(st[0].all()):
            cra.atom.serial = i + 1
        class Item:
            def __init__(self, i, sels): self.id, self.selections = i, sels
        groups = tls.groups_from_selections(st, [Item(1, ["//A/4-80/*"]), Item(2, ["//A/81-100/*", "//A/120-164/*"])])
        rng = numpy.random.default_rng(1)
        for g in groups:
            g.p = rng.normal(size=tls.NPAR)
        class Model: enforce_psd = True
        Model.groups = groups
        tls.write_tls_config("t.yaml", st, Model)
        cfg = omegaconf.OmegaConf.merge(omegaconf.OmegaConf.create({"refine": RefineConfig()}), omegaconf.OmegaConf.load("t.yaml"))
        self.assertEqual(cfg.refine.tls.input_b, "total")
        groups2 = tls.groups_from_selections(st, cfg.refine.tls.groups)
        for g, g2 in zip(groups, groups2):
            self.assertTrue(g2.has_params)
            numpy.testing.assert_array_equal(g.idx, g2.idx) # same atoms (the two ranges of group 2 are kept)
            numpy.testing.assert_allclose(g2.p, g.p, rtol=1e-12)
            numpy.testing.assert_allclose(g2.origin, g.origin, rtol=1e-12)

    def _two_group_model(self):
        st = utils.fileio.read_structure(os.path.join(root, "1l2h", "1l2h.cif.gz"))
        for i, cra in enumerate(st[0].all()):
            cra.atom.serial = i + 1
            cra.atom.b_iso = 30.
        class Item:
            def __init__(self, i, sels): self.id, self.selections = i, sels
        groups = tls.groups_from_selections(st, [Item(1, ["//A/4-80/*"]), Item(2, ["//A/81-164/*"])])
        groups[0].p[0:6] = [0.3, 0.3, 0., 0., 0., 0.]
        groups[1].p[0:6] = [0.02, 0.02, 0.6, 0., 0., 0.]
        return groups, tls.TlsModel(st, groups)

    def test_subtract_common_aniso(self):
        # the anisotropy common to all TLS contributions is moved out of T (to the overall B_aniso): the total U
        # of every TLS atom changes by the same tensor, T stays positive semi-definite (group 1 needs an isotropic
        # shift from its residual B for that) and no common part is left
        groups, model = self._two_group_model()
        u0 = numpy.array([a.aniso.elements_pdb() for a in model.atoms])
        b = model.subtract_common_aniso()
        self.assertIsNotNone(b)
        u1 = numpy.array([a.aniso.elements_pdb() for a in model.atoms])
        du = numpy.array(b.elements_pdb()) * utils.model.b_to_u
        self.assertGreater(du[2], 0.05) # elongation along z (allowed in P43)
        numpy.testing.assert_allclose(u0 - u1, numpy.tile(du, (len(u0), 1)), atol=1e-6)
        for g in groups:
            self.assertGreaterEqual(numpy.linalg.eigvalsh(g.tensors()[0]).min(), -1e-9)
        self.assertLess(max(a.b_iso for a in groups[0].atoms), 30. - 1.)
        self.assertTrue(all(a.b_iso == 30. for a in groups[1].atoms))
        self.assertIsNone(model.subtract_common_aniso()) # nothing common is left

    def test_subtract_common_aniso_with_invalid_atom(self):
        # an atom whose TLS + residual U is already below the floor (lifted by apply()) or whose residual B is below
        # min_b must not block the transfer for all the others
        groups, model = self._two_group_model()
        bad = groups[1].atoms[0]
        bad.b_iso = -2.
        model.apply()
        self.assertFalse(model.is_valid())
        u0 = numpy.array([a.aniso.elements_pdb() for a in model.atoms])
        b = model.subtract_common_aniso()
        u1 = numpy.array([a.aniso.elements_pdb() for a in model.atoms])
        du = numpy.array(b.elements_pdb()) * utils.model.b_to_u
        self.assertGreater(du[2], 0.05) # the full common part, as without the invalid atom
        keep = numpy.array([a is not bad for a in model.atoms])
        numpy.testing.assert_allclose((u0 - u1)[keep], numpy.tile(du, (keep.sum(), 1)), atol=1e-6)

    def test_restart_after_tls_records_are_lost(self):
        # An editing program (Coot) may drop REMARK 3 / the TLS categories. The yaml written with the
        # refined model (--tls_file) continues the TLS refinement from the previous solution.
        import make_tls_data
        make_tls_data.make(".")
        with open("tls.yaml", "w") as f:
            f.write("refine:\n  tls:\n    groups:\n"
                    "      - id: 1\n        selections: ['//A/4-80/*']\n"
                    "      - id: 2\n        selections: ['//A/81-164/*']\n")
        common = ["", "refine_xtal", "--hklin", "data.mtz", "-s", "xray", "--labin", "FP,SIGFP,FreeR_flag",
                  "--no_solvent", "--hydrogen", "no"]
        sys.argv = common + ["--model", "start.pdb", "--config", "tls.yaml", "--ncycle", "8", "-o", "first"]
        main()
        self.assertTrue(os.path.isfile("first_tls.yaml"))
        with open("first_stats.json") as f:
            r_first = json.load(f)[-1]["data"]["summary"]["Rwork"]
        st = utils.fileio.read_structure("first.pdb")
        self.assertEqual(len(tls.groups_from_model(st)), 2) # the model itself has them
        st.raw_remarks = []                                  # as if the editing program dropped them
        st.meta.refinement = []
        st.write_pdb("edited.pdb")
        self.assertEqual(len(tls.groups_from_model(utils.fileio.read_structure("edited.pdb"))), 0)
        r0 = {}
        for name, extra in (("restart", ["--tls_file", "first_tls.yaml"]), ("naive", ["--config", "tls.yaml"])):
            sys.argv = common + ["--model", "edited.pdb", "--ncycle", "1", "-o", name] + extra
            main()
            with open(f"{name}_stats.json") as f:
                r0[name] = json.load(f)[0]["data"]["summary"]["Rwork"]
        self.assertAlmostEqual(r0["restart"], r_first, delta=0.005)  # continues where it stopped
        self.assertGreater(r0["naive"] - r0["restart"], 0.01)         # starting from T=L=S=0 does not

    def test_tls_per_chain_is_polymer_only(self):
        # 1l2h has 126 waters: they must stay isotropic and be in no group, also after a restart from the written file
        wd = tempfile.mkdtemp(prefix="servaltest_")
        os.chdir(wd)
        try:
            import make_tls_data
            make_tls_data.make(".")
            n_poly = sum(len(r) for r in gemmi.read_structure("start.pdb")[0]["A"] if tls.is_polymer_residue(r))
            sys.argv = ["", "refine_xtal", "--model", "start.pdb", "--hklin", "data.mtz", "-s", "xray", "--labin", "FP,SIGFP,FreeR_flag",
                        "--no_solvent", "--hydrogen", "no", "--ncycle", "2", "--tls_per_chain", "-o", "pc"]
            main()
            log = open("pc.log").read()
            self.assertIn("one group per polymer chain", log)
            self.assertIn(f"group 1: {n_poly} atoms", log)
            n_water_aniso = sum(1 for l in open("pc.pdb") if l.startswith("ANISOU") and l[17:20] == "HOH")
            self.assertEqual(n_water_aniso, 0)
            self.assertEqual(sum(1 for l in open("pc.pdb") if l.startswith("ANISOU")), n_poly)
            import omegaconf
            cfg = omegaconf.OmegaConf.load("pc_tls.yaml")
            self.assertEqual(len(cfg.refine.tls.groups), 1)
            sel = cfg.refine.tls.groups[0].selections[0]
            self.assertNotEqual(sel, "//A/*")                     # a range of the polymer, not the whole chain
            sys.argv = ["", "refine_xtal", "--model", "pc.pdb", "--hklin", "data.mtz", "-s", "xray", "--labin", "FP,SIGFP,FreeR_flag",
                        "--no_solvent", "--hydrogen", "no", "--ncycle", "1", "--tls_file", "pc_tls.yaml", "-o", "pc2"]
            main()
            self.assertIn(f"group 1: {n_poly} atoms", open("pc2.log").read())
        finally:
            os.chdir(root)
            shutil.rmtree(wd)

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

    def test_aniso_fisher_shear_elements(self):
        # Regression test for LL::fisher_diag_from_table: the diagonal of the Fisher matrix of the shear elements
        # (B12, B13, B23) of the anisotropic B must be consistent with that of the diagonal elements (B11, B22, B33);
        # it was 4 times instead of 4/3 times larger, so Fisher/curvature was ~3x larger for shear elements.
        mtzin = os.path.join(root, "1l2h", "1l2h.mtz.gz")
        cifin = os.path.join(root, "1l2h", "1l2h.cif.gz")
        with open("tls.yaml", "w") as f:
            f.write("refine:\n  tls:\n    groups:\n      - id: 1\n        selections: ['//A/4-164/*']\n")
        result = {}
        def check(self_, ll, ncycle=None):
            if result:
                return []
            ll.update_fc()
            atoms = [a for a in self_.atoms if not a.is_hydrogen()]
            sel = atoms[::len(atoms) // 8][:8]
            params = ext.RefineParams(use_aniso=True, use_q_b_mixed=False)
            params.set_model(self_.st[0])
            params.set_params_selected([a.serial - 1 for a in sel], refine_adp=True)
            am = numpy.array(ll.calc_grad(params, store=False).am).reshape(len(sel), 21)
            a2p = params.atom_to_param(ext.RefineParams.Type.B)
            h = 3. # B
            fish, curv = numpy.zeros((len(sel), 6)), numpy.zeros((len(sel), 6))
            for i, a in enumerate(sel):
                fish[i] = am[a2p[a.serial - 1], :6]
                u0 = numpy.array(a.aniso.elements_pdb())
                for k in (0, 1, 2, 3, 4, 5):
                    fs = []
                    for sgn in (1, -1, 0):
                        u = u0.copy()
                        u[k] += sgn * h * utils.model.b_to_u
                        a.aniso = gemmi.SMat33f(*u)
                        ll.update_fc()
                        fs.append(ll.calc_target())
                    a.aniso = gemmi.SMat33f(*u0)
                    curv[i, k] = (fs[0] - 2 * fs[2] + fs[1]) / h**2
            ll.update_fc()
            result["diag"] = fish[:, :3].mean() / curv[:, :3].mean()
            result["shear"] = fish[:, 3:].mean() / curv[:, 3:].mean()
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
        self.assertAlmostEqual(result["diag"], 1., delta=0.4)
        self.assertLess(result["shear"] / result["diag"], 1.8, msg=str(result)) # ~3.2 with the factor 4, ~1.07 with 4/3

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
