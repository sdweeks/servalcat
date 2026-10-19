"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.

TLS (translation/libration/screw) refinement of atomic displacement parameters.

Model
-----
For an atom at position x in a TLS group with origin o, r = x - o,

    U_total = U_tls(r; T, L, S) + U_res,      U_res = B_res / (8 pi^2) * I

    U_tls = T + A L A^T + A S + S^T A^T,      A = [[0, z, -y], [-z, 0, x], [y, -x, 0]]

(Schomaker & Trueblood 1968). Units follow the PDB/Refmac/Phenix convention:
T in A^2, L in deg^2, S in A*deg. This is identical to gemmi.calculate_u_from_tls.
S has 8 free parameters; S33 = -S11 - S22, because adding c*I to S does not change U_tls.

Conventions in the atomic model while refining
----------------------------------------------
  atom.aniso : total anisotropic U (what the structure factor calculation sees)
  atom.b_iso : isotropic *residual* B (what is refined by the main optimiser and
               what the ADP restraints act on)
Before writing the model, finalize() makes b_iso the isotropic equivalent of the total U.

Algorithm (alternating, as in Refmac "refi tlsc")
-------------------------------------------------
Each macro cycle: (1) xyz / residual B / occupancies are refined by the usual
sparse Gauss-Newton step with TLS fixed; (2) T, L, S of every group are refined with
the atomic parameters fixed. In (2) the gradient and Fisher matrix of -LL with respect to
the 6 anisotropic B elements of every atom are projected on the TLS parameters,

    g_p = sum_a J_a^T g_a ,       H_p = sum_a J_a^T H_a J_a ,      J_a = dU_a / dp ,

and a Levenberg-Marquardt step with a line search on -LL is taken. T and L are kept positive
semi-definite (enforce_psd, default) by projecting every trial on the cone, and the total U of every atom
must stay positive definite. With enforce_psd=False only the total U is constrained, as in Phenix; on
test data this let T and L become unphysical (negative T) and R-free drifted up.
After every overall scaling the anisotropic part common to all T contributions is moved to the overall
anisotropic scale (subtract_common_aniso), with which it is degenerate.
"""
from __future__ import absolute_import, division, print_function, generators
import re
import gzip
import numpy
import gemmi
from servalcat.utils import logger
from servalcat import utils
from servalcat import ext
Type = ext.RefineParams.Type
b_to_u = utils.model.b_to_u
u_to_b = utils.model.u_to_b
DEG = numpy.pi / 180.
NPAR = 20 # T(6) + L(6) + S(8)
_UP = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)) # same order as gemmi elements_pdb()
# 21 (T6, L6, S9) -> 20 free parameters (S33 = -S11 - S22)
_P = numpy.zeros((21, NPAR))
_P[:NPAR, :NPAR] = numpy.eye(NPAR)
_P[20, 12] = -1.
_P[20, 16] = -1.
_IU = [numpy.array([i for i, _ in _UP]), numpy.array([j for _, j in _UP])]

def sym33(v6):
    """6-vector (pdb order) -> symmetric 3x3 matrices; works on (...,6) arrays"""
    v6 = numpy.asarray(v6, dtype=float)
    m = numpy.zeros(v6.shape[:-1] + (3, 3))
    for k, (i, j) in enumerate(_UP):
        m[..., i, j] = v6[..., k]
        m[..., j, i] = v6[..., k]
    return m

def sym6(m):
    return numpy.stack([m[..., i, j] for i, j in _UP], axis=-1)

def tls_jacobian(r):
    """dU/dp for atoms at relative positions r (N,3): shape (N, 6, 21).
    p = [T11 T22 T33 T12 T13 T23, L11 .. L23, S11 S12 S13 S21 S22 S23 S31 S32 S33]
    and U is in pdb order (u11 u22 u33 u12 u13 u23). The model is linear in p."""
    r = numpy.asarray(r, dtype=float).reshape(-1, 3)
    x, y, z = r.T
    A = numpy.zeros((len(r), 3, 3))
    A[:, 0, 1] = z
    A[:, 0, 2] = -y
    A[:, 1, 0] = -z
    A[:, 1, 2] = x
    A[:, 2, 0] = y
    A[:, 2, 1] = -x
    At = A.transpose(0, 2, 1)
    J = numpy.zeros((len(r), 6, 21))
    for k, (i, j) in enumerate(_UP):
        m = numpy.zeros((3, 3))
        m[i, j] = m[j, i] = 1.
        J[:, :, k] = sym6(m)
        J[:, :, 6 + k] = sym6(A @ m @ At) * DEG**2
    for k in range(9):
        m = numpy.zeros((3, 3))
        m[k // 3, k % 3] = 1.
        AS = A @ m
        J[:, :, 12 + k] = sym6(AS + AS.transpose(0, 2, 1)) * DEG
    return J

def tls_matrices(p):
    """20-vector -> T, L (3x3 symmetric) and S (3x3, trace-free)"""
    T = sym33(p[0:6])
    L = sym33(p[6:12])
    S = numpy.append(p[12:20], -p[12] - p[16]).reshape(3, 3)
    return T, L, S

def p_from_matrices(T, L, S):
    S = numpy.asarray(S, dtype=float).reshape(3, 3)
    S = S - numpy.trace(S) / 3. * numpy.eye(3) # does not change U_tls
    return numpy.concatenate([sym6(numpy.asarray(T, dtype=float)), sym6(numpy.asarray(L, dtype=float)),
                              S.reshape(-1)[:8]])

def project_psd(p):
    """Project T and L of a parameter vector on the positive semi-definite cone"""
    p = p.copy()
    for sl in (slice(0, 6), slice(6, 12)):
        w, V = numpy.linalg.eigh(sym33(p[sl]))
        if w.min() < 0:
            w = numpy.maximum(w, 0.)
            p[sl] = sym6((V * w) @ V.T)
    return p

def origin_shift_matrix(old_origin, new_origin, points):
    """20x20 matrix M such that the TLS parameters p' = M p about new_origin give the same U field as p about
    old_origin. The U field is a polynomial in the position (T: constant, S: linear, L: quadratic), so the
    transformation is exact; it is determined by least squares from `points`, which must be generic enough
    (at least a few non-degenerate points)."""
    points = numpy.asarray(points, dtype=float).reshape(-1, 3)
    j_old = (tls_jacobian(points - numpy.asarray(old_origin)) @ _P).reshape(-1, NPAR)
    j_new = (tls_jacobian(points - numpy.asarray(new_origin)) @ _P).reshape(-1, NPAR)
    return numpy.linalg.lstsq(j_new, j_old, rcond=None)[0]

class TlsGroup:
    def __init__(self, gid, atoms, selections=None, origin=None, p=None):
        if len(atoms) == 0:
            raise ValueError(f"TLS group {gid} has no atoms")
        self.id = gid
        self.atoms = atoms
        self.idx = numpy.array([a.serial - 1 for a in atoms])
        self.selections = selections or []
        if origin is None:
            pos = self.positions()
            w = numpy.array([a.element.weight for a in atoms])
            origin = (pos * w[:, None]).sum(axis=0) / w.sum()
        self.origin = numpy.asarray(origin, dtype=float)
        self.p = numpy.zeros(NPAR) if p is None else numpy.array(p, dtype=float)
        self.has_params = p is not None # starting values of T, L, S were given
        self.pidx = None # indices in the parameter list of the gradient calculation
        self.spec = [] # list of (local index, 6x6 matrix) for atoms on special positions

    def positions(self):
        return numpy.array([[a.pos.x, a.pos.y, a.pos.z] for a in self.atoms])

    def center_of_mass(self):
        w = numpy.array([a.element.weight for a in self.atoms])
        return (self.positions() * w[:, None]).sum(axis=0) / w.sum()

    def recenter(self, new_origin=None):
        """Move the origin (default: the current centre of mass of the atoms), e.g. after the group was moved by
        rebuilding; T, L, S are transformed such that U_tls at every position is unchanged. Returns the shift (A)."""
        new_origin = self.center_of_mass() if new_origin is None else numpy.asarray(new_origin, dtype=float)
        shift = float(numpy.linalg.norm(new_origin - self.origin))
        if shift > 1e-6:
            pos = self.positions()
            lo, hi = pos.min(axis=0) - 5., pos.max(axis=0) + 5.
            pts = numpy.random.default_rng(0).uniform(lo, hi, size=(60, 3)) # generic points: the fit is exact
            self.p = origin_shift_matrix(self.origin, new_origin, numpy.vstack([pos[:200], pts])) @ self.p
            self.origin = new_origin
        return shift

    def jacobian(self):
        return tls_jacobian(self.positions() - self.origin) @ _P # (N, 6, NPAR)

    def u_tls(self, J=None):
        """TLS contribution to U (N, 6) including the site symmetry constraint"""
        if J is None:
            J = self.jacobian()
        u = J @ self.p
        for i, R in self.spec:
            u[i] = R @ u[i]
        return u

    def tensors(self):
        return tls_matrices(self.p)

class TlsModel:
    def __init__(self, st, groups, ncycle=1, min_b=0.5, specials=(), exclude_h_ll=True, enforce_psd=True):
        self.st = st
        self.groups = groups
        self.ncycle = ncycle
        self.enforce_psd = enforce_psd
        self.min_b = min_b
        self.mu = 0.1 # Levenberg-Marquardt damping (relative to the diagonal)
        self.finalized = False
        seen = {}
        for g in groups:
            for i in g.idx:
                if int(i) in seen:
                    raise SystemExit(f"Error: atom with serial {i+1} belongs to both TLS group {seen[int(i)]} and {g.id}")
                seen[int(i)] = g.id
        self.atoms = [a for g in groups for a in g.atoms]
        self.all_idx = [int(i) for g in groups for i in g.idx]
        n_low = 0
        for a in self.atoms: # same lower limit of B as in the main refinement (RefineParams.set_x)
            if a.b_iso < min_b:
                a.b_iso = min_b
                n_low += 1
        if n_low:
            logger.writeln(f"TLS: residual B of {n_low} atoms set to the minimum value {min_b}")
        self.specials = list(specials)
        spec_map = {s.atom.serial - 1: numpy.array(s.Rspec_aniso, dtype=float) for s in self.specials}
        for g in groups:
            g.spec = [(k, spec_map[int(i)]) for k, i in enumerate(g.idx) if int(i) in spec_map]
        # anisotropic B parameters of the TLS atoms (only for gradient calculation of the -LL target)
        self.params_b = ext.RefineParams(use_aniso=True, use_q_b_mixed=False)
        self.params_b.set_model(st[0])
        self.params_b.set_params_selected(self.all_idx, refine_adp=True)
        if exclude_h_ll:
            self.params_b.exclude_h_ll()
        a2p = numpy.array(self.params_b.atom_to_param(Type.B))
        for g in groups:
            g.pidx = a2p[g.idx]
            if numpy.any(g.pidx < 0):
                raise RuntimeError("TLS atom without parameter")
            if len(g.atoms) < 6:
                logger.writeln(f"WARNING: TLS group {g.id} has only {len(g.atoms)} atoms; "
                               "parameters will be poorly determined")
        self.apply()
    # __init__()

    def residual_b(self):
        return numpy.array([a.b_iso for a in self.atoms])

    def min_residual_b(self):
        return float(self.residual_b().min())

    def _total_u(self, g, J=None):
        u = g.u_tls(J)
        u[:, :3] += (numpy.array([a.b_iso for a in g.atoms]) * b_to_u)[:, None]
        return u

    def _min_eig(self, g):
        return numpy.linalg.eigvalsh(sym33(self._total_u(g)))[:, 0]

    def is_valid(self):
        floor = self.min_b * b_to_u
        return all(self._min_eig(g).min() >= floor - 1e-9 for g in self.groups)

    def apply(self):
        """Set atom.aniso (total U) from the TLS parameters and the residual B (atom.b_iso)"""
        assert not self.finalized
        floor = self.min_b * b_to_u
        n_lifted = 0
        for g in self.groups:
            u = self._total_u(g)
            deficit = numpy.maximum(0., floor - numpy.linalg.eigvalsh(sym33(u))[:, 0])
            if deficit.any(): # safety net: keep U positive definite
                u[:, :3] += deficit[:, None]
                n_lifted += int(numpy.count_nonzero(deficit))
            for a, ui in zip(g.atoms, u):
                a.aniso = gemmi.SMat33f(*ui)
        if n_lifted:
            logger.writeln(f" TLS: total U of {n_lifted} atoms was lifted to keep it positive definite")

    def shift_b(self, delta_b, min_b=0.01):
        """Add delta_b to all B values of the model; for TLS atoms the residual B is shifted"""
        saved = [(a, a.b_iso) for a in self.atoms]
        utils.model.shift_b(self.st[0], delta_b, min_b)
        for a, b in saved:
            a.b_iso = max(b + delta_b, min_b)
        self.apply()

    def subtract_common_aniso(self):
        """Move the anisotropic part common to all TLS contributions from the T tensors to the overall anisotropic
        scale, as sigmaa.subtract_common_aniso_from_model() does for atoms without TLS. The two are degenerate; without
        this the T tensors and B_aniso drift against each other in long runs until T hits the PSD bound and -LL
        rises. T' = T - alpha*uc + lam*I with the smallest lam >= 0 that keeps T' positive semi-definite, and the
        residual B of the group lowered by lam*8pi^2, so the total U of every TLS atom changes by exactly -alpha*uc.
        alpha < 1 only if the residual B would go below min_b or the total U of an atom below its floor (atoms already
        below it, which apply() lifts, are not checked). Returns the B to be added to B_aniso, or None."""
        u = numpy.concatenate([g.u_tls() for g in self.groups])
        u[:, :3] -= u[:, :3].sum(axis=1, keepdims=True) / 3. # traceless: the isotropic part is handled by shift_b
        adpdirs = utils.model.adp_constraints(self.st.find_spacegroup().operations(), self.st.cell, tr0=True)
        uc = adpdirs.dot(u.mean(axis=0)).dot(adpdirs)
        if not numpy.any(numpy.abs(uc) > 1e-8):
            return None
        p_old = [g.p.copy() for g in self.groups]
        b_old = [numpy.array([a.b_iso for a in g.atoms]) for g in self.groups]
        floor = self.min_b * b_to_u
        valid0 = [self._min_eig(g) >= floor - 1e-9 for g in self.groups] # atoms already below are lifted by apply()
        lams = []
        def trial(alpha):
            lams.clear()
            good = True
            for g, p0, b0 in zip(self.groups, p_old, b_old):
                T = sym33(p0[0:6] - alpha * uc)
                lam = max(0., -numpy.linalg.eigvalsh(T).min()) if self.enforce_psd else 0.
                g.p = p0.copy()
                g.p[0:6] = sym6(T + lam * numpy.eye(3))
                for a, b in zip(g.atoms, b0):
                    a.b_iso = b - lam * u_to_b
                lams.append(lam)
                if lam > 0 and b0.min() - lam * u_to_b < self.min_b:
                    good = False
            return good and all(numpy.all(self._min_eig(g)[v] >= floor - 1e-9) for g, v in zip(self.groups, valid0))
        alpha = 1.
        if not trial(alpha): # largest shift that keeps the residual B above min_b and every valid total U above the floor
            lo, hi = 0., 1.
            for _ in range(20):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if trial(mid) else (lo, mid)
            alpha = lo
            trial(alpha)
        self.apply()
        b_aniso = gemmi.SMat33d(*(alpha * uc * u_to_b))
        logger.writeln(f" TLS: common anisotropic component moved to overall B_aniso: B= {b_aniso}"
                       f" (fraction {alpha:.3f}; max isotropic shift from residual B to T {max(lams) * u_to_b:.2f})")
        return b_aniso

    def finalize(self):
        """Make atom.b_iso the isotropic equivalent of the total U (for output)"""
        self.apply()
        self.residual_b_final = self.residual_b()
        for a in self.atoms:
            a.b_iso = a.aniso.trace() / 3. * u_to_b
        self.finalized = True

    def release(self):
        """Remove the anisotropic U set by apply() (atoms keep their residual B in b_iso). The model must not be used
        afterwards; it is for trial fits that must leave the structure as it was."""
        for a in self.atoms:
            a.aniso = gemmi.SMat33f(0, 0, 0, 0, 0, 0)
        self.finalized = True

    # ---- refinement of T, L and S
    def _atom_blocks(self, llobj):
        n = self.params_b.n_refined_atoms(Type.B)
        vn = numpy.array(llobj.vn)
        am = numpy.array(llobj.am)
        assert len(vn) == n * 6 and len(am) == n * 21, (len(vn), len(am), n)
        vn = vn.reshape(n, 6)
        a = am.reshape(n, 21)
        H = numpy.zeros((n, 6, 6))
        for k in range(6):
            H[:, k, k] = a[:, k]
        c = 6
        for k in range(6):
            for l in range(k + 1, 6):
                H[:, k, l] = H[:, l, k] = a[:, c]
                c += 1
        return vn, H

    def _solve(self, g, H):
        d = numpy.diag(H).copy()
        d = numpy.maximum(d, 1e-8 * max(d.max(), 1e-30))
        A = H + self.mu * numpy.diag(d)
        try:
            return -numpy.linalg.solve(A, g)
        except numpy.linalg.LinAlgError:
            return -numpy.linalg.lstsq(A, g, rcond=None)[0]

    def gradient(self, ll):
        """Gradient (20,) and approximate Hessian (20,20) of -LL with respect to the parameters of
        each group: list of (grad, hess). ll.update_fc() must have been called."""
        llobj = ll.calc_grad(self.params_b, specs=self.specials, store=False)
        vn, Hb = self._atom_blocks(llobj)
        ret = []
        for g in self.groups:
            J = g.jacobian()
            gb, hb = vn[g.pidx], Hb[g.pidx]
            # B = 8 pi^2 U
            grad = u_to_b * numpy.einsum("nip,ni->p", J, gb)
            hess = u_to_b**2 * numpy.einsum("nip,nij,njq->pq", J, hb, J)
            ret.append((grad, hess))
        return ret

    def refine(self, ll, ncycle=None):
        """Refine T, L, S of all groups against the -LL target of ll (a LL_Xtal object)"""
        ncycle = self.ncycle if ncycle is None else ncycle
        ret = []
        for icyc in range(ncycle):
            ll.update_fc()
            f0 = ll.calc_target()
            dps = [self._solve(grad, hess) for grad, hess in self.gradient(ll)]
            p_old = [g.p.copy() for g in self.groups]
            f1, scale, accepted = f0, 0., False
            for trial in range(5):
                scale = 0.5**trial
                for g, p0, dp in zip(self.groups, p_old, dps):
                    g.p = p0 + scale * dp
                    if self.enforce_psd:
                        g.p = project_psd(g.p)
                if not self.is_valid():
                    continue
                self.apply()
                ll.update_fc()
                f1 = ll.calc_target()
                logger.writeln(f"TLS step {icyc+1}: scale= {scale:.4f} -LL= {f1:.6e} (current {f0:.6e})")
                if f1 < f0:
                    accepted = True
                    break
            if accepted:
                # the full step is accepted: less damping; a shortened step: more damping
                self.mu = max(self.mu / 3., 1e-3) if scale == 1. else self.mu * (1. if scale == 0.5 else 3.)
            else:
                for g, p0 in zip(self.groups, p_old):
                    g.p = p0
                self.apply()
                ll.update_fc()
                f1, scale = f0, 0.
                self.mu = min(self.mu * 5., 1e3)
                logger.writeln("WARNING: no TLS step decreased -LL; TLS parameters unchanged in this cycle")
            ret.append({"f0": f0, "f1": f1, "scale": scale, "accepted": accepted, "mu": self.mu})
        return ret

    # ---- reporting
    def summary(self):
        out = []
        for g in self.groups:
            T, L, S = g.tensors()
            u = g.u_tls()
            out.append({"id": g.id, "n_atoms": len(g.atoms), "origin": g.origin.tolist(),
                        "T": T.tolist(), "L": L.tolist(), "S": S.tolist(),
                        "T_eigenvalues": numpy.linalg.eigvalsh(T).tolist(),
                        "L_eigenvalues": numpy.linalg.eigvalsh(L).tolist(),
                        "mean_B_tls": float(u[:, :3].sum(axis=1).mean() / 3. * u_to_b)})
        return out

    def show(self):
        logger.writeln("TLS groups")
        for g in self.groups:
            T, L, S = g.tensors()
            u = g.u_tls()
            logger.writeln(f" group {g.id}: {len(g.atoms)} atoms, mean B(TLS)= {u[:, :3].sum(axis=1).mean() / 3. * u_to_b:.2f}")
            logger.writeln(f"  origin: {g.origin[0]:.3f} {g.origin[1]:.3f} {g.origin[2]:.3f}")
            logger.writeln("  T (A^2): " + " ".join(f"{x:.4f}" for x in sym6(T)) + "   (11 22 33 12 13 23)")
            logger.writeln("  L (deg^2): " + " ".join(f"{x:.4f}" for x in sym6(L)) + "   (11 22 33 12 13 23)")
            logger.writeln("  S (A deg): " + " ".join(f"{x:.4f}" for x in S.reshape(-1)) + "   (row-major)")
            tw, lw = numpy.linalg.eigvalsh(T), numpy.linalg.eigvalsh(L)
            if tw.min() < -1e-6 or lw.min() < -1e-6:
                logger.writeln(f"  WARNING: T or L is not positive semi-definite (min eigenvalues: T {tw.min():.4f}, L {lw.min():.4f}); "
                               "the TLS model is not physical for this group")
            if lw.max() > 20.:
                logger.writeln(f"  WARNING: large libration (max eigenvalue of L= {lw.max():.1f} deg^2); check the group definition")
        logger.writeln("")

# ---------------------------------------------------------------------------------
# group definition and input/output
def _seqid_str(seqid):
    icode = seqid.icode if seqid.icode not in (" ", "\0", "") else ""
    return f"{seqid.num}{icode}"

def _selection_string(s):
    """gemmi selection string from gemmi.TlsGroup.Selection. An empty chain / residue range
    (e.g. "SELECTION: ALL" written by Phenix) means everything"""
    chain = s.chain if s.chain else "*"
    if s.res_begin.num is None or s.res_end.num is None:
        return f"//{chain}/*"
    return f"//{chain}/{_seqid_str(s.res_begin)}-{_seqid_str(s.res_end)}/*"

def groups_from_selections(st, items):
    """items: iterable of objects with .id and .selections (list of gemmi selection strings)"""
    from servalcat.refine.refine import iterate_selection
    groups = []
    for item in items:
        atoms, seen = [], set()
        for s in item.selections:
            n = 0
            for _, _, _, atom in iterate_selection(s, st):
                n += 1
                if atom.serial not in seen:
                    seen.add(atom.serial)
                    atoms.append(atom)
            if n == 0:
                logger.writeln(f"WARNING: no atom found for the selection (TLS group {item.id}): {s}")
        if not atoms:
            raise SystemExit(f"Error: TLS group {item.id} is empty")
        origin, p = None, None
        Tv, Lv, Sv = (_optional_floats(item, k, n) for k, n in (("T", 6), ("L", 6), ("S", 9)))
        if Tv is not None and Lv is not None and Sv is not None:
            p = p_from_matrices(sym33(Tv), sym33(Lv), numpy.array(Sv).reshape(3, 3))
            origin = _optional_floats(item, "origin", 3)
        elif any(x is not None for x in (Tv, Lv, Sv)):
            raise SystemExit(f"Error: TLS group {item.id}: T, L and S must be given all together")
        groups.append(TlsGroup(item.id, atoms, list(item.selections), origin=origin, p=p))
    return groups

def _optional_floats(item, key, n):
    v = item.get(key) if hasattr(item, "get") else getattr(item, key, None)
    if v is None:
        return None
    v = [float(x) for x in v]
    if len(v) != n:
        raise SystemExit(f"Error: TLS group {item.id}: {key} must have {n} values")
    return v

def is_polymer_residue(res):
    """True for a residue of a protein or nucleic-acid chain (modified residues included); False for waters, ligands,
    glycans and other non-polymer residues. Anisotropic (TLS) treatment is meant for the polymer only."""
    if res.is_water():
        return False
    if res.entity_type == gemmi.EntityType.Polymer:
        return True
    info = gemmi.find_tabulated_residue(res.name)
    return info is not None and (info.is_amino_acid() or info.is_nucleic_acid())

def range_selections(chain_name, residues, blockers=()):
    """gemmi selections for the given residues (in chain order): one per stretch of consecutive residues. A stretch is
    split where the number of a residue that must not be selected (blockers: residue numbers of waters, ligands, ..
    of the chain) lies between two selected residues, so that a range never takes in a non-polymer residue that is
    numbered inside the polymer."""
    import bisect
    blockers = sorted(set(blockers))
    stretches, start, prev = [], None, None
    for r in residues:
        if start is None:
            start = prev = r
            continue
        lo = bisect.bisect_right(blockers, prev.seqid.num)
        hi = bisect.bisect_left(blockers, r.seqid.num)
        if hi > lo:
            stretches.append((start, prev))
            start = r
        prev = r
    if start is not None:
        stretches.append((start, prev))
    out = []
    for a, b in stretches:
        sa, sb = _seqid_str(a.seqid), _seqid_str(b.seqid)
        out.append(f"//{chain_name}/{sa}/*" if sa == sb else f"//{chain_name}/{sa}-{sb}/*")
    return out

def groups_per_chain(st):
    """One group for the polymer residues of every chain; waters, ligands and other non-polymer residues are in no
    group (they keep isotropic B). The selections never include them either, so that a restart from the written
    configuration gives the same groups."""
    groups = []
    for chain in st[0]:
        residues = [r for r in chain if is_polymer_residue(r)]
        atoms = [a for r in residues for a in r]
        if atoms:
            blockers = [r.seqid.num for r in chain if not is_polymer_residue(r)]
            groups.append(TlsGroup(len(groups) + 1, atoms, range_selections(chain.name, residues, blockers)))
    return groups

def report_non_polymer(st, groups):
    """Groups that are defined by the user (or taken from a model) may include waters or ligands; say so"""
    nonpoly = {a.serial for chain in st[0] for r in chain if not is_polymer_residue(r) for a in r}
    for g in groups:
        n = sum(1 for a in g.atoms if a.serial in nonpoly)
        if n:
            logger.writeln(f"NOTE: TLS group {g.id} includes {n} atoms of non-polymer residues (waters, ligands). Anisotropic "
                           "treatment is usually meant for protein and nucleic-acid chains only; use ranges of the polymer "
                           "residues if they should keep isotropic B.")

_RE_TLS_VALUE = re.compile(r"([TLS][123][123]):\s*(-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)")
_RE_SEQ = re.compile(r"^(-?\d+)([A-Za-z]?)$")

def parse_remark3_tls(remarks):
    """Parse the TLS section of REMARK 3 (Refmac/Servalcat style). gemmi only reads it for known programs.
    Returns a list of (id, selection strings, origin, p)"""
    ret, cur = [], None
    def finish():
        if cur is None:
            return
        v = cur["values"]
        try:
            Tm = sym33([v[k] for k in ("T11", "T22", "T33", "T12", "T13", "T23")])
            Lm = sym33([v[k] for k in ("L11", "L22", "L33", "L12", "L13", "L23")])
            Sm = numpy.array([[v[f"S{i}{j}"] for j in (1, 2, 3)] for i in (1, 2, 3)])
        except KeyError:
            logger.writeln(f"WARNING: incomplete TLS group {cur['id']} in REMARK 3; ignored")
            return
        if cur["origin"] is None or not cur["sels"]:
            logger.writeln(f"WARNING: incomplete TLS group {cur['id']} in REMARK 3; ignored")
            return
        ret.append((cur["id"], cur["sels"], cur["origin"], p_from_matrices(Tm, Lm, Sm)))
    for line in remarks:
        t = line[10:] if line.startswith("REMARK   3") else line
        m = re.match(r"\s*TLS GROUP\s*:\s*(\S+)", t)
        if m:
            finish()
            cur = {"id": m.group(1), "sels": [], "origin": None, "values": {}}
            continue
        if cur is None:
            continue
        m = re.match(r"\s*RESIDUE RANGE\s*:\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", t)
        if m:
            c1, b, _, e = m.groups()
            if _RE_SEQ.match(b) and _RE_SEQ.match(e):
                cur["sels"].append(f"//{c1}/{b}-{e}/*")
            continue
        m = re.match(r"\s*ORIGIN FOR THE GROUP.*?:\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)", t)
        if m:
            cur["origin"] = [float(x) for x in m.groups()]
            continue
        for k, x in _RE_TLS_VALUE.findall(t):
            cur["values"][k] = float(x)
    finish()
    return ret

def _model_tls_records(st):
    """TLS groups stored in the model: list of (id, selection strings, origin, p)"""
    recs = []
    if st.meta.refinement:
        for tg in st.meta.refinement[0].tls_groups:
            p = p_from_matrices(sym33(list(tg.T.elements_pdb())), sym33(list(tg.L.elements_pdb())),
                                numpy.array(tg.S.tolist()))
            recs.append((tg.id, [_selection_string(s) for s in tg.selections],
                         [tg.origin.x, tg.origin.y, tg.origin.z], p))
    if not recs:
        recs = parse_remark3_tls(st.raw_remarks)
    return recs

def groups_from_model(st):
    """TLS groups (with parameters) stored in the input model. Returns [] if none."""
    from servalcat.refine.refine import iterate_selection
    groups = []
    for gid, sels, origin, p in _model_tls_records(st):
        atoms, seen = [], set()
        for sel in sels:
            for _, _, _, atom in iterate_selection(sel, st):
                if atom.serial not in seen:
                    seen.add(atom.serial)
                    atoms.append(atom)
        if not atoms:
            logger.writeln(f"WARNING: TLS group {gid} in the input model has no atoms; ignored")
            continue
        if not numpy.allclose(p, project_psd(p)):
            logger.writeln(f"WARNING: T or L of TLS group {gid} in the input model is not positive semi-definite")
        try:
            gid = int(gid)
        except ValueError:
            gid = len(groups) + 1
        groups.append(TlsGroup(gid, atoms, sels, origin=origin, p=p))
    return groups

def detect_residual_only(path):
    """True if the file says that its B/ANISOU are residual (without TLS contribution), as Refmac does"""
    try:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", errors="replace") as f:
            txt = f.read()
    except OSError:
        return False
    return re.search(r"U\s*VALUES\s*:\s*RESIDUAL\s*ONLY", txt, re.I) is not None or \
        re.search(r"U values:\s*residual only", txt, re.I) is not None

def initialise_residual_from_total(model, min_b=0.5):
    """Input B-factors are total (iso-equivalent): remove the TLS contribution to get residual B"""
    for g in model.groups:
        btls = g.u_tls()[:, :3].sum(axis=1) / 3. * u_to_b
        for a, b in zip(g.atoms, btls):
            a.b_iso = max(a.b_iso - b, min_b)
    model.apply()

def setup_tls(st, cfg_tls, per_chain=False, ncycle=None, specials=(), exclude_h_ll=True, model_path=None):
    """Create a TlsModel from (in this order of priority) the config, the input model, or per chain"""
    from_model = False
    if cfg_tls.groups:
        groups = groups_from_selections(st, cfg_tls.groups)
        logger.writeln(f"TLS: {len(groups)} group(s) defined in the config")
    else:
        groups = groups_from_model(st)
        from_model = bool(groups)
        if groups:
            logger.writeln(f"TLS: {len(groups)} group(s) taken from the input model")
        elif per_chain:
            groups = groups_per_chain(st)
            if not groups:
                raise SystemExit("Error: --tls_per_chain: the model has no protein or nucleic-acid chain")
            logger.writeln(f"TLS: one group per polymer chain ({len(groups)} groups); waters, ligands and other "
                           "non-polymer residues are not in a group")
        else:
            raise SystemExit("Error: TLS refinement requested, but no TLS groups are defined. Give "
                             "refine.tls.groups in the config file, provide a model with TLS groups, "
                             "or use --tls_per_chain.")
    if ncycle is None:
        ncycle = cfg_tls.ncycle
    if cfg_tls.groups or from_model:
        report_non_polymer(st, groups)
    for g in groups:
        if g.has_params: # the group may have moved since T, L, S were determined (rebuilding)
            shift = g.recenter()
            if shift > 0.5:
                logger.writeln(f"TLS group {g.id}: origin moved by {shift:.2f} A to the current centre of mass "
                               "(T, L, S transformed; the TLS model is unchanged)")
    model = TlsModel(st, groups, ncycle=ncycle, specials=specials, exclude_h_ll=exclude_h_ll,
                     enforce_psd=cfg_tls.enforce_psd)
    if any(g.has_params for g in groups):
        mode = cfg_tls.input_b
        if mode not in ("auto", "total", "residual"):
            raise SystemExit(f"Error: refine.tls.input_b must be auto, total or residual (got {mode})")
        if mode == "auto":
            mode = "residual" if model_path and detect_residual_only(model_path) else "total"
        if mode == "total":
            logger.writeln("TLS: B-factors in the input model are taken as total; TLS contribution removed to get residual B")
            initialise_residual_from_total(model)
        else:
            logger.writeln("TLS: B-factors in the input model are taken as residual")
    model.show()
    return model

def _runs(st, g):
    """(chain, first seqid, last seqid) of consecutive residues containing atoms of group g"""
    idx = {int(i) for i in g.idx}
    ret = []
    for chain in st[0]:
        run = None
        for res in chain:
            if any(a.serial - 1 in idx for a in res):
                if run is None:
                    run = [res.seqid, res.seqid]
                else:
                    run[1] = res.seqid
            elif run is not None:
                ret.append((chain.name, *run))
                run = None
        if run is not None:
            ret.append((chain.name, *run))
    return ret

def to_gemmi_groups(st, model):
    ret = []
    for g in model.groups:
        tg = gemmi.TlsGroup()
        tg.id = str(g.id)
        T, L, S = g.tensors()
        tg.origin = gemmi.Position(*g.origin)
        tg.T = gemmi.SMat33d(*sym6(T))
        tg.L = gemmi.SMat33d(*sym6(L))
        tg.S = gemmi.Mat33(S.tolist())
        sels = []
        for chain, b, e in _runs(st, g):
            s = gemmi.TlsGroup.Selection()
            s.chain = chain
            s.res_begin = b
            s.res_end = e
            sels.append(s)
        tg.selections = sels
        ret.append(tg)
    return ret

def remark3_lines(st, model):
    """REMARK 3 lines (TLS section) in the format Refmac/Phenix write"""
    lines = ["REMARK   3", "REMARK   3  TLS DETAILS", f"REMARK   3   NUMBER OF TLS GROUPS  : {len(model.groups)}",
             "REMARK   3   U VALUES      : WITH TLS ADDED", "REMARK   3"]
    for g in model.groups:
        T, L, S = g.tensors()
        runs = _runs(st, g)
        lines.append(f"REMARK   3   TLS GROUP : {g.id}")
        lines.append(f"REMARK   3    NUMBER OF COMPONENTS GROUP : {len(runs)}")
        lines.append("REMARK   3    COMPONENTS        C SSSEQI   TO  C SSSEQI")
        for chain, b, e in runs:
            lines.append(f"REMARK   3    RESIDUE RANGE :   {chain:1s} {_seqid_str(b):>5s}        {chain:1s} {_seqid_str(e):>5s}")
        lines.append(f"REMARK   3    ORIGIN FOR THE GROUP (A): {g.origin[0]:9.4f}{g.origin[1]:9.4f}{g.origin[2]:9.4f}")
        lines.append("REMARK   3    T TENSOR")
        lines.append(f"REMARK   3      T11: {T[0,0]:9.4f} T22: {T[1,1]:9.4f}")
        lines.append(f"REMARK   3      T33: {T[2,2]:9.4f} T12: {T[0,1]:9.4f}")
        lines.append(f"REMARK   3      T13: {T[0,2]:9.4f} T23: {T[1,2]:9.4f}")
        lines.append("REMARK   3    L TENSOR")
        lines.append(f"REMARK   3      L11: {L[0,0]:9.4f} L22: {L[1,1]:9.4f}")
        lines.append(f"REMARK   3      L33: {L[2,2]:9.4f} L12: {L[0,1]:9.4f}")
        lines.append(f"REMARK   3      L13: {L[0,2]:9.4f} L23: {L[1,2]:9.4f}")
        lines.append("REMARK   3    S TENSOR")
        lines.append(f"REMARK   3      S11: {S[0,0]:9.4f} S12: {S[0,1]:9.4f} S13: {S[0,2]:9.4f}")
        lines.append(f"REMARK   3      S21: {S[1,0]:9.4f} S22: {S[1,1]:9.4f} S23: {S[1,2]:9.4f}")
        lines.append(f"REMARK   3      S31: {S[2,0]:9.4f} S32: {S[2,1]:9.4f} S33: {S[2,2]:9.4f}")
        lines.append("REMARK   3")
    return lines

def update_meta_tls(st, model):
    """Store TLS groups in the structure's metadata and PDB remarks (call after update_meta)"""
    if not st.meta.refinement:
        st.meta.refinement = [gemmi.RefinementInfo()]
    st.meta.refinement[0].tls_groups = to_gemmi_groups(st, model)
    st.raw_remarks = list(st.raw_remarks) + remark3_lines(st, model)

def tls_config_dict(st, model, with_parameters=True):
    """TLS definition in the format of the refine.tls section of the config file"""
    groups = []
    for g in model.groups:
        d = {"id": int(g.id),
             "selections": [f"//{c}/{_seqid_str(b)}-{_seqid_str(e)}/*" for c, b, e in _runs(st, g)]}
        if with_parameters:
            T, L, S = g.tensors()
            d["origin"] = [float(x) for x in g.origin]
            d["T"] = [float(x) for x in sym6(T)]
            d["L"] = [float(x) for x in sym6(L)]
            d["S"] = [float(x) for x in S.reshape(-1)]
        groups.append(d)
    tls = {"groups": groups, "enforce_psd": bool(model.enforce_psd)}
    if with_parameters:
        tls["input_b"] = "total" # the model written together has the B-factors of the total U
    return {"refine": {"tls": tls}}

def write_tls_config(path, st, model, with_parameters=True):
    """Write the TLS groups (residue ranges and, optionally, T, L, S) as yaml for --tls_file, so that refinement can
    be continued after the model was edited (e.g. in Coot) even if the editing program drops the TLS records"""
    import omegaconf
    omegaconf.OmegaConf.save(omegaconf.OmegaConf.create(tls_config_dict(st, model, with_parameters)), path)
    logger.writeln(f"TLS definition written: {path}")
    if with_parameters:
        logger.writeln(f" (to continue after editing the model: --tls_file {path})")
