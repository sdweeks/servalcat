"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.

Automated selection of TLS groups (servalcat util tls_select)

Each polymer chain is divided into contiguous segments. For N = 1, 2, ... groups per chain the best
division is found by dynamic programming, using the gain in the likelihood target that TLS is predicted to give
for every candidate segment (the quadratic model of -LL with respect to T, L and S about the isotropic model; the same
gradient and Fisher matrix that the TLS refinement uses). Every N is then evaluated by a short refinement (coordinates
fixed; residual B and TLS refined) with the same machinery as refine_xtal, and R-values are reported as a function of N.
The result is a table/graph, the recommended N and the residue ranges as a yaml file for refine_xtal --tls_file.

The selection is conservative. Let R(N) be Rfree (or R without a test set) after refinement with N groups per
chain, R(0) without TLS, and d(N) = R(N-1) - R(N). TLS is recommended at all only if d(1) >= min_delta. The
number of groups is then the lowest N >= 2 whose jump d(N) is within `tie` of the biggest jump among N >= 2 -
and 1 if no such jump reaches min_delta.
"""
from __future__ import absolute_import, division, print_function, generators
import argparse
import contextlib
import copy
import csv
import json
import os
import re
import textwrap
import time
import types
import numpy
import pandas
import gemmi
from servalcat.utils import logger
from servalcat import utils
from servalcat.refine import refine_xtal
from servalcat.refine import tls as tls_mod

u_to_b = utils.model.u_to_b

# options of refine_xtal that tls_select sets itself or does not use: not shown in its help
_HIDDEN = ("ncycle", "fix_xyz", "tls", "tls_per_chain", "tls_file", "tls_ncycle", "write_trajectory",
           "no_weight_adjust", "target_bond_rmsz_range", "adp", "randomize", "refine_occ", "occ_waters",
           "occ_adpr_weight")

def add_arguments(parser, method="likelihood"):
    bfit = method == "bfactor"

    def own_options(parser):
        add = utils.cli.add_arg
        g = parser.add_argument_group("TLS selection: optional arguments with default values")
        add(g, "--max_groups", type=int, default=8, metavar="N",
            help="Largest number of groups per chain in the scan")
        add(g, "--min_residues", type=int, default=10, metavar="N",
            help="Smallest number of residues in a group")
        add(g, "--tls_cycles", type=int, default=8, metavar="N",
            help="Cycles of the refinement (coordinates fixed) that evaluates each number of groups")
        add(g, "--min_delta", type=float, default=0.001, metavar="X",
            help="Smallest decrease of R that counts as an improvement")
        add(g, "--tie", type=float, default=0.25, metavar="X",
            help="Jumps within this fraction of the biggest one are equivalent; the lowest number of groups "
            "among them is chosen")
        if bfit:
            add(g, "--atoms", choices=list(ATOM_SETS), default="all",
                help="Atoms whose B-factors are fitted: all heavy atoms; main: main-chain (and sugar-phosphate) atoms; "
                "ca: C-alpha / P atoms (then groups need more residues)")
            add(g, "--max_boundaries", type=int, default=400, metavar="N",
                help="Largest number of positions where a chain may be cut. Chains with more residues are cut only at "
                "every k-th residue; with every residue the fit is done for every possible segment (about 1 ms per "
                "segment: 300 residues take about 1 minute)")
            add(g, "--min_gain", type=float, default=0.05, metavar="X",
                help="Without refinement (no --hklin): the number of groups is the smallest N for which one more group "
                "lowers the residual of the B-factor fit by less than this fraction")
        add(g, "-j", "--jobs", type=int, default=0, metavar="N", show="0 = all available CPUs",
            help="Number of trial refinements run in parallel, each in its own single-threaded process; "
            "the next one starts when one finishes. Every process holds its own copy of the data, so use a "
            "smaller number if memory is short; 1 = one after another")
        g = parser.add_argument_group("TLS selection: options (off unless given)")
        add(g, "--ngroups", nargs="+", metavar="N | CHAIN:N", show="none: scan the number of groups",
            help="Do not scan: use this number of groups in every polymer chain (e.g. 4), or per chain "
            "(e.g. A:4 B:3; chains not given: 1). The program finds the residue ranges")
        add(g, "--no_eval", action="store_true",
            help=("Do not refine to get R-values (always so without --hklin): only the fit of the B-factors" if bfit else
                  "With --ngroups: only determine the residue ranges, no refinement for R-values"))

    refine_xtal.add_arguments(parser, after_required=own_options, hklin_required=not bfit)
    if bfit:
        parser.usage = "{} --model FILE [--hklin FILE] [options]".format(parser.prog)
        parser.description = textwrap.fill(
            "Select TLS groups from the B-factors of a model, in the spirit of TLSMD: every polymer chain is divided "
            "into contiguous segments, a TLS model is fitted by least squares to the isotropic B-factors of each "
            "segment, and the division with the smallest residual is found for every number of groups. The model "
            "should have been refined WITHOUT TLS, so that its B-factors contain the motion. Without --hklin only this "
            "fit is done (no reflection data or monomer library needed). With --hklin every number of groups is also "
            "tested by refinement and R-values are reported, as 'util tls_select' does. The result is written as yaml "
            "for 'refine_xtal --tls_file'.", 78)
        parser.epilog = "Example: {} --model refined.pdb -o tlsbfit".format(parser.prog)
        for a in parser._actions:
            if a.dest in _HIDDEN:
                a.help = argparse.SUPPRESS
        return
    parser.description = textwrap.fill(
        "Select TLS groups automatically: divide every polymer chain into contiguous segments and show how "
        "R-values depend on the number of groups. The model should be an isotropic model refined with refine_xtal. "
        "The result is written as yaml for 'refine_xtal --tls_file'. Data, model and restraint options are those "
        "of refine_xtal.", 78)
    parser.epilog = "Example: {} --model refined.pdb --hklin data.mtz -o tlssel".format(parser.prog)
    for a in parser._actions:
        if a.dest in _HIDDEN:
            a.help = argparse.SUPPRESS
# add_arguments()

def set_prefix(args):
    if not args.output_prefix:
        args.output_prefix = utils.fileio.splitext(os.path.basename(args.model))[0] + \
                             ("_tlsbfit" if getattr(args, "method", "likelihood") == "bfactor" else "_tlsselect")

# ---------------------------------------------------------------------------------------------------
# dynamic programming
def candidate_boundaries(n, max_boundaries=120):
    """Allowed positions of segment boundaries among n residues (0 and n always included)"""
    step = max(1, int(numpy.ceil(n / max_boundaries)))
    b = sorted(set(list(range(0, n, step)) + [n]))
    return b

def best_partitions(gain, bounds, min_residues, kmax):
    """gain[s, t]: gain of the segment between boundaries bounds[s] and bounds[t] (s < t).
    Returns {k: (total gain, [residue index of boundaries, from 0 to n])} for the k that are possible with
    segments of at least min_residues residues"""
    m = len(bounds)
    ok = numpy.full((m, m), False)
    for s in range(m):
        for t in range(s + 1, m):
            ok[s, t] = bounds[t] - bounds[s] >= min_residues
    g = numpy.where(ok, gain, -numpy.inf)
    F = numpy.full((kmax + 1, m), -numpy.inf)
    arg = numpy.zeros((kmax + 1, m), dtype=int)
    F[0, 0] = 0.
    for k in range(1, kmax + 1):
        for t in range(1, m):
            cand = F[k - 1, :t] + g[:t, t]
            s = int(numpy.argmax(cand))
            F[k, t] = cand[s]
            arg[k, t] = s
    ret = {}
    for k in range(1, kmax + 1):
        if not numpy.isfinite(F[k, m - 1]):
            continue
        idx, t = [m - 1], m - 1
        for kk in range(k, 0, -1):
            t = arg[kk, t]
            idx.append(t)
        ret[k] = (float(F[k, m - 1]), [bounds[i] for i in reversed(idx)])
    return ret

class ChainGain:
    """Predicted decrease of -LL by TLS for every segment of consecutive residues of a chain.

    From the per-atom gradient g_a and Fisher matrix H_a of -LL with respect to the 6 anisotropic B elements,
    the gradient and Hessian with respect to the 20 TLS parameters of a segment are sums over its atoms
    (J_a^T g_a, J_a^T H_a J_a); they are accumulated per residue (prefix sums, about one origin) and moved to the
    centre of mass of the segment (an exact linear transformation of the parameters). The gain is that of a
    damped Newton step: 0.5 g^T (H + mu diag(H))^-1 g."""
    def __init__(self, res_index, pos, mass, gb, Hb, n_res, mu=0.1):
        self.n_res = n_res
        self.mu = mu
        self.origin = (pos * mass[:, None]).sum(axis=0) / mass.sum()
        J = tls_mod.tls_jacobian(pos - self.origin) @ tls_mod._P # (N, 6, 20)
        ga = u_to_b * numpy.einsum("nip,ni->np", J, gb)
        Ha = u_to_b**2 * numpy.einsum("nip,nij,njq->npq", J, Hb, J)
        self.pg = numpy.zeros((n_res + 1, tls_mod.NPAR))
        self.ph = numpy.zeros((n_res + 1, tls_mod.NPAR, tls_mod.NPAR))
        self.pw = numpy.zeros(n_res + 1)
        self.pp = numpy.zeros((n_res + 1, 3))
        for r in range(n_res):
            sel = res_index == r
            self.pg[r + 1] = self.pg[r] + ga[sel].sum(axis=0)
            self.ph[r + 1] = self.ph[r] + Ha[sel].sum(axis=0)
            self.pw[r + 1] = self.pw[r] + mass[sel].sum()
            self.pp[r + 1] = self.pp[r] + (pos[sel] * mass[sel, None]).sum(axis=0)
        self.pts = numpy.random.default_rng(0).normal(size=(60, 3)) * 12.

    def gain(self, i, j):
        """segment of residues i .. j-1"""
        g = self.pg[j] - self.pg[i]
        H = self.ph[j] - self.ph[i]
        w = self.pw[j] - self.pw[i]
        if w <= 0:
            return 0.
        o_seg = (self.pp[j] - self.pp[i]) / w
        # p_chain = M p_seg (same U field)
        M = tls_mod.origin_shift_matrix(o_seg, self.origin, self.pts + o_seg)
        g, H = M.T @ g, M.T @ H @ M
        d = numpy.maximum(numpy.diag(H), 1e-8 * max(numpy.diag(H).max(), 1e-30))
        A = H + self.mu * numpy.diag(d)
        try:
            x = numpy.linalg.solve(A, g)
        except numpy.linalg.LinAlgError:
            x = numpy.linalg.lstsq(A, g, rcond=None)[0]
        return float(0.5 * g @ x)

    def gain_matrix(self, bounds):
        m = len(bounds)
        G = numpy.full((m, m), -numpy.inf)
        for s in range(m):
            for t in range(s + 1, m):
                G[s, t] = self.gain(bounds[s], bounds[t])
        return G


def ls_residual(H, g, yy):
    """Minimum of sum (b - X p)^2 from the normal equations H = X^T X, g = X^T b, yy = sum b^2. The parameters that the
    data do not determine (H is singular) do not matter, only the residual is returned."""
    d = numpy.sqrt(numpy.maximum(numpy.diag(H), 1e-30))
    Hs, gs = H / d[:, None] / d[None, :], g / d
    w, V = numpy.linalg.eigh(Hs)
    keep = w > 1e-9 * w.max()                     # the combinations that the data determine
    c = V[:, keep].T @ gs
    return float(max(yy - numpy.sum(c * c / w[keep]), 0.))

def bfit_residual(pos, b):
    """Least-squares fit of ONE TLS group to the isotropic B-factors b of atoms at pos (N, 3): the residual sum of squares"""
    pos = numpy.asarray(pos, float)
    b = numpy.asarray(b, float)
    J = tls_mod.tls_jacobian(pos - pos.mean(axis=0)) @ tls_mod._P
    X = u_to_b * (J[:, 0] + J[:, 1] + J[:, 2]) / 3.
    return ls_residual(X.T @ X, X.T @ b, float(b @ b))


class ChainBFit:
    """Least-squares fit of a TLS model to the isotropic B-factors of the atoms of a segment of consecutive residues
    (the idea of TLSMD: the B-factors of a model describe what the TLS groups would have to explain). The isotropic B
    of an atom is u_to_b * trace(U_TLS) / 3, linear in the 20 TLS parameters. Normal equations are accumulated per
    residue (prefix sums, about one origin) and moved to the centre of the segment (an exact linear transformation of the
    parameters, as in ChainGain); the minimum of sum (B_obs - B_TLS)^2 follows from the normal equations. Only the
    residual is needed, so the parameters that an isotropic B cannot determine do not matter. The fit is unconstrained
    (T and L are not forced to be positive semi-definite); the refinement that follows is."""
    def __init__(self, res_index, pos, b, n_res):
        self.n_res = n_res
        self.origin = pos.mean(axis=0)
        J = tls_mod.tls_jacobian(pos - self.origin) @ tls_mod._P # (N, 6, 20)
        X = u_to_b * (J[:, 0] + J[:, 1] + J[:, 2]) / 3.           # (N, 20): B of the atom per TLS parameter
        Ha = numpy.einsum("np,nq->npq", X, X)
        ga = X * b[:, None]
        self.ph = numpy.zeros((n_res + 1, tls_mod.NPAR, tls_mod.NPAR))
        self.pg = numpy.zeros((n_res + 1, tls_mod.NPAR))
        self.pyy = numpy.zeros(n_res + 1)
        self.pn = numpy.zeros(n_res + 1)
        self.pp = numpy.zeros((n_res + 1, 3))
        for r in range(n_res):
            sel = res_index == r
            self.ph[r + 1] = self.ph[r] + Ha[sel].sum(axis=0)
            self.pg[r + 1] = self.pg[r] + ga[sel].sum(axis=0)
            self.pyy[r + 1] = self.pyy[r] + (b[sel] ** 2).sum()
            self.pn[r + 1] = self.pn[r] + sel.sum()
            self.pp[r + 1] = self.pp[r] + pos[sel].sum(axis=0)
        self.pts = numpy.random.default_rng(0).normal(size=(60, 3)) * 12.

    def residual(self, i, j):
        """sum of squares of B_obs - B_TLS for residues i .. j-1 after the best fit"""
        n = self.pn[j] - self.pn[i]
        if n <= 0:
            return 0.
        o_seg = (self.pp[j] - self.pp[i]) / n
        M = tls_mod.origin_shift_matrix(o_seg, self.origin, self.pts + o_seg)
        g = M.T @ (self.pg[j] - self.pg[i])
        H = M.T @ (self.ph[j] - self.ph[i]) @ M
        return ls_residual(H, g, self.pyy[j] - self.pyy[i])

    def gain_matrix(self, bounds):
        """minus the residual of every segment, so that the division with the largest total is the best fit"""
        m = len(bounds)
        G = numpy.full((m, m), -numpy.inf)
        for s in range(m):
            for t in range(s + 1, m):
                G[s, t] = -self.residual(bounds[s], bounds[t])
        return G

ATOM_SETS = {"all": None, "main": ("N", "CA", "C", "O", "P", "OP1", "OP2", "O5'", "C5'", "C4'", "C3'", "O3'"), "ca": ("CA", "P")}

def partitions_by_bfactor(chains, args):
    """best divisions of every chain for each k by the B-factor fit. Returns {chain name: {k: (-residual, bounds)}};
    also stores the number of atoms of each chain in args._bfit_atoms"""
    names = ATOM_SETS[args.atoms]
    ret, natoms = {}, {}
    for c in chains:
        res_index, pos, b = [], [], []
        for i, r in enumerate(c.residues):
            for a in r:
                if a.is_hydrogen() or a.occ <= 0 or (names is not None and a.name not in names):
                    continue
                res_index.append(i)
                pos.append([a.pos.x, a.pos.y, a.pos.z])
                b.append(a.b_iso)
        res_index, pos, b = numpy.array(res_index), numpy.array(pos), numpy.array(b)
        natoms[c.name] = len(b)
        # about 10 parameters can be determined from isotropic B: a segment needs clearly more atoms than that
        per_res = max(1., len(b) / c.n)
        min_res = max(args.min_residues, int(numpy.ceil(25. / per_res)))
        if min_res > args.min_residues:
            logger.writeln(f"Chain {c.name}: with the atoms chosen ({args.atoms}) a group needs at least {min_res} residues")
        cf = ChainBFit(res_index, pos, b, c.n)
        max_b = getattr(args, "max_boundaries", 400)
        bounds = candidate_boundaries(c.n, max_b) # every residue as a possible cut, unless the chain is very long
        if len(bounds) < c.n + 1:
            logger.writeln(f"Chain {c.name}: cuts only at every {bounds[1] - bounds[0]}th residue (--max_boundaries {max_b})")
        kmax = min(args.max_groups, c.n // min_res)
        t0 = time.time()
        ret[c.name] = best_partitions(cf.gain_matrix(bounds), bounds, min_res, kmax)
        logger.writeln(f"Chain {c.name}: {c.n} residues, {len(b)} atoms in the fit, best divisions for up to "
                       f"{max(ret[c.name])} groups ({time.time() - t0:.1f} s)")
    args._bfit_atoms = natoms
    return ret

def choose_n_elbow(chi2, min_gain=0.05):
    """chi2[i]: residual of the B-factor fit with i+1 groups per chain. Returns (N, explanation): the smallest N after
    which one more group lowers the residual by less than min_gain (relative)"""
    K = len(chi2)
    for n in range(1, K):
        drop = (chi2[n - 1] - chi2[n]) / chi2[n - 1] if chi2[n - 1] > 0 else 0.
        if drop < min_gain:
            return n, (f"going from {n} to {n + 1} groups lowers the residual of the B-factor fit by only {100 * drop:.1f}% "
                       f"(< {100 * min_gain:g}%)")
    return K, f"the residual was still falling by {min_gain:g} or more per group at the largest number of groups evaluated"

# ---------------------------------------------------------------------------------------------------
# selection rule
def choose_n(M, min_delta=0.001, tie=0.25):
    """M[0]: R without TLS, M[N]: R with N groups per chain (lower is better).
    Returns (N, explanation); N = 0 means that TLS is not recommended."""
    K = len(M) - 1
    if K < 1:
        return 0, "no number of groups was evaluated"
    d = [None] + [M[n - 1] - M[n] for n in range(1, K + 1)]
    if d[1] < min_delta:
        return 0, f"one group per chain improves R by only {d[1]:.4f} (< min_delta {min_delta:g}): TLS is not recommended"
    if K == 1:
        return 1, "only one group per chain could be evaluated"
    dmax = max(d[2:])
    if dmax < min_delta:
        return 1, (f"no further group improves R by min_delta ({min_delta:g}); the biggest jump after one group is "
                   f"{dmax:.4f}")
    n = min(k for k in range(2, K + 1) if d[k] >= (1. - tie) * dmax)
    msg = (f"biggest jump of R after the first group: {dmax:.4f}; the lowest number of groups with a jump within "
           f"{tie:g} of it is {n} (jump {d[n]:.4f})")
    if d[n] < 3 * min_delta:
        msg += (f"\n  NOTE: this jump is small (< {3 * min_delta:g}) and the curve has no clear step, so the evidence "
                "for more than one group is weak; compare with --ngroups 1 and consider fewer groups at lower resolution")
    return n, msg

def parse_ngroups(items, chain_names):
    """'4' -> every chain; 'A:4' / 'A=4' -> per chain. Chains not mentioned (when some are) get 1 group"""
    default, per_chain = None, {}
    for it in items:
        m = re.match(r"^\s*(?:(\S+?)\s*[:=]\s*)?(\d+)\s*$", it)
        if not m or int(m.group(2)) < 1:
            raise SystemExit(f"Error: cannot interpret --ngroups {it!r}; use N or CHAIN:N (N >= 1)")
        if m.group(1) is None:
            default = int(m.group(2))
        else:
            if m.group(1) not in chain_names:
                raise SystemExit(f"Error: chain {m.group(1)!r} in --ngroups is not a polymer chain of the model "
                                 f"(polymer chains: {' '.join(chain_names)})")
            per_chain[m.group(1)] = int(m.group(2))
    base = 1 if default is None else default
    return {c: per_chain.get(c, base) for c in chain_names}

# ---------------------------------------------------------------------------------------------------
# model-dependent part
class Chain:
    def __init__(self, name, residues, blockers=()):
        self.name = name
        self.residues = residues # gemmi residues of the polymer, in order
        self.blockers = sorted(blockers) # numbers of the non-polymer residues (waters, ligands) of the chain
        self.n = len(residues)
        self.atoms = [a for r in residues for a in r]
        self.res_index = numpy.array([i for i, r in enumerate(residues) for _ in r])

    def selections(self, i, j):
        """gemmi selections for residues i .. j-1: one range, or several if a water or ligand is numbered inside it"""
        return tls_mod.range_selections(self.name, self.residues[i:j], self.blockers)

    def ranges(self, bounds):
        return [(self.residues[i].seqid, self.residues[j - 1].seqid) for i, j in zip(bounds[:-1], bounds[1:])]

def polymer_chains(st, min_residues):
    chains = []
    for ch in st[0]:
        poly = [r for r in ch.get_polymer()]
        if not poly: # no entity information (e.g. PDB file with --keep_entities): use the residue types
            for r in ch:
                info = gemmi.find_tabulated_residue(r.name)
                if info is not None and (info.is_amino_acid() or info.is_nucleic_acid()):
                    poly.append(r)
        if len(poly) == 0:
            continue
        if len(poly) < min_residues:
            logger.writeln(f"Chain {ch.name}: only {len(poly)} polymer residues; not used")
            continue
        in_polymer = {r.seqid.num for r in poly}
        chains.append(Chain(ch.name, poly, [r.seqid.num for r in ch if r.seqid.num not in in_polymer]))
    return chains

def partitions_for_chains(ctx, chains, args):
    """per-atom gradient and Fisher matrix of -LL for the anisotropic B at the current (isotropic) state, then the
    best divisions of every chain for each k. Returns {chain name: {k: (gain, bounds)}}"""
    st, ll = ctx.st, ctx.ll
    groups = [tls_mod.TlsGroup(i + 1, c.atoms) for i, c in enumerate(chains)]
    tm = tls_mod.TlsModel(st, groups, specials=ctx.geom.geom.specials, exclude_h_ll=not args.refine_h)
    ll.tls = tm
    ll.update_fc()
    llobj = ll.calc_grad(tm.params_b, specs=tm.specials, store=False)
    vn, Hb = tm._atom_blocks(llobj)
    ret = {}
    for g, c in zip(groups, chains):
        pos = g.positions()
        mass = numpy.array([a.element.weight for a in c.atoms])
        cg = ChainGain(c.res_index, pos, mass, vn[g.pidx], Hb[g.pidx], c.n)
        bounds = candidate_boundaries(c.n)
        kmax = min(args.max_groups, c.n // args.min_residues)
        t0 = time.time()
        ret[c.name] = best_partitions(cg.gain_matrix(bounds), bounds, args.min_residues, kmax)
        logger.writeln(f"Chain {c.name}: {c.n} residues, best divisions for up to {max(ret[c.name])} groups "
                       f"({time.time() - t0:.1f} s)")
    tm.release()
    ll.tls = None
    return ret

@contextlib.contextmanager
def quiet():
    logger._logger.stop_logging()
    try:
        yield
    finally:
        logger._logger.start_logging()

def write_yaml(path, groups):
    """groups: list of selection strings lists"""
    import omegaconf
    d = {"refine": {"tls": {"enforce_psd": True,
                            "groups": [{"id": i + 1, "selections": sels} for i, sels in enumerate(groups)]}}}
    omegaconf.OmegaConf.save(omegaconf.OmegaConf.create(d), path)

def write_domain_csv(path, items):
    """items: [(chain name, residue, group number)]. CSV: the TLS group of every residue, readable by any program"""
    rows = [(ch, tls_mod._seqid_str(r.seqid), r.name, int(g)) for ch, r, g in items]
    pandas.DataFrame(rows, columns=["chain", "resi", "resname", "domain"]).to_csv(path, index=False)

def write_domains_pdb(path, st, items):
    """The model (without hydrogen atoms) with the TLS group number in the B-factor column, 0 for atoms in no group,
    to colour by B in Coot or PyMOL"""
    st2 = st.clone()
    st2.remove_hydrogens()
    dom = {(ch, str(r.seqid)): int(g) for ch, r, g in items}
    for chain in st2[0]:
        for r in chain:
            v = dom.get((chain.name, str(r.seqid)), 0)
            for a in r:
                a.b_iso = float(v)
                a.aniso = gemmi.SMat33f(0, 0, 0, 0, 0, 0)
    utils.fileio.write_model(st2, file_name=path)

def run_trial(args0, yaml_path, prefix, ncycle):
    """Refinement with xyz fixed (residual B and, if yaml_path is given, TLS refined); returns the data statistics
    of the last cycle, and of cycle 0"""
    a = copy.deepcopy(args0)
    a.fix_xyz = True
    a.ncycle = ncycle
    a.output_prefix = prefix
    a.tls, a.tls_per_chain, a.tls_ncycle = False, False, None
    a.tls_file = yaml_path
    t0 = time.time()
    with quiet():
        ctx = refine_xtal.setup(a, use_tls=yaml_path is not None)
        stats = ctx.refiner.run_cycles(ncycle, weight=a.weight, weight_adjust=False)
    res = dict(stats[-1]["data"]["summary"])
    res["seconds"] = time.time() - t0
    del ctx
    return res

def available_cpus():
    """CPUs this process may use: the affinity mask, limited by a container (cgroup) CPU quota if there is one"""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError: # macOS, Windows
        n = os.cpu_count() or 1
    quota = None
    try: # cgroup v2
        with open("/sys/fs/cgroup/cpu.max") as f:
            q, per = f.read().split()[:2]
        if q != "max":
            quota = int(q) // int(per)
    except (OSError, ValueError):
        try: # cgroup v1
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
                q = int(f.read())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                per = int(f.read())
            if q > 0 and per > 0:
                quota = q // per
        except (OSError, ValueError):
            pass
    if quota is not None:
        n = min(n, max(1, quota))
    return max(1, n)

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS")

@contextlib.contextmanager
def single_threaded_libraries():
    """Worker processes are started with one thread for numerical libraries; otherwise N processes x M threads
    would oversubscribe the CPUs."""
    old = {k: os.environ.get(k) for k in _THREAD_VARS}
    for k in _THREAD_VARS:
        os.environ[k] = "1"
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v

def run_trials(args0, specs, ncycle, jobs, on_done=None):
    """Run trial refinements, `jobs` at a time. specs: [(key, yaml_path, prefix)], to be given with the most expensive
    first so that the processes finish at about the same time; a new trial is started whenever one has finished.
    on_done(key, result) is called in this process as each trial completes. Returns {key: result}."""
    results = {}
    jobs = max(1, min(jobs, len(specs)))
    if jobs == 1:
        for key, yaml_path, prefix in specs:
            results[key] = run_trial(args0, yaml_path, prefix, ncycle)
            if on_done: on_done(key, results[key])
        return results
    import concurrent.futures
    import concurrent.futures.process
    import multiprocessing
    # "spawn": clean processes that do not inherit the log file, open data or library state of this one
    mp_ctx = multiprocessing.get_context("spawn")
    with single_threaded_libraries():
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs, mp_context=mp_ctx) as ex:
            futs = {ex.submit(run_trial, args0, yaml_path, prefix, ncycle): key for key, yaml_path, prefix in specs}
            try:
                for f in concurrent.futures.as_completed(futs):
                    key = futs[f]
                    results[key] = f.result()
                    if on_done: on_done(key, results[key])
            except concurrent.futures.process.BrokenProcessPool:
                ex.shutdown(cancel_futures=True)
                raise SystemExit("Error: a trial process died (out of memory?). Try a smaller --jobs, or --jobs 1")
            except BaseException:
                ex.shutdown(cancel_futures=True)
                raise
    return results

def n_jobs(args, n_tasks):
    jobs = args.jobs if args.jobs and args.jobs > 0 else available_cpus()
    return max(1, min(jobs, n_tasks))

def r_keys(summary):
    for w, f in (("Rwork", "Rfree"), ("R1work", "R1free")):
        if f in summary:
            return w, f
    for k in ("R", "R1"):
        if k in summary:
            return k, None
    return None, None

def ascii_chart(rows, key, width=40):
    vals = [r[key] for r in rows]
    lo, hi = min(vals), max(vals)
    lines = []
    for r in rows:
        n = 1 + int(round((r[key] - lo) / (hi - lo) * (width - 1))) if hi > lo else width
        lines.append(f" {r['N']:3d} {r[key]:.4f} " + "#" * n)
    return "\n".join(lines)

def finish_bfactor_scan(args, prefix, configs, preds, ngroups_n, descs, natoms_total, t_start, config_for_n, show_ranges,
                        write_pdb):
    """Table, graph and recommendation of the B-factor fit alone (no refinement)"""
    rows = []
    for n, _ in configs:
        chi2 = max(-preds[n], 0.)
        rows.append({"N": n, "groups": ngroups_n[n], "params": tls_mod.NPAR * ngroups_n[n], "B_chi2": chi2,
                     "B_rms": float(numpy.sqrt(chi2 / natoms_total))})
    for i, r in enumerate(rows):
        r["drop_pct"] = 100. * (rows[i - 1]["B_chi2"] - r["B_chi2"]) / rows[i - 1]["B_chi2"] if i > 0 and rows[i - 1]["B_chi2"] > 0 else float("nan")
    best, why = choose_n_elbow([r["B_chi2"] for r in rows], args.min_gain)
    for r in rows:
        r["selected"] = r["N"] == best
    df = pandas.DataFrame(rows)
    df.to_csv(prefix + "_scan.csv", index=False)
    with open(prefix + "_scan.json", "w") as ofs:
        json.dump({"rows": rows, "selected_N": best, "reason": why, "metric": "B_rms",
                   "partitions": {str(n): {c: [[tls_mod._seqid_str(a), tls_mod._seqid_str(b)] for a, b in r]
                                           for c, r in d.items()} for n, d in descs.items()}}, ofs, indent=2, default=float)
    logger.writeln("\n" + "=" * 70)
    logger.writeln(f"Fit of TLS to the isotropic B-factors ({args.atoms} atoms) vs the number of groups per chain")
    logger.writeln("B_rms: r.m.s. of B_obs - B_TLS in A^2; drop: decrease of the residual from N-1 to N groups")
    logger.writeln(df[["N", "groups", "params", "B_rms", "B_chi2", "drop_pct", "selected"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}" if abs(x) < 100 else f"{x:.1f}"))
    logger.writeln("\nB_rms vs N:")
    logger.writeln(ascii_chart(rows, "B_rms"))
    try: # optional
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(df.N, df.B_rms, "o-")
        ax.axvline(best, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("TLS groups per chain")
        ax.set_ylabel("r.m.s. of B(obs) - B(TLS) (A$^2$)")
        fig.tight_layout()
        fig.savefig(prefix + "_scan.png", dpi=150)
        plt.close(fig)
        logger.writeln(f"Plot written: {prefix}_scan.png")
    except ImportError:
        pass
    logger.writeln("\n" + "=" * 70)
    groups, desc, _ = config_for_n(best)
    logger.writeln(f"Recommendation: {best} group(s) per chain. {why}")
    logger.writeln("  NOTE: a fit to B-factors always improves with more groups. This is a rule of thumb, not a test:"
                   "\n  give --hklin to compare R-free for each number of groups")
    show_ranges(desc)
    write_yaml(prefix + "_tls.yaml", groups)
    write_pdb(best)
    logger.writeln(f"\nTLS definition written: {prefix}_tls.yaml\n  use it with: refine_xtal --tls_file {prefix}_tls.yaml\n"
                   f"  (other numbers of groups: {prefix}_tls_N*.yaml; the number can be forced with --ngroups)\n"
                   f"Group of every residue: {prefix}_domains_N*.csv; model with the group number as B-factor: {prefix}_domains.pdb")
    logger.writeln(f"Table: {prefix}_scan.csv, {prefix}_scan.json    total time {time.time() - t_start:.0f} s")

def main(args):
    set_prefix(args)
    method = getattr(args, "method", "likelihood")
    if method == "bfactor" and not args.hklin:
        args.no_eval = True # nothing to refine against
    data_free = method == "bfactor" and args.no_eval
    args0 = copy.deepcopy(args)
    if data_free:
        logger.writeln("Reading the model..")
        st = utils.fileio.read_structure(args.model)
        st.setup_entities()
    else:
        logger.writeln("Preparing the model and data..")
        ctx = refine_xtal.setup(args, use_tls=False)
        st = ctx.st
        if method == "likelihood":
            ll = ctx.ll
            ll.update_fc()
            ll.overall_scale()
            ll.update_ml_params()
            ll.prepare_target()
    chains = polymer_chains(st, args.min_residues)
    if not chains:
        raise SystemExit("Error: no polymer chain with enough residues")
    logger.writeln("Polymer chains: " + ", ".join(f"{c.name} ({c.n} residues)" for c in chains) + "\n")
    if method == "bfactor":
        parts = partitions_by_bfactor(chains, args)
        natoms_total = sum(args._bfit_atoms.values())
    else:
        parts = partitions_for_chains(ctx, chains, args)
    prefix = args.output_prefix
    trial_dir = prefix + "_trials" # files written by the trial refinements
    if not data_free:
        os.makedirs(trial_dir, exist_ok=True)
    by_name = {c.name: c for c in chains}

    def resolve(name, k): # the division of a chain into k groups (fewer if k is not possible)
        avail = parts[name]
        if k not in avail:
            kk = max(x for x in avail if x <= k) if any(x <= k for x in avail) else None
            if kk is None:
                raise SystemExit(f"Error: chain {name} cannot be divided into {k} groups of at least "
                                 f"{args.min_residues} residues")
            logger.writeln(f"WARNING: chain {name}: {k} groups not possible with min_residues= {args.min_residues}; using {kk}")
            k = kk
        return avail[k]

    def config_for(counts): # counts: chain name -> requested number of groups
        groups, desc, pred = [], {}, 0.
        for name, k in counts.items():
            c = by_name[name]
            gain, bounds = resolve(name, k)
            pred += gain
            desc[name] = c.ranges(bounds)
            groups += [c.selections(i, j) for i, j in zip(bounds[:-1], bounds[1:])]
        return groups, desc, pred

    def items_for(counts): # (chain, residue, group number) for every residue; groups are numbered as in the yaml
        items, g = [], 0
        for name, k in counts.items():
            c = by_name[name]
            _, bounds = resolve(name, k)
            for i, j in zip(bounds[:-1], bounds[1:]):
                g += 1
                items += [(name, c.residues[x], g) for x in range(i, j)]
        return items

    def show_ranges(desc):
        for name, rngs in desc.items():
            logger.writeln(f"  chain {name}: " + ", ".join(f"{tls_mod._seqid_str(a)}-{tls_mod._seqid_str(b)}" for a, b in rngs))

    rows = []
    t_start = time.time()
    if args.ngroups:
        counts = parse_ngroups(args.ngroups, [c.name for c in chains])
        groups, desc, pred = config_for(counts)
        logger.writeln(f"\nTLS groups for {' '.join(f'{k}:{v}' for k, v in counts.items())} (groups per chain):")
        show_ranges(desc)
        out_yaml = prefix + "_tls.yaml"
        write_yaml(out_yaml, groups)
        items = items_for(counts)
        write_domain_csv(prefix + "_domains.csv", items)
        write_domains_pdb(prefix + "_domains.pdb", st, items)
        logger.writeln(f"\nTLS definition written: {out_yaml}\n  use it with: refine_xtal --tls_file {out_yaml}\n"
                       f"Group of every residue: {prefix}_domains.csv; model with the group number as B-factor: {prefix}_domains.pdb")
        if not args.no_eval:
            jobs = n_jobs(args, 2)
            logger.writeln(f"\nEvaluating by refinement (xyz fixed, {args.tls_cycles} cycles){f' on {jobs} processes' if jobs > 1 else ''}..")
            res = run_trials(args0, [("custom", out_yaml, f"{trial_dir}/custom"), ("n0", None, f"{trial_dir}/n0")],
                             args.tls_cycles, jobs)
            r0, r1 = res["n0"], res["custom"]
            kw, kf = r_keys(r1)
            for lab, r in (("no TLS", r0), (f"{len(groups)} groups", r1)):
                logger.writeln(f" {lab:12s} " + "  ".join(f"{k}= {r[k]:.4f}" for k in (kw, kf) if k) + f"  -LL= {r['-LL']:.1f}")
        return
    # scan
    kmax_chain = {name: max(p) for name, p in parts.items()}
    n_max = min(args.max_groups, max(kmax_chain.values()))
    configs, prev = [], None
    for n in range(1, n_max + 1):
        counts = {name: min(n, kmax_chain[name]) for name in kmax_chain}
        if counts == prev:
            break
        configs.append((n, counts))
        prev = counts
    if method == "bfactor" and args.no_eval:
        logger.writeln(f"\nScanning 1 .. {configs[-1][0]} groups per chain (fit of the B-factors only, no refinement)")
    else:
        logger.writeln(f"\nScanning 0 .. {configs[-1][0]} groups per chain; each evaluated by {args.tls_cycles} cycles of refinement "
                       "(xyz fixed, residual B + TLS)")
    descs, preds, ngroups_n, specs = {}, {}, {}, []
    for n, counts in configs:
        groups, desc, pred = config_for(counts)
        yaml_n = f"{prefix}_tls_N{n}.yaml"
        write_yaml(yaml_n, groups)
        write_domain_csv(f"{prefix}_domains_N{n}.csv", items_for(counts))
        descs[n], preds[n], ngroups_n[n] = desc, pred, len(groups)
        specs.append((n, yaml_n, f"{trial_dir}/n{n}"))
    if method == "bfactor" and args.no_eval:
        finish_bfactor_scan(args, prefix, configs, preds, ngroups_n, descs, natoms_total, t_start,
                            lambda n: config_for(dict(configs)[n]), show_ranges,
                            lambda n: write_domains_pdb(prefix + "_domains.pdb", st, items_for(dict(configs)[n])))
        return
    specs.reverse()                                   # most groups (most expensive) first
    specs.append((0, None, f"{trial_dir}/n0"))        # cheapest last
    jobs = n_jobs(args, len(specs))
    logger.writeln(f"Running {len(specs)} trial refinements" +
                   (f" in parallel on {jobs} processes ({available_cpus()} CPUs available; the next one starts when one finishes)"
                    if jobs > 1 else " one after another"))
    def report(n, r): # called as each trial finishes (not in the order of N)
        kw, kf = r_keys(r)
        lab = "N= 0 (no TLS)" if n == 0 else f"N={n:2d} ({ngroups_n[n]:2d} groups)"
        logger.writeln(f"  {lab}  {kw}= {r[kw]:.4f}" + (f"  {kf}= {r[kf]:.4f}" if kf else "") + f"  [{r['seconds']:.0f} s]")
    t_trials = time.time()
    res = run_trials(args0, specs, args.tls_cycles, jobs, report)
    wall = time.time() - t_trials
    work = sum(r["seconds"] for r in res.values())
    if jobs > 1:
        logger.writeln(f"  {len(specs)} trials took {wall:.0f} s ({work:.0f} s of work, {work / max(wall, 1e-9):.1f}x faster than one after another)")
    kw, kf = r_keys(res[0])
    if kf is None:
        logger.writeln("NOTE: no test set; R (all data) is used for the selection")
    key_sel = kf or kw
    rows.append({"N": 0, "groups": 0, "params": 0, "dp_score": 0., **{k: v for k, v in res[0].items()}})
    for n, counts in configs:
        rows.append({"N": n, "groups": ngroups_n[n], "params": tls_mod.NPAR * ngroups_n[n], "dp_score": preds[n], **res[n]})
        if method == "bfactor":
            rows[-1]["B_rms"] = float(numpy.sqrt(max(-preds[n], 0.) / natoms_total))
    M = [r[key_sel] for r in rows]
    best, why = choose_n(M, args.min_delta, args.tie)
    for r in rows:
        r["d_" + key_sel] = (M[r["N"] - 1] - r[key_sel]) if r["N"] > 0 else float("nan")
        r["selected"] = r["N"] == best
    df = pandas.DataFrame(rows)
    cols = ["N", "groups", "params", kw] + ([kf] if kf else []) + ["d_" + key_sel, "-LL", "dp_score"]
    if method == "bfactor":
        cols.insert(3, "B_rms")
    df.to_csv(prefix + "_scan.csv", index=False, columns=cols + ["selected"] + ["seconds"])
    with open(prefix + "_scan.json", "w") as ofs:
        json.dump({"rows": rows, "selected_N": best, "reason": why, "metric": key_sel,
                   "partitions": {str(n): {c: [[tls_mod._seqid_str(a), tls_mod._seqid_str(b)] for a, b in r]
                                           for c, r in d.items()} for n, d in descs.items()}},
                  ofs, indent=2, default=float)
    logger.writeln("\n" + "=" * 70)
    logger.writeln(f"R-values vs the number of TLS groups per chain (refinement of {args.tls_cycles} cycles, xyz fixed)")
    logger.writeln(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}" if abs(x) < 100 else f"{x:.1f}"))
    logger.writeln(f"\n{key_sel} vs N:")
    logger.writeln(ascii_chart(rows, key_sel))
    forplot = [["R vs groups per chain", ["N", kw] + ([kf] if kf else [])]]
    logger.writeln(utils.make_loggraph_str(df[["N", kw] + ([kf] if kf else [])], "R vs groups per chain", forplot,
                                           float_format="{:.4f}".format))
    try: # optional
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(df.N, df[kw], "o-", label=kw)
        if kf:
            ax.plot(df.N, df[kf], "s-", label=kf)
        if best > 0:
            ax.axvline(best, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("TLS groups per chain")
        ax.set_ylabel("R")
        ax.legend()
        fig.tight_layout()
        fig.savefig(prefix + "_scan.png", dpi=150)
        logger.writeln(f"Plot written: {prefix}_scan.png")
    except ImportError:
        pass
    logger.writeln("\n" + "=" * 70)
    if best == 0:
        logger.writeln(f"Recommendation: no TLS. {why}")
    else:
        groups, desc, _ = config_for({c: min(best, kmax_chain[c]) for c in kmax_chain})
        logger.writeln(f"Recommendation: {best} group(s) per chain. {why}")
        show_ranges(desc)
        write_yaml(prefix + "_tls.yaml", groups)
        write_domains_pdb(prefix + "_domains.pdb", st, items_for({c: min(best, kmax_chain[c]) for c in kmax_chain}))
        logger.writeln(f"\nTLS definition written: {prefix}_tls.yaml\n  use it with: refine_xtal --tls_file {prefix}_tls.yaml\n"
                       f"  (other numbers of groups: {prefix}_tls_N*.yaml; the number can be forced with --ngroups)\n"
                       f"Group of every residue: {prefix}_domains_N*.csv; model with the group number as B-factor: {prefix}_domains.pdb")
    logger.writeln(f"Table: {prefix}_scan.csv, {prefix}_scan.json    total time {time.time() - t_start:.0f} s")
# main()
