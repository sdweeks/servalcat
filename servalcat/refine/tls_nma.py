"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.

TLS groups from the collective motions of the structure (servalcat util tls_nma)

An elastic network model of the polymer residues (one node per residue: C-alpha, or C4' in nucleic acids; springs
between nodes closer than a cutoff) gives its low-frequency normal modes. Two models are available:

* anm: anisotropic network model (Atilgan et al. 2001), a 3N x 3N Hessian, cutoff 15 A; the modes are displacement
  vectors. Two residues belong to the same rigid body if the distance between them hardly changes in the modes
  (Hinsen et al. 1999): the variance of that distance is the dissimilarity.
* gnm: Gaussian network model (Bahar et al. 1997), an N x N Kirchhoff matrix, cutoff 10 A; the modes are
  fluctuation amplitudes. Residues that move together have correlation +1 in the slow modes: 1 - correlation is
  the dissimilarity.

Residues are clustered into k domains by spectral clustering (normalised cut) of the network graph weighted by
exp(-(dissimilarity/sigma)^2). Because the graph is the contact network, the domains are compact in space, but they
need not be contiguous in sequence (an internal domain, a domain made of two stretches, a domain of two chains). Small
fragments are merged into their neighbours. Every domain becomes one TLS group with one selection per stretch of
residues. Both models are implemented here with scipy (sparse matrices, shift-invert for the lowest modes), after the
definitions that ProDy uses (the eigenvalues are identical).
"""
from __future__ import absolute_import, division, print_function, generators
import argparse
import copy
import json
import os
import time
import numpy
import pandas
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg
import scipy.spatial
from scipy.cluster.vq import kmeans2
import gemmi
from servalcat.utils import logger
from servalcat import utils
from servalcat.refine import refine_xtal
from servalcat.refine import tls as tls_mod
from servalcat.refine import tls_select as ts

ENM_CUTOFF = {"anm": 15.0, "gnm": 10.0}
DENSE_LIMIT = 1800      # matrices up to this size are diagonalised completely (faster and safer)

# ---------------------------------------------------------------------------------------------------
# nodes
class Node:
    __slots__ = ("chain", "residue", "pos", "index")
    def __init__(self, chain, residue, pos, index):
        self.chain, self.residue, self.pos, self.index = chain, residue, pos, index

def collect_nodes(st):
    """One node per polymer residue of the first model: C-alpha of amino acids, C4' of nucleotides. Residues are
    recognised by name (not by entity or connectivity), waters and ligands are ignored.
    Returns the list of Node and {chain name: [residues of the chain that are nodes, in order]}"""
    nodes, by_chain = [], {}
    for ch in st[0]:
        for r in ch:
            info = gemmi.find_tabulated_residue(r.name)
            if info is None or r.is_water():
                continue
            if info.is_amino_acid():
                a = r.find_atom("CA", "*")
            elif info.is_nucleic_acid():
                a = r.find_atom("C4'", "*") or r.find_atom("P", "*")
            else:
                continue
            if a is None:
                continue
            nodes.append(Node(ch.name, r, numpy.array([a.pos.x, a.pos.y, a.pos.z]), len(nodes)))
            by_chain.setdefault(ch.name, []).append(r)
    return nodes, by_chain

# ---------------------------------------------------------------------------------------------------
# elastic network models
def contact_edges(pos, cutoff):
    """pairs (i < j) of nodes closer than cutoff, as an (E, 2) array"""
    pairs = scipy.spatial.cKDTree(pos).query_pairs(cutoff, output_type="ndarray")
    return pairs.reshape(-1, 2)

def anm_hessian(pos, edges, gamma=1.0):
    """3N x 3N Hessian of the anisotropic network model, as a sparse matrix"""
    n = len(pos)
    i, j = edges[:, 0], edges[:, 1]
    d = pos[j] - pos[i]
    b = -gamma * d[:, :, None] * d[:, None, :] / (d ** 2).sum(1)[:, None, None]   # (E, 3, 3) super-element, i != j
    ar = numpy.arange(3)
    rows_ij = (3 * i[:, None, None] + ar[None, :, None] + 0 * ar[None, None, :])
    cols_ij = (3 * j[:, None, None] + 0 * ar[None, :, None] + ar[None, None, :])
    rows = [rows_ij.ravel(), cols_ij.ravel()]
    cols = [cols_ij.ravel(), rows_ij.ravel()]
    vals = [b.ravel(), b.ravel()]
    H = scipy.sparse.coo_matrix((numpy.concatenate(vals), (numpy.concatenate(rows), numpy.concatenate(cols))),
                                shape=(3 * n, 3 * n)).tocsr()
    # diagonal blocks: minus the sum of the off-diagonal blocks of the node
    diag = numpy.zeros((n, 3, 3))
    numpy.add.at(diag, i, -b)
    numpy.add.at(diag, j, -b)
    rows_d = (3 * numpy.arange(n)[:, None, None] + ar[None, :, None] + 0 * ar[None, None, :]).ravel()
    cols_d = (3 * numpy.arange(n)[:, None, None] + 0 * ar[None, :, None] + ar[None, None, :]).ravel()
    return (H + scipy.sparse.coo_matrix((diag.ravel(), (rows_d, cols_d)), shape=(3 * n, 3 * n)).tocsr()).tocsr()

def gnm_kirchhoff(n, edges, gamma=1.0):
    """N x N Kirchhoff (connectivity) matrix of the Gaussian network model, as a sparse matrix"""
    i, j = edges[:, 0], edges[:, 1]
    A = scipy.sparse.coo_matrix((numpy.full(len(i), -gamma), (i, j)), shape=(n, n))
    A = (A + A.T).tocsr()
    deg = -numpy.asarray(A.sum(axis=1)).ravel()
    return (A + scipy.sparse.diags(deg)).tocsr()

def lowest_modes(M, nmodes, n_zero, tol=1e-6):
    """The nmodes lowest non-zero eigenvalues and eigenvectors of the symmetric positive semi-definite matrix M.
    n_zero: number of zero modes expected (rigid-body motions of every connected component)."""
    n = M.shape[0]
    if n <= DENSE_LIMIT:
        w, V = numpy.linalg.eigh(M.toarray())
    else:
        k = min(nmodes + n_zero + 4, n - 2)
        w, V = scipy.sparse.linalg.eigsh(M, k=k, sigma=-1e-3, which="LM")
        order = numpy.argsort(w)
        w, V = w[order], V[:, order]
    keep = numpy.where(w > tol)[0][:nmodes]
    return w[keep], V[:, keep]

def n_components(n, edges):
    g = scipy.sparse.coo_matrix((numpy.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    return scipy.sparse.csgraph.connected_components(g, directed=False)[0]

def network_modes(pos, enm, cutoff, nmodes, gamma=1.0):
    """Returns (edges, eigenvalues, eigenvectors): for anm vectors are (3N, m), for gnm (N, m)"""
    edges = contact_edges(pos, cutoff)
    nc = n_components(len(pos), edges)
    if enm == "anm":
        w, V = lowest_modes(anm_hessian(pos, edges, gamma), nmodes, 6 * nc)
    elif enm == "gnm":
        w, V = lowest_modes(gnm_kirchhoff(len(pos), edges, gamma), nmodes, nc)
    else:
        raise ValueError(enm)
    return edges, w, V

def edge_dissimilarity(pos, edges, enm, w, V):
    """Dissimilarity of the two nodes of every edge: 0 = move together as a rigid body"""
    wt = 1.0 / w
    wt = wt / wt.sum()
    i, j = edges[:, 0], edges[:, 1]
    if enm == "anm":
        d = pos[j] - pos[i]
        e = d / numpy.linalg.norm(d, axis=1, keepdims=True)
        var = numpy.zeros(len(edges))
        for k in range(V.shape[1]):
            u = V[:, k].reshape(-1, 3)
            var += wt[k] * (((u[j] - u[i]) * e).sum(1)) ** 2   # variance of the distance i-j from mode k
        return numpy.sqrt(var)
    # gnm: 1 - normalised correlation in the slow modes
    corr = (V[i] * V[j] * wt).sum(1)
    norm = numpy.sqrt((V[i] ** 2 * wt).sum(1) * (V[j] ** 2 * wt).sum(1))
    return numpy.sqrt(numpy.maximum(1.0 - corr / numpy.maximum(norm, 1e-30), 0.0))

# ---------------------------------------------------------------------------------------------------
# clustering
def graph_affinity(n, edges, dist):
    sigma = max(float(numpy.median(dist)), 1e-9)
    a = numpy.exp(-(dist / sigma) ** 2)
    A = scipy.sparse.coo_matrix((a, (edges[:, 0], edges[:, 1])), shape=(n, n))
    return (A + A.T).tocsr()

def spectral_labels(A, k, restarts=10, seed=1):
    """Normalised-cut clustering of the weighted graph A into k groups"""
    n = A.shape[0]
    if k <= 1:
        return numpy.zeros(n, dtype=int)
    d = numpy.asarray(A.sum(axis=1)).ravel()
    dinv = 1.0 / numpy.sqrt(numpy.maximum(d, 1e-12))
    S = scipy.sparse.diags(dinv) @ A @ scipy.sparse.diags(dinv)
    if n <= DENSE_LIMIT:
        w, V = numpy.linalg.eigh(S.toarray())
        X = V[:, -k:]
    else:
        w, X = scipy.sparse.linalg.eigsh(S, k=k, which="LA")
    X = X / numpy.maximum(numpy.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    best = None
    for r in range(restarts):
        c, lab = kmeans2(X, k, minit="++", seed=seed + r)
        cost = float(((X - c[lab]) ** 2).sum())
        if best is None or cost < best[0]:
            best = (cost, lab)
    return best[1].astype(int)

def clean_labels(labels, A, min_fragment, min_group):
    """Merge fragments into the neighbouring group with which they share most contacts: connected pieces of a group
    with fewer than min_fragment nodes, then whole groups with fewer than min_group nodes. Groups are renumbered 0..g-1
    by their smallest node."""
    labels = labels.copy()
    n = len(labels)
    A = A.tocsr()

    def merge(nodes_, own):
        """label of the other group with the largest summed affinity to the nodes; None if there is none"""
        coo = A[nodes_].tocoo()
        weights = {}
        for c, v in zip(coo.col, coo.data):
            if labels[c] != own:
                weights[labels[c]] = weights.get(labels[c], 0.) + v
        return max(weights, key=weights.get) if weights else None

    for _ in range(50):
        changed = False
        for lab in numpy.unique(labels):
            idx = numpy.where(labels == lab)[0]
            g = scipy.sparse.csgraph.connected_components(A[idx][:, idx], directed=False)[1]
            for comp in numpy.unique(g):
                nodes = idx[g == comp]
                if len(nodes) < min_fragment:
                    target = merge(nodes, lab)
                    if target is not None:
                        labels[nodes] = target
                        changed = True
        if not changed:
            break
    while True:
        uniq, counts = numpy.unique(labels, return_counts=True)
        if len(uniq) <= 1 or counts.min() >= min_group:
            break
        small = uniq[counts.argmin()]
        nodes = numpy.where(labels == small)[0]
        target = merge(nodes, small)
        if target is None:
            break
        labels[nodes] = target
    order = {lab: i for i, lab in enumerate(sorted(set(labels), key=lambda l: numpy.where(labels == l)[0][0]))}
    return numpy.array([order[l] for l in labels])

def smooth_stretches(labels, nodes, A, min_stretch):
    """A run of fewer than min_stretch consecutive residues of a chain that belongs to another group than its
    neighbours in the chain is given to the group of those neighbours (if they differ: to the one it touches more)."""
    labels = labels.copy()
    chains = {}
    for n in nodes:
        chains.setdefault(n.chain, []).append(n.index)
    for _ in range(10):
        changed = False
        for idx in chains.values():
            runs, start = [], 0
            for i in range(1, len(idx) + 1):
                if i == len(idx) or labels[idx[i]] != labels[idx[start]]:
                    runs.append((start, i))
                    start = i
            for r, (a, b) in enumerate(runs):
                if b - a >= min_stretch or len(runs) < 2:
                    continue
                left = labels[idx[runs[r - 1][0]]] if r > 0 else None
                right = labels[idx[runs[r + 1][0]]] if r + 1 < len(runs) else None
                if left is not None and right is not None and left != right:
                    members = numpy.array(idx[a:b])
                    sub = A[members]
                    wl = sub[:, numpy.where(labels == left)[0]].sum()
                    wr = sub[:, numpy.where(labels == right)[0]].sum()
                    target = left if wl >= wr else right
                else:
                    target = left if left is not None else right
                if target is not None and target != labels[idx[a]]:
                    labels[idx[a:b]] = target
                    changed = True
        if not changed:
            break
    return labels

# ---------------------------------------------------------------------------------------------------
# groups
def stretches(nodes, labels, label, by_chain):
    """selections (gemmi) for the residues of one group: one per stretch of consecutive residues of a chain"""
    in_group = {id(n.residue) for n, l in zip(nodes, labels) if l == label}
    sels = []
    for chain, residues in by_chain.items():
        start = prev = None
        for r in residues:
            if id(r) in in_group:
                if start is None:
                    start = r
                prev = r
            elif start is not None:
                sels.append((chain, start, prev))
                start = None
        if start is not None:
            sels.append((chain, start, prev))
    out = []
    for chain, a, b in sels:
        sa, sb = tls_mod._seqid_str(a.seqid), tls_mod._seqid_str(b.seqid)
        out.append(f"//{chain}/{sa}/*" if sa == sb else f"//{chain}/{sa}-{sb}/*")
    return out

def groups_from_labels(nodes, labels, by_chain):
    return [stretches(nodes, labels, lab, by_chain) for lab in range(int(labels.max()) + 1)]

def group_bfit_chi2(nodes, labels, atoms_choice="all"):
    """Total residual of fitting a TLS model to the isotropic B-factors of every group; returns (chi2, number of atoms)"""
    names = ts.ATOM_SETS[atoms_choice]
    total, natoms = 0., 0
    for lab in range(int(labels.max()) + 1):
        pos, b = [], []
        for n, l in zip(nodes, labels):
            if l != lab:
                continue
            for a in n.residue:
                if a.is_hydrogen() or a.occ <= 0 or (names is not None and a.name not in names):
                    continue
                pos.append([a.pos.x, a.pos.y, a.pos.z])
                b.append(a.b_iso)
        if len(b) < 25:      # too few atoms to determine the ~10 parameters of an isotropic B
            continue
        total += ts.bfit_residual(numpy.array(pos), numpy.array(b))
        natoms += len(b)
    return total, natoms

def partitions(nodes, by_chain, args, edges, enm, w, V):
    """{k: (labels, groups)} for k = 1 .. max_groups domains (a number is left out if the cleaning gave the
    same partition as a smaller one)"""
    pos = numpy.array([n.pos for n in nodes])
    dist = edge_dissimilarity(pos, edges, enm, w, V)
    A = graph_affinity(len(nodes), edges, dist)
    ret, seen = {}, set()
    for k in range(1, args.max_groups + 1):
        labels = clean_labels(spectral_labels(A, k), A, args.min_fragment, args.min_residues)
        if getattr(args, "min_stretch", 0) > 1:
            labels = clean_labels(smooth_stretches(labels, nodes, A, args.min_stretch), A, args.min_fragment, args.min_residues)
        key = tuple(labels)
        if key in seen:
            continue
        seen.add(key)
        ret[k] = (labels, groups_from_labels(nodes, labels, by_chain))
    return ret

# ---------------------------------------------------------------------------------------------------
# command
def add_arguments(parser):
    def own_options(parser):
        add = utils.cli.add_arg
        g = parser.add_argument_group("Normal-mode domains: optional arguments with default values")
        add(g, "--enm", choices=["anm", "gnm"], default="anm",
            help="Elastic network model: anm: anisotropic (3D displacement vectors); gnm: Gaussian (fluctuation amplitudes)")
        add(g, "--cutoff", type=float, metavar="A", show="15 for anm, 10 for gnm",
            help="Residues closer than this (Angstrom, C-alpha / C4') are connected by a spring")
        add(g, "--nmodes", type=int, default=10, metavar="N", help="Number of lowest-frequency modes used")
        add(g, "--max_groups", type=int, default=6, metavar="N", help="Largest number of domains (TLS groups)")
        add(g, "--min_residues", type=int, default=20, metavar="N", help="Smallest number of residues in a domain")
        add(g, "--min_fragment", type=int, default=5, metavar="N",
            help="A piece of a domain that is separate in space and smaller than this is given to its neighbour")
        add(g, "--min_stretch", type=int, default=3, metavar="N",
            help="A stretch of fewer than this many consecutive residues inside another domain is given to that domain "
            "(1: keep every stretch)")
        add(g, "--tls_cycles", type=int, default=8, metavar="N",
            help="Cycles of the refinement (coordinates fixed) that evaluates each number of domains, with --hklin")
        add(g, "--min_delta", type=float, default=0.001, metavar="X",
            help="With --hklin: smallest decrease of R that counts as an improvement")
        add(g, "--tie", type=float, default=0.25, metavar="X",
            help="With --hklin: jumps within this fraction of the biggest one are equivalent; the fewest domains win")
        add(g, "--min_gain", type=float, default=0.05, metavar="X",
            help="Without --hklin: the smallest number of domains for which one more lowers the residual of the "
            "B-factor fit by less than this fraction")
        add(g, "--atoms", choices=list(ts.ATOM_SETS), default="all",
            help="Atoms whose B-factors are fitted to compare the partitions")
        add(g, "-j", "--jobs", type=int, default=0, metavar="N", show="0 = all available CPUs",
            help="Number of trial refinements run in parallel, with --hklin")
        g = parser.add_argument_group("Normal-mode domains: options (off unless given)")
        add(g, "--ngroups", type=int, metavar="N", show="none: scan 1 .. --max_groups",
            help="Use exactly this number of domains")
        add(g, "--no_eval", action="store_true", help="Do not refine to get R-values (always so without --hklin)")
    refine_xtal.add_arguments(parser, after_required=own_options, hklin_required=False)
    import textwrap
    parser.usage = "{} --model FILE [--hklin FILE] [options]".format(parser.prog)
    parser.description = textwrap.fill(
        "Find domains that move together from the low-frequency normal modes of an elastic network model of the "
        "polymer (anisotropic or Gaussian) and write them as TLS groups. A domain may consist of several stretches of "
        "residues, or of several chains. Without --hklin only the model is read. With --hklin every number of domains "
        "is also tested by refinement and R-values are reported, as 'util tls_select' does. The result is written as "
        "yaml for 'refine_xtal --tls_file' and as a PDB file with the domain number in the B-factor column.", 78)
    parser.epilog = "Example: {} --model refined.pdb --enm gnm -o tlsnma".format(parser.prog)
    for a in parser._actions:
        if a.dest in ts._HIDDEN:
            a.help = argparse.SUPPRESS

def set_prefix(args):
    if not args.output_prefix:
        args.output_prefix = utils.fileio.splitext(os.path.basename(args.model))[0] + "_tlsnma"

def write_domain_files(prefix, n, nodes, labels, st):
    """CSV with the domain of every residue (chain, residue number, name, domain), for use by other programs"""
    rows = [(nd.chain, tls_mod._seqid_str(nd.residue.seqid), nd.residue.name, int(l) + 1) for nd, l in zip(nodes, labels)]
    pandas.DataFrame(rows, columns=["chain", "resi", "resname", "domain"]).to_csv(f"{prefix}_domains_N{n}.csv", index=False)

def write_domains_pdb(path, st, nodes, labels):
    """the model with the domain number in the B-factor column (0 for atoms outside the network), to colour by B"""
    st2 = st.clone()
    dom = {(nd.chain, str(nd.residue.seqid)): int(l) + 1 for nd, l in zip(nodes, labels)}
    for ch in st2[0]:
        for r in ch:
            v = dom.get((ch.name, str(r.seqid)), 0)
            for a in r:
                a.b_iso = float(v)
                a.aniso = gemmi.SMat33f(0, 0, 0, 0, 0, 0)
    utils.fileio.write_model(st2, file_name=path)

def main(args):
    set_prefix(args)
    if not args.hklin:
        args.no_eval = True
    data_free = args.no_eval
    args0 = copy.deepcopy(args)
    if data_free:
        logger.writeln("Reading the model..")
        st = utils.fileio.read_structure(args.model)
        st.setup_entities()
    else:
        logger.writeln("Preparing the model and data..")
        ctx = refine_xtal.setup(args, use_tls=False)
        st = ctx.st
    prefix = args.output_prefix
    nodes, by_chain = collect_nodes(st)
    if len(nodes) < 2 * args.min_residues:
        raise SystemExit(f"Error: only {len(nodes)} polymer residues; at least {2 * args.min_residues} are needed")
    logger.writeln("Polymer chains: " + ", ".join(f"{c} ({len(r)} residues)" for c, r in by_chain.items()) + "\n")
    cutoff = args.cutoff if args.cutoff else ENM_CUTOFF[args.enm]
    pos = numpy.array([n.pos for n in nodes])
    t0 = time.time()
    edges, w, V = network_modes(pos, args.enm, cutoff, args.nmodes)
    nc = n_components(len(nodes), edges)
    logger.writeln(f"{args.enm.upper()}: {len(nodes)} nodes, {len(edges)} springs (cutoff {cutoff:g} A), "
                   f"{len(w)} lowest modes, eigenvalues {w[0]:.3f} .. {w[-1]:.3f} ({time.time() - t0:.1f} s)")
    if nc > 1:
        logger.writeln(f"NOTE: the network has {nc} separate parts (chains or domains farther apart than the cutoff); "
                       "they move independently")
    parts = partitions(nodes, by_chain, args, edges, args.enm, w, V)
    if args.ngroups:
        if args.ngroups not in parts:
            raise SystemExit(f"Error: no partition into {args.ngroups} domains of at least {args.min_residues} residues "
                             f"(possible: {', '.join(map(str, sorted(parts)))})")
        parts = {args.ngroups: parts[args.ngroups]}
    t_start = time.time()
    ks = sorted(parts)
    rows, specs = [], []
    for k in ks:
        labels, groups = parts[k]
        yaml_k = f"{prefix}_tls_N{k}.yaml"
        ts.write_yaml(yaml_k, groups)
        write_domain_files(prefix, k, nodes, labels, st)
        chi2, na = group_bfit_chi2(nodes, labels, args.atoms)
        sizes = sorted([int((labels == g).sum()) for g in range(len(groups))], reverse=True)
        rows.append({"N": k, "groups": len(groups), "params": tls_mod.NPAR * len(groups), "sizes": "/".join(map(str, sizes)),
                     "stretches": sum(len(g) for g in groups), "B_chi2": chi2,
                     "B_rms": float(numpy.sqrt(chi2 / na)) if na else float("nan")})
        specs.append((k, yaml_k, f"{prefix}_trials/n{k}"))
    logger.writeln(f"\nDomains for 1 .. {ks[-1]}: " + "; ".join(f"{r['N']}: sizes {r['sizes']}" for r in rows))

    kw = kf = None
    if not data_free:
        os.makedirs(prefix + "_trials", exist_ok=True)
        specs.reverse()
        specs.append((0, None, f"{prefix}_trials/n0"))
        jobs = ts.n_jobs(args, len(specs))
        logger.writeln(f"\nRunning {len(specs)} trial refinements ({args.tls_cycles} cycles, xyz fixed)" +
                       (f" in parallel on {jobs} processes" if jobs > 1 else " one after another"))
        sizes_by_k = {r["N"]: r for r in rows}
        def report(n, r):
            a, b = ts.r_keys(r)
            lab = "no TLS" if n == 0 else f"{sizes_by_k[n]['groups']} domains"
            logger.writeln(f"  {lab:12s} {a}= {r[a]:.4f}" + (f"  {b}= {r[b]:.4f}" if b else "") + f"  [{r['seconds']:.0f} s]")
        res = ts.run_trials(args0, specs, args.tls_cycles, jobs, report)
        kw, kf = ts.r_keys(res[0])
        key_sel = kf or kw
        rows.insert(0, {"N": 0, "groups": 0, "params": 0, "sizes": "", "stretches": 0, "B_chi2": float("nan"),
                        "B_rms": float("nan"), **res[0]})
        for r in rows[1:]:
            r.update(res[r["N"]])
        M = [r[key_sel] for r in rows]
        if args.ngroups is None:
            idx, why = ts.choose_n(M, args.min_delta, args.tie)    # idx: position in the list (0 = no TLS)
            best = rows[idx]["N"]
        else:
            best, why = args.ngroups, "set by --ngroups"
        for i, r in enumerate(rows):
            r["d_" + key_sel] = (M[i - 1] - r[key_sel]) if i > 0 else float("nan")
    else:
        if args.ngroups:
            best, why = args.ngroups, "set by --ngroups"
        else:
            idx, why = ts.choose_n_elbow([r["B_chi2"] for r in rows], args.min_gain)   # idx: number in the list (1-based)
            best = rows[idx - 1]["N"]
            why += ("\n  NOTE: a fit to B-factors improves with more groups; this is a rule of thumb. "
                    "Give --hklin to compare R-free for each number of domains")
    best_k = best
    for r in rows:
        r["selected"] = r["N"] == best_k
    df = pandas.DataFrame(rows)
    cols = ["N", "groups", "params", "sizes", "stretches", "B_rms"] + ([kw] if kw else []) + ([kf] if kf else []) + \
           (["d_" + (kf or kw), "-LL"] if kw else []) + ["selected"]
    df.to_csv(prefix + "_scan.csv", index=False)
    with open(prefix + "_scan.json", "w") as ofs:
        json.dump({"rows": rows, "selected_N": best_k, "reason": why, "enm": args.enm, "cutoff": cutoff,
                   "nmodes": len(w), "groups": {str(k): parts[k][1] for k in parts}}, ofs, indent=2, default=float)
    logger.writeln("\n" + "=" * 70)
    logger.writeln(f"Domains from the {args.enm.upper()} normal modes vs the number of domains")
    logger.writeln("B_rms: r.m.s. of B_obs - B_TLS (A^2) when one TLS group is fitted to the B-factors of every domain")
    logger.writeln(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}" if abs(x) < 100 else f"{x:.1f}"))
    if best_k == 0:
        logger.writeln(f"\nRecommendation: no TLS. {why}")
    else:
        labels, groups = parts[best_k]
        logger.writeln(f"\nRecommendation: {len(groups)} domain(s). {why}")
        for gi, g in enumerate(groups):
            logger.writeln(f"  domain {gi + 1} ({int((labels == gi).sum())} residues): " + ", ".join(
                "{}:{}".format(*s.split("/")[2:4]) for s in g))
        ts.write_yaml(prefix + "_tls.yaml", groups)
        write_domains_pdb(prefix + "_domains.pdb", st, nodes, labels)
        logger.writeln(f"\nTLS definition written: {prefix}_tls.yaml\n  use it with: refine_xtal --tls_file {prefix}_tls.yaml\n"
                       f"Model with the domain number as B-factor: {prefix}_domains.pdb  (colour by B to inspect the domains)\n"
                       f"Domain of every residue: {prefix}_domains_N*.csv; other numbers of domains: {prefix}_tls_N*.yaml")
    logger.writeln(f"Table: {prefix}_scan.csv, {prefix}_scan.json    total time {time.time() - t_start:.0f} s")
