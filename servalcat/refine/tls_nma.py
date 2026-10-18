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

class ChainResidues(list):
    """the residues of a chain that are nodes, in order; blockers: numbers of the chain's other residues (waters, ligands)"""
    blockers = ()

def collect_nodes(st):
    """One node per polymer residue of the first model: C-alpha of amino acids, C4' of nucleotides. Residues are
    recognised by name (not by entity or connectivity), waters and ligands are ignored.
    Returns the list of Node and {chain name: ChainResidues, the residues of the chain that are nodes, in order}"""
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
            by_chain.setdefault(ch.name, ChainResidues()).append(r)
    for ch in st[0]:
        if ch.name in by_chain:
            polymer = {r.seqid.num for r in ch if tls_mod.is_polymer_residue(r)}
            by_chain[ch.name].blockers = sorted({r.seqid.num for r in ch} - polymer)
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

def edge_dissimilarity(pos, edges, enm, w, V, chain_ids=None):
    """Dissimilarity of the two nodes of every edge: 0 = move together as a rigid body. chain_ids: chain of every
    node; for gnm the fluctuation of a chain as a whole is then removed from its modes (in a complex, the slow modes are
    mostly the motions of the chains against each other, which would hide the domains inside a chain; in anm the
    relative displacement of two nodes of a chain already ignores it)."""
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
    if chain_ids is not None:
        V = V.copy()
        for c in numpy.unique(chain_ids):
            m = chain_ids == c
            V[m] -= V[m].mean(axis=0)
    corr = (V[i] * V[j] * wt).sum(1)
    norm = numpy.sqrt((V[i] ** 2 * wt).sum(1) * (V[j] ** 2 * wt).sum(1))
    return numpy.sqrt(numpy.maximum(1.0 - corr / numpy.maximum(norm, 1e-30), 0.0))

# ---------------------------------------------------------------------------------------------------
# clustering
def graph_affinity(n, edges, dist):
    sigma = max(float(numpy.median(dist)) if len(dist) else 1.0, 1e-9)
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
    """selections (gemmi) for the residues of one group: one per stretch of consecutive residues of a chain (a water or
    ligand that is numbered inside a stretch is not selected)"""
    in_group = {id(n.residue) for n, l in zip(nodes, labels) if l == label}
    out = []
    for chain, residues in by_chain.items():
        run = []   # consecutive residues of the chain that are in the group
        for r in residues:
            if id(r) in in_group:
                run.append(r)
            elif run:
                out += tls_mod.range_selections(chain, run, residues.blockers)
                run = []
        if run:
            out += tls_mod.range_selections(chain, run, residues.blockers)
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

# ---------------------------------------------------------------------------------------------------
# networks and domains
class Network:
    """One elastic network: its nodes (indices into the list of all nodes), their positions, the springs (local
    indices) and the lowest modes"""
    def __init__(self, members, pos, edges, w, V):
        self.members, self.pos, self.edges, self.w, self.V = members, pos, edges, w, V

def build_networks(nodes, enm, cutoff, nmodes, scope="complex", min_nodes=0):
    """scope 'complex': ONE network over all polymer chains, so that the architecture of the assembly (the springs
    between neighbouring chains) shapes the modes; 'chain': a network for every chain on its own. In the second case
    chains with fewer than min_nodes residues are left out."""
    pos = numpy.array([n.pos for n in nodes])
    chain_of = numpy.array([n.chain for n in nodes])
    if scope == "complex":
        member_sets = [numpy.arange(len(nodes))]
    else:
        member_sets = [numpy.where(chain_of == c)[0] for c in dict.fromkeys(chain_of.tolist())]
        member_sets = [m for m in member_sets if len(m) >= max(min_nodes, nmodes + 8)]
    nets = []
    for members in member_sets:
        edges, w, V = network_modes(pos[members], enm, cutoff, nmodes)
        nets.append(Network(members, pos[members], edges, w, V))
    return nets

class Unit:
    """A set of nodes that is clustered on its own: a chain (default) or a whole network. A: weighted contact graph"""
    def __init__(self, chain, members, A):
        self.chain, self.members, self.A = chain, members, A

def make_units(nodes, networks, enm, confine):
    units = []
    for net in networks:
        chain_ids = numpy.array([nodes[g].chain for g in net.members])
        dist = edge_dissimilarity(net.pos, net.edges, enm, net.w, net.V, chain_ids)
        if not confine:
            units.append(Unit(None, net.members, graph_affinity(len(net.members), net.edges, dist)))
            continue
        for c in dict.fromkeys(chain_ids.tolist()):
            m = chain_ids == c
            sel = numpy.where(m)[0]
            keep = m[net.edges[:, 0]] & m[net.edges[:, 1]]       # only the springs inside the chain
            remap = -numpy.ones(len(m), dtype=int)
            remap[sel] = numpy.arange(len(sel))
            units.append(Unit(c, net.members[sel], graph_affinity(len(sel), remap[net.edges[keep]], dist[keep])))
    return units

class DomainFinder:
    """Domains from the modes of the networks. By default (span_chains off) every chain is clustered on its own, using
    the modes of the network it is in, so that a group never contains residues of two chains and a number of domains
    means domains per chain. With span_chains the whole network is clustered at once and a domain may consist of
    several chains."""
    def __init__(self, nodes, by_chain, args, networks, enm):
        self.nodes, self.by_chain, self.args = nodes, by_chain, args
        self.confine = not getattr(args, "span_chains", False)
        self.units = [u for u in make_units(nodes, networks, enm, self.confine) if len(u.members) >= args.min_residues]
        self.cache = {}

    def chain_names(self):
        return [u.chain for u in self.units]

    def capacity(self, unit):
        return max(1, len(unit.members) // self.args.min_residues)

    def labels_for(self, k_of):
        """label (group number) of every node, -1 for nodes in no group; k_of(unit): number of domains of the unit"""
        labels = -numpy.ones(len(self.nodes), dtype=int)
        off = 0
        for u in self.units:
            k = min(k_of(u), self.capacity(u))
            key = (id(u), k)
            if key not in self.cache:
                self.cache[key] = self._cluster(u, k)
            lab = self.cache[key]
            labels[u.members] = lab + off
            off += int(lab.max()) + 1
        return labels

    def _cluster(self, u, k):
        a = self.args
        if k <= 1:
            return numpy.zeros(len(u.members), dtype=int)
        lab = clean_labels(spectral_labels(u.A, k), u.A, a.min_fragment, a.min_residues)
        if getattr(a, "min_stretch", 0) > 1:
            local = [Node(self.nodes[g].chain, self.nodes[g].residue, self.nodes[g].pos, i) for i, g in enumerate(u.members)]
            lab = clean_labels(smooth_stretches(lab, local, u.A, a.min_stretch), u.A, a.min_fragment, a.min_residues)
        return lab

    def result(self, labels):
        return labels, groups_from_labels(self.nodes, labels, self.by_chain)

    def scan(self):
        """{k: (labels, groups)} for k = 1 .. max_groups (per chain, or in total with span_chains); a number is left
        out if it gives the same partition as a smaller one"""
        ret, seen = {}, set()
        for k in range(1, self.args.max_groups + 1):
            labels = self.labels_for(lambda u: k)
            key = tuple(labels)
            if key not in seen:
                seen.add(key)
                ret[k] = self.result(labels)
        return ret

def partitions(nodes, by_chain, args, networks, enm):
    return DomainFinder(nodes, by_chain, args, networks, enm).scan()

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
        add(g, "--network", choices=["complex", "chain"], default="complex",
            help="complex: one elastic network of the whole assembly, so that the contacts between chains shape the modes; "
            "chain: a network for every chain on its own")
        add(g, "--max_groups", type=int, default=6, metavar="N",
            help="Largest number of domains (TLS groups) per chain (in total with --span_chains)")
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
        add(g, "--ngroups", nargs="+", metavar="N | CHAIN:N", show="none: scan 1 .. --max_groups",
            help="Use exactly this number of domains in every chain (e.g. 2), or per chain (e.g. A:2 B:1; chains not "
            "given: 1). With --span_chains: the total number")
        add(g, "--span_chains", action="store_true",
            help="Cluster the whole network at once, so that a domain may consist of several chains (default: every "
            "chain is divided on its own and a group never contains two chains)")
        add(g, "--no_eval", action="store_true", help="Do not refine to get R-values (always so without --hklin)")
    refine_xtal.add_arguments(parser, after_required=own_options, hklin_required=False)
    import textwrap
    parser.usage = "{} --model FILE [--hklin FILE] [options]".format(parser.prog)
    parser.description = textwrap.fill(
        "Find domains that move together from the low-frequency normal modes of an elastic network model of the "
        "polymer (anisotropic or Gaussian) and write them as TLS groups. Every chain of the assembly is divided on its "
        "own (--span_chains: domains of several chains), by default using the network of the whole complex. A domain "
        "may consist of several stretches of residues. Without --hklin only the model is read. With --hklin every number of domains "
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
    ts.write_domain_csv(f"{prefix}_domains_N{n}.csv", [(nd.chain, nd.residue, int(l) + 1) for nd, l in zip(nodes, labels)])

def write_domains_pdb(path, st, nodes, labels):
    """the model with the domain number in the B-factor column (0 for atoms outside the network), to colour by B"""
    ts.write_domains_pdb(path, st, [(nd.chain, nd.residue, int(l) + 1) for nd, l in zip(nodes, labels)])

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
    span = getattr(args, "span_chains", False)
    if span and args.network == "chain":
        raise SystemExit("Error: --span_chains needs the network of the whole assembly (--network complex)")
    t0 = time.time()
    networks = build_networks(nodes, args.enm, cutoff, args.nmodes, args.network, args.min_residues)
    if not networks:
        raise SystemExit(f"Error: no chain with at least {args.min_residues} polymer residues")
    for net in networks:
        chains = [nodes[g].chain for g in net.members]
        n_chains = len(set(chains))
        between = int(sum(1 for a_, b_ in net.edges if chains[a_] != chains[b_]))
        what = f"whole assembly ({n_chains} chains)" if n_chains > 1 else f"chain {chains[0]}"
        logger.writeln(f"{args.enm.upper()} of the {what}: {len(net.members)} nodes, {len(net.edges)} springs "
                       f"(cutoff {cutoff:g} A" + (f"; {between} between chains" if n_chains > 1 else "") + "), "
                       f"{len(net.w)} lowest modes, eigenvalues {net.w[0]:.3f} .. {net.w[-1]:.3f}")
        nc = n_components(len(net.members), net.edges)
        if nc > 1 and args.network == "complex":
            logger.writeln(f"NOTE: the network has {nc} separate parts (chains or domains farther apart than the cutoff); "
                           "they move independently")
    logger.writeln(f"({time.time() - t0:.1f} s) " + (
        "Every chain is divided into domains on its own; a group never contains residues of two chains."
        if not span else "The whole network is clustered at once; a domain may consist of several chains."))
    finder = DomainFinder(nodes, by_chain, args, networks, args.enm)
    skipped = [c for c, r in by_chain.items() if c not in finder.chain_names()] if not span else []
    if skipped:
        logger.writeln(f"NOTE: chain(s) {', '.join(skipped)} have fewer than {args.min_residues} polymer residues and are "
                       "not in a TLS group")
    parts = finder.scan()
    if args.ngroups:
        if span:
            try:
                k = int(args.ngroups[0])
            except ValueError:
                raise SystemExit("Error: with --span_chains --ngroups is one number")
            labels = finder.labels_for(lambda u: k)
        else:
            counts = ts.parse_ngroups(args.ngroups, finder.chain_names())
            by_unit = {u.chain: counts[u.chain] for u in finder.units}
            labels = finder.labels_for(lambda u: by_unit[u.chain])
            for u in finder.units:
                got = len(set(labels[u.members]))
                if got < by_unit[u.chain]:
                    logger.writeln(f"WARNING: chain {u.chain}: {by_unit[u.chain]} domains requested, {got} possible "
                                   f"with at least {args.min_residues} residues")
        res = finder.result(labels)
        parts = {len(res[1]): res}
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
            best, why = rows[-1]["N"], "set by --ngroups"
        for i, r in enumerate(rows):
            r["d_" + key_sel] = (M[i - 1] - r[key_sel]) if i > 0 else float("nan")
    else:
        if args.ngroups:
            best, why = rows[-1]["N"], "set by --ngroups"
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
                   "network": args.network, "span_chains": bool(span), "nmodes": min(len(net.w) for net in networks),
                   "groups": {str(k): parts[k][1] for k in parts}}, ofs, indent=2, default=float)
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
