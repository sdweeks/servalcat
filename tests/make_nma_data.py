"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.

Pseudo-proteins with a known domain structure, to test the normal-mode domain finder (test_tls_nma.py): compact
clouds of C-alpha atoms that touch only weakly. The sequence order is interleaved: the second domain is INTERNAL
(residues 41-80 of one chain) and the first is made of the two flanks (1-40 and 81-120).
"""
from __future__ import absolute_import, division, print_function, generators
import numpy
import servalcat # must be imported before gemmi
import gemmi

def cloud(rng, n, radius, center, min_dist=3.8):
    """n points in a sphere, no two closer than min_dist"""
    pts = []
    while len(pts) < n:
        p = rng.normal(size=3)
        p = p / numpy.linalg.norm(p) * radius * rng.uniform() ** (1. / 3)
        if all(numpy.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
    return numpy.array(pts) + numpy.asarray(center, float)

def build(layout, seed=0, separation=36.0):
    """layout: list of (domain index, residue count) in sequence order for ONE chain, e.g. [(0, 40), (1, 40), (0, 40)].
    Domains sit on a line `separation` A apart (domain d at z = d * separation), radius from the number of residues.
    Returns (structure, truth) with truth = domain index of every residue in order."""
    rng = numpy.random.default_rng(seed)
    counts = {}
    for d, n in layout:
        counts[d] = counts.get(d, 0) + n
    pts = {d: cloud(rng, n, (n * 120. * 3 / (4 * numpy.pi)) ** (1. / 3), [0, 0, d * separation]) for d, n in counts.items()}
    used = {d: 0 for d in counts}
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(150, 150, 150, 90, 90, 90)
    st.spacegroup_hm = "P 1"
    model = gemmi.Model(1)
    ch = gemmi.Chain("A")
    truth = []
    aa = ["ALA", "LEU", "GLU", "LYS", "VAL", "SER", "GLY", "ILE", "ASP", "THR"]
    i = 0
    for d, n in layout:
        for _ in range(n):
            ca = pts[d][used[d]]
            used[d] += 1
            i += 1
            r = gemmi.Residue()
            r.name = aa[i % len(aa)]
            r.seqid = gemmi.SeqId(i, " ")
            r.entity_type = gemmi.EntityType.Polymer
            for name, off in (("N", (-1.2, 0.5, 0.0)), ("CA", (0, 0, 0)), ("C", (1.2, -0.5, 0.3)), ("O", (1.5, -1.5, 0.5))):
                a = gemmi.Atom()
                a.name = name
                a.element = gemmi.Element("C" if name == "CA" else name)
                a.pos = gemmi.Position(*(ca + numpy.array(off)))
                a.occ = 1.0
                a.b_iso = 20.0
                r.add_atom(a)
            ch.add_residue(r)
            truth.append(d)
    model.add_chain(ch)
    st.add_model(model)
    return st, numpy.array(truth)

def accuracy(labels, truth):
    """fraction of residues in agreement with the true domains, for the best assignment of the label numbers"""
    import itertools
    labs, tr = sorted(set(labels)), sorted(set(truth))
    best = 0
    for perm in itertools.permutations(range(max(len(labs), len(tr)))):
        m = {l: perm[i] for i, l in enumerate(labs)}
        best = max(best, sum(1 for l, t in zip(labels, truth) if m[l] == tr[perm[0]] * 0 + t) / len(truth))
    return best
