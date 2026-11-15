"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.

Noise-free synthetic amplitudes of 1l2h with known occupancies of alternative conformations, a lone conformer, a partially
occupied side chain and a partially occupied water. Used by test_occ_groups.py.
"""
from __future__ import absolute_import, division, print_function, generators
import os
import numpy
import servalcat # must be imported before gemmi
import gemmi
from servalcat import utils

root = os.path.abspath(os.path.dirname(__file__))
MAIN_CHAIN = ("N", "CA", "C", "O")
TRUE_PAIR = (0.7, 0.3)   # occupancies of conformers A, B of three residues
TRUE_LONE = 0.6          # residue with only conformer A
TRUE_PARTIAL = 0.5       # side chain without altloc
TRUE_WATER = 0.5

def _residues(st, names, n):
    out = []
    for r in st[0]["A"]:
        if r.name in names and r.het_flag == "A" and len(r) > 6 and not any(a.has_altloc() for a in r):
            out.append(r)
        if len(out) == n:
            break
    return out

def make(outdir, d_min=2.2):
    """Writes data.mtz (FP, SIGFP, FreeR_flag), truth.pdb and start.pdb (same coordinates and B; occupancies of
    the conformers 0.5/0.5, lone conformer 0.9, partial side chain and water 0.8). Returns a dict that describes the
    sites: pairs [(seqnum, ...)], lone, partial, water (seqnum) and the true occupancies."""
    st = utils.fileio.read_structure(os.path.join(root, "1l2h", "1l2h.cif.gz"))
    st.setup_entities()
    for cra in st[0].all():
        cra.atom.aniso = gemmi.SMat33f(0, 0, 0, 0, 0, 0)
    chain = st[0]["A"]
    # distinct residues far apart in the sequence, long side chains
    cands = _residues(st, ("LYS", "ARG", "GLU", "GLN", "MET"), 40)
    picks = [cands[i] for i in (3, 10, 18)]
    lone = cands[26]
    partial = cands[34]
    water = [r for r in chain if r.name == "HOH"][5]
    shift = gemmi.Position(1.1, 0.7, 0.5)

    for r in picks:
        clones = []
        for a in r:                     # clone first, add later (adding invalidates references to atoms)
            if a.name in MAIN_CHAIN:
                continue
            b = a.clone()
            b.altloc = "B"
            b.pos = gemmi.Position(a.pos.x + shift.x, a.pos.y + shift.y, a.pos.z + shift.z)
            b.occ = TRUE_PAIR[1]
            clones.append(b)
            a.altloc = "A"
            a.occ = TRUE_PAIR[0]
        for b in clones:
            r.add_atom(b)
    for a in lone:
        if a.name not in MAIN_CHAIN:
            a.altloc = "A"
            a.occ = TRUE_LONE
    for a in partial:
        if a.name not in MAIN_CHAIN:
            a.occ = TRUE_PARTIAL
    for a in water:
        a.occ = TRUE_WATER
    for i, cra in enumerate(st[0].all()):
        cra.atom.serial = i + 1
    utils.fileio.write_model(st, os.path.join(outdir, "truth"), pdb=True)

    mtz = gemmi.read_mtz_file(os.path.join(root, "1l2h", "1l2h.mtz.gz"))
    arr = numpy.array(mtz, copy=False)
    hkl = arr[:, :3].astype(int)
    free = arr[:, mtz.column_with_label("FreeR_flag").idx]
    d = mtz.cell.calculate_d_array(hkl) if hasattr(mtz.cell, "calculate_d_array") else None
    sel = numpy.ones(len(hkl), dtype=bool) if d is None else (d >= d_min)
    hkl, free = hkl[sel], free[sel]
    fc = utils.model.calc_fc_fft(st, d_min - 1e-6, source="xray", mott_bethe=False, miller_array=hkl)
    amp = numpy.abs(fc)
    m = gemmi.Mtz(with_base=True)
    m.cell = mtz.cell
    m.spacegroup = mtz.spacegroup
    m.add_dataset("synth")
    m.add_column("FreeR_flag", "I")
    m.add_column("FP", "F")
    m.add_column("SIGFP", "Q")
    m.set_data(numpy.column_stack([hkl, free, amp, 0.02 * amp + 0.01 * amp.mean()]))
    m.write_to_file(os.path.join(outdir, "data.mtz"))

    # starting occupancies
    for r in picks:
        for a in r:
            if a.has_altloc():
                a.occ = 0.5
    for a in lone:
        if a.has_altloc():
            a.occ = 0.9
    for a in partial:
        if a.name not in MAIN_CHAIN:
            a.occ = 0.8
    for a in water:
        a.occ = 0.8
    utils.fileio.write_model(st, os.path.join(outdir, "start"), pdb=True)
    return dict(pairs=[r.seqid.num for r in picks], lone=lone.seqid.num, partial=partial.seqid.num,
                water=water.seqid.num, true_pair=TRUE_PAIR, true_lone=TRUE_LONE, true_partial=TRUE_PARTIAL,
                true_water=TRUE_WATER)
