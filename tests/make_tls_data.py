"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.

Noise-free synthetic amplitudes calculated from a model (1l2h) with known TLS groups. Used by test_tls.py.
"""
from __future__ import absolute_import, division, print_function, generators
import os
import numpy
import servalcat # must be imported before gemmi
import gemmi
from servalcat import utils
from servalcat.refine import tls

root = os.path.abspath(os.path.dirname(__file__))

class _Item:
    def __init__(self, i, sels):
        self.id, self.selections = i, sels

# true T (A^2), L (deg^2), S (A*deg) of the two groups: residues 4-80 and 81-164 (waters are not in TLS groups)
TRUTH = [
    (numpy.array([[0.040, 0.005, 0.000], [0.005, 0.050, 0.004], [0.000, 0.004, 0.030]]),
     numpy.array([[3.0, 0.4, 0.2], [0.4, 2.0, -0.3], [0.2, -0.3, 1.5]]),
     numpy.array([[0.03, 0.02, -0.01], [0.00, 0.01, 0.04], [-0.02, 0.03, -0.04]])),
    (numpy.array([[0.060, -0.004, 0.002], [-0.004, 0.035, 0.000], [0.002, 0.000, 0.045]]),
     numpy.array([[1.5, -0.2, 0.0], [-0.2, 4.0, 0.5], [0.0, 0.5, 2.5]]),
     numpy.array([[-0.02, 0.05, 0.00], [0.03, 0.04, -0.03], [0.01, 0.00, -0.02]])),
]

def make(outdir):
    """Writes data.mtz (FP, SIGFP, FreeR_flag), start.pdb (isotropic B, no TLS), truth.pdb and truth_u.npy
    (true total U, pdb order) in outdir"""
    b_to_u = utils.model.b_to_u
    st = utils.fileio.read_structure(os.path.join(root, "1l2h", "1l2h.cif.gz"))
    st.setup_entities()
    for i, cra in enumerate(st[0].all()):
        cra.atom.serial = i + 1
        cra.atom.aniso = gemmi.SMat33f(0, 0, 0, 0, 0, 0)
    groups = tls.groups_from_selections(st, [_Item(1, ["//A/4-80/*"]), _Item(2, ["//A/81-164/*"])])
    true_u = numpy.zeros((st[0].count_atom_sites(), 6))
    for g, (T, L, S) in zip(groups, TRUTH):
        g.p = tls.p_from_matrices(T, L, S)
        for a, u in zip(g.atoms, g.u_tls()):
            u[:3] += a.b_iso * b_to_u # B in the model is the residual B
            a.aniso = gemmi.SMat33f(*u)
            true_u[a.serial - 1] = u
    for cra in st[0].all():
        if not cra.atom.aniso.nonzero():
            u = cra.atom.b_iso * b_to_u
            true_u[cra.atom.serial - 1] = [u, u, u, 0, 0, 0]
    numpy.save(os.path.join(outdir, "truth_u.npy"), true_u)
    utils.fileio.write_model(st, os.path.join(outdir, "truth"), pdb=True)

    mtz = gemmi.read_mtz_file(os.path.join(root, "1l2h", "1l2h.mtz.gz"))
    arr = numpy.array(mtz, copy=False)
    hkl = arr[:, :3].astype(int)
    free = arr[:, mtz.column_with_label("FreeR_flag").idx]
    fc = utils.model.calc_fc_fft(st, mtz.resolution_high() - 1e-6, source="xray", mott_bethe=False, miller_array=hkl)
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

    for cra in st[0].all(): # starting model: same coordinates and residual B, no anisotropy
        cra.atom.aniso = gemmi.SMat33f(0, 0, 0, 0, 0, 0)
    utils.fileio.write_model(st, os.path.join(outdir, "start"), pdb=True)
    return outdir
