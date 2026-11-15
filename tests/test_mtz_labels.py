"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import unittest
import numpy
import servalcat # must be imported before gemmi (registers the bundled gemmi)
import gemmi
from servalcat.xtal import sigmaa

def make_mtz(columns):
    """columns: list of (label, type)"""
    mtz = gemmi.Mtz(with_base=True)
    mtz.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    mtz.set_cell_for_all(gemmi.UnitCell(20, 20, 20, 90, 90, 90))
    for lab, typ in columns:
        mtz.add_column(lab, typ)
    n = 6
    mtz.set_data(numpy.zeros((n, 3 + len(columns)), dtype=numpy.float32))
    arr = numpy.array(mtz, copy=False)
    arr[:, 0] = numpy.arange(1, n + 1)
    arr[:, 1] = 1
    arr[:, 2] = 0
    for i, (lab, typ) in enumerate(columns):
        if typ == "I":
            arr[:, 3 + i] = numpy.arange(n) % 2 # test flags with two distinct values
        else:
            arr[:, 3 + i] = 1.0
    return mtz

class TestDecideMtzLabels(unittest.TestCase):
    def test_default_names_preferred(self):
        mtz = make_mtz([("FP", "F"), ("SIGFP", "Q"), ("F", "F"), ("SIGF", "Q"),
                        ("FREE", "I"), ("FreeR_flag", "I")])
        self.assertEqual(sigmaa.decide_mtz_labels(mtz), ["F", "SIGF", "FreeR_flag"])

    def test_fallback_to_other_names(self):
        mtz = make_mtz([("FP", "F"), ("SIGFP", "Q"), ("FREE", "I")])
        self.assertEqual(sigmaa.decide_mtz_labels(mtz), ["FP", "SIGFP", "FREE"])

    def test_default_free_only(self):
        mtz = make_mtz([("IMEAN", "J"), ("SIGIMEAN", "Q"), ("FREE", "I"), ("FreeR_flag", "I")])
        self.assertEqual(sigmaa.decide_mtz_labels(mtz), ["IMEAN", "SIGIMEAN", "FreeR_flag"])

    def test_amplitudes_before_intensities(self):
        mtz = make_mtz([("I", "J"), ("SIGI", "Q"), ("F", "F"), ("SIGF", "Q"), ("FreeR_flag", "I")])
        self.assertEqual(sigmaa.decide_mtz_labels(mtz), ["F", "SIGF", "FreeR_flag"])
        self.assertEqual(sigmaa.decide_mtz_labels(mtz, prefer_intensity=True), ["I", "SIGI", "FreeR_flag"])

if __name__ == '__main__':
    unittest.main()
