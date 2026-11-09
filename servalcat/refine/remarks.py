"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.

REMARK 3 of the refined PDB file and the matching mmCIF items, in the layout of a PDB deposition: data used in
the refinement, fit to the data, highest-resolution bin, atom counts, B values, coordinate error (Cruickshank DPI),
correlation coefficients, restraint statistics, bulk solvent and other remarks. Everything is taken from what the
refinement itself has at the end (statistics of the last cycle, the likelihood object and the model); items that are
not available are written as NULL, as REFMAC does. Nothing needs another program.
"""
from __future__ import absolute_import, division, print_function, generators
import math
import gemmi

SOLVENT_NAMES = ("HOH", "DOD", "WAT", "H2O", "TIP", "SOL")

# (name in the statistics, text in REMARK 3, mmCIF restraint type)
RESTRAINT_ROWS = (
    ("Bond distances, non H",     "BOND LENGTHS REFINED ATOMS        (A)", "s_bond_nonh_d"),
    ("Bond angles, non H",        "BOND ANGLES REFINED ATOMS   (DEGREES)", "s_angle_nonh_deg"),
    ("Torsion angles, period 1",  "TORSION ANGLES, PERIOD 1    (DEGREES)", "s_dihedral_angle_1_deg"),
    ("Torsion angles, period 2",  "TORSION ANGLES, PERIOD 2    (DEGREES)", "s_dihedral_angle_2_deg"),
    ("Torsion angles, period 3",  "TORSION ANGLES, PERIOD 3    (DEGREES)", "s_dihedral_angle_3_deg"),
    ("Torsion angles, period 6",  "TORSION ANGLES, PERIOD 6    (DEGREES)", "s_dihedral_angle_6_deg"),
    ("Chiral centres",            "CHIRAL-CENTER RESTRAINTS       (A**3)", "s_chiral_restr"),
    ("Planar groups",             "GENERAL PLANES REFINED ATOMS      (A)", "s_planes"),
    ("VDW nonbonded",             "NON-BONDED CONTACTS REFINED ATOMS (A)", "s_nbd"),
    ("VDW torsion",               "NON-BONDED TORSION REFINED ATOMS  (A)", "s_nbtor"),
    ("VDW hbond",                 "H-BOND (X...Y) REFINED ATOMS      (A)", "s_hbond_nbd"),
    ("VDW metal",                 "POTENTIAL METAL-ION REFINED ATOMS (A)", "s_metal_ion"),
    ("VDW dummy",                 "DUMMY-ATOM CONTACTS REFINED ATOMS (A)", "s_dummy_nbd"),
    ("VDW nonbonded, symmetry",   "SYMMETRY VDW REFINED ATOMS        (A)", "s_symmetry_nbd"),
    ("VDW torsion, symmetry",     "SYMMETRY NON-BONDED TORSION ATOMS (A)", "s_symmetry_nbtor"),
    ("VDW hbond, symmetry",       "SYMMETRY H-BOND REFINED ATOMS     (A)", "s_symmetry_hbond_nbd"),
    ("VDW metal, symmetry",       "SYMMETRY METAL-ION REFINED ATOMS  (A)", "s_symmetry_metal_ion"),
    ("VDW dummy, symmetry",       "SYMMETRY DUMMY-ATOM CONTACTS  (A)", "s_symmetry_dummy_nbd"),
)
# restraints on the B values (the main-chain/side-chain split of REFMAC is not made here)
B_ROWS = (
    ("B values (bond)",   "B VALUES, BONDED ATOMS (1-2)    (A**2)"),
    ("B values (angle)",  "B VALUES, BONDED ATOMS (1-3)    (A**2)"),
    ("B values (others)", "B VALUES, NEIGHBOURS IN SPACE   (A**2)"),
)

def null(x, fmt="{}"):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NULL"
    return fmt.format(x)

def squeeze(lines):
    """no two empty REMARK 3 lines in a row"""
    out = []
    for l in lines:
        if not (l == "REMARK   3" and out and out[-1] == "REMARK   3"):
            out.append(l)
    return out

def atom_counts(st):
    """Non-hydrogen atoms of the first model: protein, nucleic acid, heterogen (ligands, ions, ...) and solvent"""
    n = dict(protein=0, nucleic=0, hetero=0, solvent=0)
    for chain in st[0]:
        for res in chain:
            n_heavy = sum(1 for a in res if not a.is_hydrogen())
            info = gemmi.find_tabulated_residue(res.name)
            if res.name in SOLVENT_NAMES or (info is not None and info.is_water()):
                n["solvent"] += n_heavy
            elif info is not None and info.is_amino_acid() and res.entity_type != gemmi.EntityType.NonPolymer:
                n["protein"] += n_heavy
            elif info is not None and info.is_nucleic_acid() and res.entity_type != gemmi.EntityType.NonPolymer:
                n["nucleic"] += n_heavy
            else:
                n["hetero"] += n_heavy
    return n

def cruickshank_dpi(r, d_min, n_atoms, n_obs, completeness):
    """Cruickshank's diffraction-component precision index of the coordinates, sigma(r) in A (Cruickshank 1999,
    Acta Cryst. D55, 583): sqrt(N_atoms / (N_obs - N_param)) * C**(-1/3) * R * d_min, with N_param = 4 N_atoms
    (x, y, z, B) and C the completeness as a fraction. None if it cannot be calculated."""
    if r is None or not completeness or completeness <= 0 or n_obs <= 4 * n_atoms or n_atoms == 0:
        return None
    return math.sqrt(n_atoms / (n_obs - 4. * n_atoms)) * completeness ** (-1. / 3) * r * d_min

def _block(title, rows):
    """lines 'REMARK   3   LABEL : value' with the colons aligned; rows: [(label, value)]"""
    w = max(len(l) for l, _ in rows)
    out = ["REMARK   3", f"REMARK   3  {title}"] if title else []
    out += [f"REMARK   3   {l:<{w}} : {v}" for l, v in rows]
    return out

def target_name(ll):
    if ll is None:
        return None
    if getattr(ll, "twin_data", None) is not None:
        return "MAXIMUM LIKELIHOOD ON " + ("INTENSITIES" if ll.is_int else "AMPLITUDES") + " (TWINNED)"
    return "MAXIMUM LIKELIHOOD ON " + ("INTENSITIES" if ll.is_int else "AMPLITUDES")

def solvent_lines(ll):
    if ll is None:
        return []
    if not ll.use_solvent:
        return ["REMARK   3", "REMARK   3  BULK SOLVENT MODELLING.", "REMARK   3   METHOD USED : NONE"]
    m = gemmi.SolventMasker(gemmi.AtomicRadiiSet.Refmac)
    rows = [("METHOD USED", "MASK BULK SOLVENT" + (" (NON-BINARY MASK)" if ll.use_solvent == 2 else "")),
            ("VDW PROBE RADIUS", f"{m.rprobe:.2f}"), ("SHRINKAGE RADIUS", f"{m.rshrink:.2f}")]
    sc = getattr(ll, "scaling", None)
    if sc is not None and getattr(sc, "k_sol", None) is not None:
        rows += [("K_SOL (E/A**3)", f"{sc.k_sol:.3f}"), ("B_SOL (A**2)", f"{sc.b_sol:.2f}")]
    return _block("BULK SOLVENT MODELLING.", rows)

def bin_values(b):
    """(R work, R free) of a bin of the statistics, for amplitudes or intensities"""
    rw = b.get("Rwork", b.get("R1work"))
    rf = b.get("Rfree", b.get("R1free"))
    return rw, rf

def fill_refinement_info(ri, st, stats, ll, counts, n_cycles):
    """Items of the gemmi.RefinementInfo that are not set yet (written to the _refine categories of the mmCIF file)"""
    summ = stats["data"]["summary"]
    for k_work, k_free in (("CCFworkavg", "CCFfreeavg"), ("CCIworkavg", "CCIfreeavg")):
        if k_work in summ:
            ri.cc_fo_fc_work = round(summ[k_work], 4)
            if k_free in summ:
                ri.cc_fo_fc_free = round(summ[k_free], 4)
            break
    n_atoms = sum(counts.values())
    d_min = ri.resolution_high
    c = ri.completeness / 100. if ri.completeness and ri.completeness > 0 else None
    for r, attr in ((ri.r_work, "dpi_cruickshank_r"), (ri.r_free, "dpi_cruickshank_rfree")):
        if r is not None and r > 0:
            v = cruickshank_dpi(r, d_min, n_atoms, ri.reflection_count, c)
            if v is not None:
                setattr(ri, attr, round(v, 3))
    if ri.bins:
        ri.bin_count = len(ri.bins)

def has_resolution(ri):
    return ri is not None and ri.resolution_high is not None and ri.resolution_high > 0

def remark3(st, stats, ll, ri, geom_rows, counts, mean_b):
    """The complete REMARK 3 text (without the TLS section, which update_meta_tls() adds before the bulk solvent)"""
    import servalcat
    lines = []
    if has_resolution(ri):
        lines += ["REMARK   2", f"REMARK   2 RESOLUTION. {ri.resolution_high:7.2f} ANGSTROMS."]
    lines += ["REMARK   3", "REMARK   3 REFINEMENT.",
             f"REMARK   3   PROGRAM     : SERVALCAT {servalcat.__version__}",
             "REMARK   3   AUTHORS     : YAMASHITA,MURSHUDOV",
             "REMARK   3"]
    has_data = ll is not None and "data" in stats
    if has_data:
        lines.append(f"REMARK   3    REFINEMENT TARGET : {target_name(ll)}")
        lines.append("REMARK   3")
        lines += _block("DATA USED IN REFINEMENT.", [
            ("RESOLUTION RANGE HIGH (ANGSTROMS)", null(ri.resolution_high, "{:.2f}")),
            ("RESOLUTION RANGE LOW  (ANGSTROMS)", null(ri.resolution_low, "{:.2f}")),
            ("DATA CUTOFF            (SIGMA(F))", "NULL"),
            ("COMPLETENESS FOR RANGE        (%)", null(ri.completeness, "{:.1f}")),
            ("NUMBER OF REFLECTIONS", null(ri.reflection_count, "{}"))])
        rtest = None
        if ri.reflection_count > 0 and ri.rfree_set_count >= 0:
            rtest = 100. * ri.rfree_set_count / ri.reflection_count
        lines += _block("FIT TO DATA USED IN REFINEMENT.", [
            ("CROSS-VALIDATION METHOD", "FREE R-VALUE" if ri.rfree_set_count > 0 else "NULL"),
            ("FREE R VALUE TEST SET SELECTION", "NULL"),
            ("R VALUE     (WORKING + TEST SET)", null(ri.r_all, "{:.3f}")),
            ("R VALUE            (WORKING SET)", null(ri.r_work, "{:.3f}")),
            ("FREE R VALUE", null(ri.r_free, "{:.3f}")),
            ("FREE R VALUE TEST SET SIZE   (%)", null(rtest, "{:.3f}")),
            ("FREE R VALUE TEST SET COUNT", null(ri.rfree_set_count if ri.rfree_set_count >= 0 else None))])
        binned = stats["data"].get("binned") or []
        if binned:
            hi = min(binned, key=lambda b: b["d_min"])
            rw, rf = bin_values(hi)
            lines += _block("FIT IN THE HIGHEST RESOLUTION BIN.", [
                ("TOTAL NUMBER OF BINS USED", str(len(binned))),
                ("BIN RESOLUTION RANGE HIGH       (A)", f"{hi['d_min']:.2f}"),
                ("BIN RESOLUTION RANGE LOW        (A)", f"{hi['d_max']:.2f}"),
                ("REFLECTION IN BIN     (WORKING SET)", null(hi.get("n_work"))),
                ("BIN COMPLETENESS (WORKING+TEST) (%)", null(hi.get("Cmpl"), "{:.2f}")),
                ("BIN R VALUE           (WORKING SET)", null(rw, "{:.4f}")),
                ("BIN FREE R VALUE SET COUNT", null(hi.get("n_free"))),
                ("BIN FREE R VALUE", null(rf, "{:.4f}"))])
    lines += _block("NUMBER OF NON-HYDROGEN ATOMS USED IN REFINEMENT.", [
        ("PROTEIN ATOMS", str(counts["protein"])), ("NUCLEIC ACID ATOMS", str(counts["nucleic"])),
        ("HETEROGEN ATOMS", str(counts["hetero"])), ("SOLVENT ATOMS", str(counts["solvent"]))])
    brows = [("FROM WILSON PLOT           (A**2)", "NULL"),
             ("MEAN B VALUE      (OVERALL, A**2)", null(mean_b, "{:.2f}"))]
    lines += _block("B VALUES.", brows)
    if ll is not None and ll.b_aniso is not None:
        e = ll.b_aniso.elements_pdb()
        lines.append("REMARK   3   OVERALL ANISOTROPIC B VALUE.")
        for name, v in zip(("B11", "B22", "B33", "B12", "B13", "B23"), e):
            lines.append(f"REMARK   3    {name} (A**2) : {v:.5f}")
    if has_data:
        lines += _block("ESTIMATED OVERALL COORDINATE ERROR.", [
            ("DPI (CRUICKSHANK) BASED ON R VALUE         (A)", null(ri.dpi_cruickshank_r, "{:.3f}")),
            ("DPI (CRUICKSHANK) BASED ON FREE R VALUE    (A)", null(ri.dpi_cruickshank_rfree, "{:.3f}"))])
        lines += _block("CORRELATION COEFFICIENTS (MEAN OF THE RESOLUTION BINS).", [
            ("CORRELATION COEFFICIENT FO-FC", null(ri.cc_fo_fc_work, "{:.3f}")),
            ("CORRELATION COEFFICIENT FO-FC FREE", null(ri.cc_fo_fc_free, "{:.3f}"))])
    if geom_rows:
        lines.append("REMARK   3")
        lines.append("REMARK   3  RMS DEVIATIONS FROM IDEAL VALUES        COUNT    RMS    WEIGHT")
        for label, (count, rms, weight) in geom_rows["restraints"]:
            lines.append(f"REMARK   3   {label}:{count:6d} ;{rms:6.3f} ;{weight:6.3f}")
        if geom_rows["b"]:
            lines.append("REMARK   3")
            lines.append("REMARK   3  THERMAL FACTOR RESTRAINTS.              COUNT   RMS     WEIGHT")
            for label, (count, rms, weight) in geom_rows["b"]:
                lines.append(f"REMARK   3   {label}:{count:6d} ;{rms:6.3f} ;{weight:6.3f}")
    # the TLS section goes here (update_meta_tls)
    lines += solvent_lines(ll)
    other = []
    if st[0].has_hydrogen():
        other.append("HYDROGENS HAVE BEEN ADDED IN THEIR RIDING POSITIONS")
    if stats.get("Ncyc") is not None:
        other.append(f"NUMBER OF REFINEMENT CYCLES: {stats['Ncyc']}")
    if stats.get("weight") is not None:
        other.append(f"WEIGHT OF THE GEOMETRY RESTRAINTS (MATRIX WEIGHT): {stats['weight']:.3f}")
    if other:
        lines += ["REMARK   3", "REMARK   3  OTHER REFINEMENT REMARKS:"] + [f"REMARK   3   {o}" for o in other]
    return squeeze(lines)

def refine_hist_rows(st):
    """Items of _refine_hist: atoms by kind at the end of the refinement"""
    n = atom_counts(st)
    ri = st.meta.refinement[0] if st.meta.refinement else None
    return dict(pdbx_refine_id=ri.id if ri and ri.id else "X-RAY DIFFRACTION", cycle_id="LAST",
                d_res_high=ri.resolution_high if ri and ri.resolution_high > 0 else None,
                d_res_low=ri.resolution_low if ri and ri.resolution_low > 0 else None,
                pdbx_number_atoms_protein=n["protein"], pdbx_number_atoms_nucleic_acid=n["nucleic"],
                pdbx_number_atoms_ligand=n["hetero"], number_atoms_solvent=n["solvent"],
                number_atoms_total=sum(n.values()))

def add_refine_hist(block, st):
    """Write the _refine_hist category of the mmCIF block"""
    row = refine_hist_rows(st)
    tags = list(row)
    vals = ["?" if row[k] is None else (f"{row[k]:.3f}" if isinstance(row[k], float) else str(row[k])) for k in tags]
    block.set_mmcif_category("_refine_hist.", {k: [v] for k, v in zip(tags, vals)})
