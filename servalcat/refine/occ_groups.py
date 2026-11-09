"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.

Occupancy groups from the input model (refine_xtal --refine_occ, servalcat util occ_groups)

The model is scanned for

* alternative conformations: every conformer (altloc) of a residue is one occupancy group, and the conformers of the
  residue are constrained to sum to 1 ("complete"; a residue with a single altloc is "incomplete": sum <= 1),
* partially occupied atoms without altloc: the atoms of a residue with 0 < occ < 1 share one group (sum <= 1),
* waters (policy 'partial': waters with occ < 1; 'all': every water; 'none'): one group per water (sum <= 1).

and the description that Servalcat needs (see docs/occ_refine.rst) is generated: atom_selection.occ.include,
occ_groups, occ_group_constraints and, because occupancy and B are strongly correlated, stronger ADP restraints
(local_adpr_weights) around the atoms whose occupancy is refined. The rules follow the occupancy keywords of REFMAC5
(http://www.ysbl.york.ac.uk/refmac/data/refmac_news.html#Occupancy).
"""
from __future__ import absolute_import, division, print_function, generators
import dataclasses
from typing import List
import gemmi
import omegaconf
from servalcat.utils import logger

KINDS = ("alt", "partial", "water")

@dataclasses.dataclass
class OccGroup:
    id: int
    kind: str                       # alt, partial or water
    chain: str
    resi: str                       # residue number (with insertion code)
    resname: str
    alt: str                        # altloc ("" if none)
    selections: List[str]
    natoms: int
    occ: float                      # mean occupancy in the input model

@dataclasses.dataclass
class OccConstraint:
    ids: List[int]
    complete: bool                  # sum of occupancies = 1 (else <= 1)

@dataclasses.dataclass
class OccResult:
    groups: List[OccGroup] = dataclasses.field(default_factory=list)
    constraints: List[OccConstraint] = dataclasses.field(default_factory=list)
    adpr_selections: List[str] = dataclasses.field(default_factory=list)
    notes: List[str] = dataclasses.field(default_factory=list)

    @property
    def include(self):
        return [s for g in self.groups for s in g.selections]

    def count(self, kind):
        return sum(1 for g in self.groups if g.kind == kind)

def _is_water(res):
    return res.is_water() or res.name in ("HOH", "WAT", "DOD", "H2O")

def find_occupancy_groups(st, waters="partial", alts_complete=True, tol=1e-3):
    """Scan the first model of st. waters: 'partial', 'all' or 'none'. alts_complete: conformers of a residue sum
    to 1 (otherwise only <= 1). Returns an OccResult."""
    assert waters in ("partial", "all", "none")
    res = OccResult()
    adpr = {}
    n_zero = 0
    for chain in st[0]:
        by_seqid = {}
        for r in chain:   # residues with the same seqid (microheterogeneity) are one site
            by_seqid.setdefault(str(r.seqid), []).append(r)
        for resi, residues in by_seqid.items():
            atoms = [a for r in residues for a in r]
            names = "/".join(dict.fromkeys(r.name for r in residues))
            site = "//{}/{}".format(chain.name, resi)
            alts = sorted({a.altloc for a in atoms if a.has_altloc()})
            plain = [a for a in atoms if not a.has_altloc()]
            n_zero += sum(1 for a in plain if a.occ <= 0)

            if alts:
                ids = []
                for alt in alts:
                    members = [a for a in atoms if a.altloc == alt]
                    g = OccGroup(len(res.groups) + 1, "alt", chain.name, resi, names, alt, ["{}/*:{}".format(site, alt)],
                                 len(members), sum(a.occ for a in members) / len(members))
                    res.groups.append(g)
                    ids.append(g.id)
                complete = alts_complete and len(alts) >= 2
                res.constraints.append(OccConstraint(ids, complete))
                total = sum(res.groups[i - 1].occ for i in ids)
                if complete and abs(total - 1) > 0.05:
                    res.notes.append("{}/{} {}: the occupancies of the conformers {} sum to {:.2f} in the input; "
                                     "they are constrained to sum to 1".format(chain.name, resi, names, "/".join(alts), total))
                elif len(alts) == 1:
                    res.notes.append("{}/{} {}: only conformer {} is present (occupancy {:.2f}); its occupancy is "
                                     "refined with sum <= 1".format(chain.name, resi, names, alts[0], res.groups[ids[0] - 1].occ))
                adpr[(chain.name, resi)] = "{}/*".format(site)

            part = [a for a in plain if tol < a.occ < 1 - tol]
            if _is_water(residues[0]) and not alts:
                if waters == "none" or (waters == "partial" and not part):
                    continue
                members = plain if waters == "all" else part
                g = OccGroup(len(res.groups) + 1, "water", chain.name, resi, names, "", ["{}/*".format(site)],
                             len(members), sum(a.occ for a in members) / max(1, len(members)))
                res.groups.append(g)
                res.constraints.append(OccConstraint([g.id], False))
            elif part:
                if len(part) == len(atoms):
                    sels = ["{}/*".format(site)]
                else:
                    sels = ["{}/{}".format(site, a.name) for a in part]
                g = OccGroup(len(res.groups) + 1, "partial", chain.name, resi, names, "", sels, len(part),
                             sum(a.occ for a in part) / len(part))
                res.groups.append(g)
                res.constraints.append(OccConstraint([g.id], False))
                adpr[(chain.name, resi)] = "{}/*".format(site)
    res.adpr_selections = list(adpr.values())
    if n_zero:
        res.notes.append("{} atoms have occupancy 0 in the input and are not given an occupancy group".format(n_zero))
    return res

def to_config_dict(res, adpr_weight=5.):
    """The refinement settings as a dict (the structure of the yaml given to --config)"""
    refine = {"atom_selection": {"occ": {"include": res.include, "exclude": []}},
              "occ_groups": [{"id": g.id, "selections": list(g.selections)} for g in res.groups],
              "occ_group_constraints": [{"ids": list(c.ids), "complete": bool(c.complete)} for c in res.constraints]}
    if adpr_weight and adpr_weight > 0 and res.adpr_selections:
        refine["local_adpr_weights"] = [{"sel": s, "w": float(adpr_weight)} for s in res.adpr_selections]
    return {"refine": refine}

def write_yaml(path, res, adpr_weight=5.):
    omegaconf.OmegaConf.save(omegaconf.OmegaConf.create(to_config_dict(res, adpr_weight)), path)

def apply_to_config(cfg, res, adpr_weight=5.):
    """Add the groups to the refine configuration (cfg = RefineConfig). Refuses to mix with groups of the user."""
    from servalcat.refine.refine import OccGroupItem, OccGroupConstItem, SelectionAndWeightItem
    if cfg.occ_groups or cfg.occ_group_constraints:
        raise SystemExit("Error: --refine_occ generates occupancy groups, but the configuration/keywords already define "
                         "occ_groups. Use one or the other (the generated groups can be written with "
                         "'servalcat util occ_groups' and edited).")
    for g in res.groups:
        cfg.occ_groups.append(OccGroupItem(g.id, list(g.selections)))
    for c in res.constraints:
        cfg.occ_group_constraints.append(OccGroupConstItem(list(c.ids), bool(c.complete)))
    cfg.atom_selection["occ"].include.extend(res.include)
    if adpr_weight and adpr_weight > 0:
        for s in res.adpr_selections:
            cfg.local_adpr_weights.append(SelectionAndWeightItem(s, float(adpr_weight)))

def _sites(res):
    """One entry per site (residue or water) with its groups: [(chain, resi, resname, kind, [groups], complete)]"""
    cons = {tuple(c.ids): c.complete for c in res.constraints}
    sites = {}
    for g in res.groups:
        sites.setdefault((g.chain, g.resi, g.kind), []).append(g)
    return [(chain, resi, gs[0].resname, kind, gs, cons.get(tuple(g.id for g in gs), False))
            for (chain, resi, kind), gs in sites.items()]

def _kind_text(kind, gs):
    if kind == "alt":
        return "conformers " + "/".join(g.alt for g in gs)
    return "partial occupancy" if kind == "partial" else "water"

def _rule_text(kind, complete):
    return "sum = 1" if kind == "alt" and complete else "sum <= 1"

def residue_table(res, max_rows=None):
    """Lines of a table: one row per residue with alternative conformations or partial occupancy"""
    head = " {:<5} {:<6} {:<9} {:<18} {:<16} {:<11} {}".format("chain", "resi", "residue", "kind", "input occupancy",
                                                           "constraint", "group id(s)")
    lines = [head]
    for i, (chain, resi, name, kind, gs, complete) in enumerate(_sites(res)):
        if max_rows and i >= max_rows:
            lines.append(" ... and {} more".format(len(_sites(res)) - max_rows))
            break
        lines.append(" {:<5} {:<6} {:<9} {:<18} {:<16} {:<11} {}".format(
            chain, resi, name[:9], _kind_text(kind, gs), " / ".join("{:.2f}".format(g.occ) for g in gs),
            _rule_text(kind, complete), ",".join(str(g.id) for g in gs)))
    return lines

def _mean_occ(st, group):
    occs = []
    for sel_str in group.selections:
        sel = gemmi.Selection(sel_str)
        for model in sel.models(st):
            for chain in sel.chains(model):
                for residue in sel.residues(chain):
                    occs.extend(a.occ for a in sel.atoms(residue))
    return sum(occs) / len(occs) if occs else float("nan")

def final_table(st, res, max_rows=40):
    """Lines: input -> refined occupancy of every site"""
    lines = [" {:<5} {:<6} {:<9} {:<18} {:<16} {:<16} {}".format("chain", "resi", "residue", "kind", "input", "refined", "sum")]
    sites = _sites(res)
    for i, (chain, resi, name, kind, gs, complete) in enumerate(sites):
        if i >= max_rows:
            lines.append(" ... and {} more".format(len(sites) - max_rows))
            break
        final = [_mean_occ(st, g) for g in gs]
        lines.append(" {:<5} {:<6} {:<9} {:<18} {:<16} {:<16} {:.2f}".format(
            chain, resi, name[:9], _kind_text(kind, gs), " / ".join("{:.2f}".format(g.occ) for g in gs),
            " / ".join("{:.2f}".format(v) for v in final), sum(final)))
    return lines

def log_summary(res, max_rows=40):
    n_alt_res = len({(g.chain, g.resi) for g in res.groups if g.kind == "alt"})
    logger.writeln("\nOccupancy groups found in the input model:")
    logger.writeln("  residues with alternative conformations: {}  ({} conformer groups)".format(n_alt_res, res.count("alt")))
    logger.writeln("  residues with partially occupied atoms:  {}".format(res.count("partial")))
    logger.writeln("  waters with refined occupancy:           {}".format(res.count("water")))
    if res.groups:
        for l in residue_table(res, max_rows):
            logger.writeln(l)
    for n in res.notes:
        logger.writeln("NOTE: " + n)
    logger.writeln("")

# ---------------------------------------------------------------------------------------------------
# servalcat util occ_groups

UTIL_DESCRIPTION = ("Find alternative conformations and partially occupied atoms in a model and write the occupancy "
                    "groups, constraints and ADP restraint weights that refine_xtal needs as a yaml file "
                    "(refine_xtal --config). The same is done in the refinement by 'refine_xtal --refine_occ'.")

def add_arguments(parser):
    import textwrap
    from servalcat.utils import cli
    add = cli.add_arg
    cli.setup_parser(parser, UTIL_DESCRIPTION, "--model FILE", "--model model.pdb")
    g = parser.add_argument_group(cli.REQUIRED)
    add(g, "--model", required=True, metavar="FILE", show="", help="Atomic model: PDB or mmCIF")
    g = parser.add_argument_group(cli.DEFAULTS)
    add(g, "--waters", choices=["none", "partial", "all"], default="partial",
        help="Waters that get an occupancy group: none; partial: waters with occupancy below 1; all")
    add(g, "--adpr_weight", type=float, default=5., metavar="W",
        help="Weight of the ADP restraints around the atoms with refined occupancy (0: no extra weight)")
    add(g, "-o", "--output_prefix", metavar="PREFIX", show="<model name>_occ", help="Prefix of the output file PREFIX.yaml")
    g = parser.add_argument_group("Options (off unless given)")
    add(g, "--alts_incomplete", action="store_true",
        help="The occupancies of the conformers of a residue sum to at most 1 (otherwise exactly 1)")

def main(args):
    from servalcat.utils import fileio
    st = fileio.read_structure(args.model)
    res = find_occupancy_groups(st, waters=args.waters, alts_complete=not args.alts_incomplete)
    log_summary(res, max_rows=1000)
    if not res.groups:
        logger.writeln("Nothing to write.")
        return
    path = args.output_prefix + ".yaml"
    write_yaml(path, res, args.adpr_weight)
    logger.writeln("Written: {}".format(path))
    logger.writeln("Use it with: refine_xtal --config {} --non_binary_solvent_mask --ncycle 30 ...".format(path))
    logger.writeln("(or let refine_xtal do all of this with --refine_occ)")
