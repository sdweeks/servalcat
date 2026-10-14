"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import argparse
import os
import gemmi
import numpy
import json
import servalcat # for version
from servalcat.utils import logger
from servalcat import utils
from servalcat.refine.refine import RefineParams, Geom, Refine, convert_stats_to_dicts, update_meta, print_h_options, load_config
from servalcat.refmac import refmac_keywords

def add_arguments(parser):
    add = utils.cli.add_arg
    utils.cli.setup_parser(parser, "Geometry optimisation of an atomic model, or of the coordinates of a ligand "
                           "restraint dictionary, with the restraints of the monomer library (no diffraction data).",
                           "(--model FILE | --update_dictionary FILE) [--monlib DIR]", "--model model.pdb -o regularised")

    g = parser.add_argument_group(utils.cli.REQUIRED)
    one_of = g.add_mutually_exclusive_group(required=True)
    add(one_of, "--model", metavar="FILE", show="", help="Atomic model: PDB or mmCIF (one of --model and --update_dictionary is required)")
    add(one_of, "--update_dictionary", metavar="FILE", show="",
        help="Ligand restraint dictionary (cif) whose ideal coordinates are updated")
    add(g, "--monlib", metavar="DIR", show="$CLIBD_MON; required if that variable is not set",
        help="Monomer library: directory with the CCP4 monomer cif dictionaries")

    g = parser.add_argument_group(utils.cli.DEFAULTS)
    add(g, "--ncycle", type=int, default=10, metavar="N",
        help="Number of cycles, with --model")
    add(g, "--ncycle_update", type=int, nargs=2, default=[10, 30], metavar=("N1", "N2"),
        help="Number of cycles of the first and the second step, with --update_dictionary")
    add(g, "--hydrogen", default="all", choices=["all", "yes", "no"],
        help="all: add riding hydrogen atoms; yes: use hydrogen atoms if present in the model; no: remove hydrogen atoms")
    add(g, "--randomize", type=float, default=0, metavar="RMSD", show="0, no shaking",
        help="Shake the coordinates with this rmsd in Angstrom before refinement")
    add(g, "-o", "--output_prefix", metavar="PREFIX", show="<input file name>_refined",
        help="Prefix of the output files")

    g = parser.add_argument_group("Refinement options (off unless given)")
    add(g, "--ncsr", action="store_true", help="Local NCS restraints")
    add(g, "--find_links", action="store_true", help="Find and add links such as metal coordination")

    g = parser.add_argument_group("Additional inputs (off unless given)")
    add(g, "--ligand", nargs="*", action="append", metavar="CIF", show="none",
        help="Restraint dictionary file(s) for ligands that are not in the monomer library")
    add(g, "--config", metavar="YAML", show="none", help="Configuration file with all refinement settings")
    add(g, "--keywords", nargs="+", action="append", metavar="KEYWORD", show="none",
        help="REFMAC-style keyword(s), e.g. restraint weights")
    add(g, "--keyword_file", nargs="+", action="append", metavar="FILE", show="none",
        help="File(s) with REFMAC-style keywords")

    g = parser.add_argument_group("Advanced options (off unless given)")
    add(g, "--write_trajectory", action="store_true", help="Write the model of every cycle")
    parser.add_argument('--adp', choices=["fix"], default="fix", help=argparse.SUPPRESS) # dummy for omegaconf
# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def add_program_info_to_dictionary(block, comp_id, program_name="servalcat", descriptor="optimization tool"):
    # old acedrg used _pdbx_chem_comp_description_generator. and descriptor
    # new acedrg (>280?) uses _acedrg_chem_comp_descriptor. and type
    for tag, name in (("_acedrg_chem_comp_descriptor.", "type"),
                      ("_pdbx_chem_comp_description_generator.", "descriptor")):
        tab = block.find(tag, ["program_name", "program_version", name])
        if tab:
            loop = tab.loop
            # just overwrite version if it's there
            for row in tab:
                if row.str(0) == program_name and row.str(2) == descriptor:
                    row[1] = gemmi.cif.quote(servalcat.__version__)
                    return
            break
    else:
        # it may be strange to say _acedrg in this case..
        name = "type"
        loop = block.init_loop("_acedrg_chem_comp_descriptor.", ["comp_id",
                                                                 "program_name",
                                                                 "program_version",
                                                                 name])
    tags = [x[x.index(".")+1:] for x in loop.tags]
    row = ["" for _ in range(len(tags))]
    for tag, val in (("comp_id", comp_id),
                     ("program_name", program_name),
                     ("program_version", servalcat.__version__),
                     (name, descriptor)):
        if tag in tags: row[tags.index(tag)] = val
    loop.add_row(gemmi.cif.quote_list(row))
# add_program_info_to_dictionary()

def refine_and_update_dictionary(cif_in, monomer_dir, output_prefix, refine_cfg, randomize=0, ncycle1=10, ncycle2=30):
    doc = gemmi.cif.read(cif_in)
    for block in doc: # this block will be reused below
        st = gemmi.make_structure_from_chemcomp_block(block)
        if len(st) > 0: break
    else:
        raise SystemExit("No model in the cif file")
    for i in range(len(st)-1):
        del st[1]
    try:
        monlib = utils.restraints.load_monomer_library(st, monomer_dir=monomer_dir, # monlib is needed for ener_lib
                                                       cif_files=[cif_in],
                                                       stop_for_unknowns=True)
    except RuntimeError as e:
        raise SystemExit("Error: {}".format(e))
    all_stats = []
    for i_macro in 0, 1:
        try:
            topo, _ = utils.restraints.prepare_topology(st, monlib, h_change=[gemmi.HydrogenChange.Remove, gemmi.HydrogenChange.ReAdd][i_macro],
                                                        check_hydrogen=(i_macro == 1))
        except RuntimeError as e:
            raise SystemExit("Error: {}".format(e))

        refine_params = RefineParams(st, refine_xyz=True)
        geom = Geom(st, topo, monlib, refine_params, refine_cfg, shake_rms=randomize)
        refiner = Refine(st, geom, refine_cfg, refine_params)
        logger.writeln("Running {} cycles with wchir=4 wvdw=2 {} hydrogen".format(ncycle1, ["without","with"][i_macro]))
        geom.calc_kwds["wchir"] = 4
        geom.calc_kwds["wvdw"] = 2
        all_stats.append(refiner.run_cycles(ncycle1))

        logger.writeln("Running {} cycles with wchir=1 wvdw=2 {} hydrogen".format(ncycle2, ["without","with"][i_macro]))
        geom.calc_kwds["wchir"] = 1
        geom.calc_kwds["wvdw"] = 2
        all_stats.append(refiner.run_cycles(ncycle2))

    # replace xyz
    pos = {cra.atom.name: cra.atom.pos.tolist() for cra in refiner.st[0].all()}
    for row in block.find("_chem_comp_atom.", ["atom_id", "?x", "?y", "?z",
                                               "?pdbx_model_Cartn_x_ideal",
                                               "?pdbx_model_Cartn_y_ideal",
                                               "?pdbx_model_Cartn_z_ideal",
                                               "?model_Cartn_x", "?model_Cartn_y", "?model_Cartn_z"]):
        p = pos[row.str(0)]
        for i in range(3):
            if row.has(i+1):
                row[i+1] = "{:.3f}".format(p[i])
            if row.has(i+4):
                row[i+4] = "{:.3f}".format(p[i])
            if row.has(i+7):
                row[i+7] = "{:.3f}".format(p[i])
    # add description
    add_program_info_to_dictionary(block, st[0][0][0].name)
    doc.write_file(output_prefix + "_updated.cif", options=gemmi.cif.Style.Aligned)
    logger.writeln("Updated dictionary saved: {}".format(output_prefix + "_updated.cif"))
    with open(output_prefix + "_stats.json", "w") as ofs:
        json.dump([convert_stats_to_dicts(x) for x in all_stats],
                  ofs, indent=2)
        logger.writeln("Refinement statistics saved: {}".format(ofs.name))
# refine_and_update_dictionary()

def refine_geom(model_in, monomer_dir, cif_files, h_change, ncycle, output_prefix, randomize, refmackwds,
                refine_cfg, find_links=False, use_ncsr=False):
    st = utils.fileio.read_structure(model_in)
    utils.model.setup_entities(st, clear=True, force_subchain_names=True, overwrite_entity_type=True,
                               fix_sequences=True)
    if not all(op.given for op in st.ncs):
        st2 = st.clone()
        logger.writeln("Take NCS constraints into account.")
        st2.expand_ncs(gemmi.HowToNameCopiedChain.Dup, merge_dist=0)
        utils.fileio.write_model(st2, file_name="input_expanded.pdb")
    try:
        monlib = utils.restraints.load_monomer_library(st, monomer_dir=monomer_dir,
                                                       cif_files=cif_files,
                                                       stop_for_unknowns=True,
                                                       refmackwds=refmackwds)
    except RuntimeError as e:
        raise SystemExit("Error: {}".format(e))
    utils.restraints.find_and_fix_links(st, monlib, find_metal_links=find_links,
                                        add_found=find_links) # should remove unknown id here?
    try:
        topo, _ = utils.restraints.prepare_topology(st, monlib, h_change=h_change,
                                                    check_hydrogen=(h_change==gemmi.HydrogenChange.NoChange),
                                                    refmackwds=refmackwds)
    except RuntimeError as e:
        raise SystemExit("Error: {}".format(e))

    print_h_options(h_change, st[0].has_hydrogen(), refine_h=True, hout=True, geom_only=True)
        
    if use_ncsr:
        ncslist = utils.restraints.prepare_ncs_restraints(st)
    else:
        ncslist = False
    refine_params = RefineParams(st, refine_xyz=True, cfg=refine_cfg)
    geom = Geom(st, topo, monlib, refine_params, refine_cfg, shake_rms=randomize, refmackwds=refmackwds, ncslist=ncslist)
    refiner = Refine(st, geom, refine_cfg, refine_params)
    stats = refiner.run_cycles(ncycle,
                               stats_json_out=output_prefix + "_stats.json")
    update_meta(st, stats[-1])
    refiner.st.name = output_prefix
    utils.fileio.write_model(refiner.st, output_prefix, pdb=True, cif=True)
    if refine_cfg.write_trajectory:
        utils.fileio.write_model(refiner.st_traj, output_prefix + "_traj", cif=True)
# refine_geom()

def set_prefix(args):
    if not args.output_prefix:
        xyzin = args.model if args.model else args.update_dictionary
        args.output_prefix = utils.fileio.splitext(os.path.basename(xyzin))[0] + "_refined"
# set_prefix()

def main(args):
    refmackwds = refmac_keywords.RefmacKeywords(args.keywords, args.keyword_file)
    refine_cfg = load_config(args.config, args, refmackwds)
    refmackwds.dump_exte_blocks(args.output_prefix + "_exte.json")
    set_prefix(args)
    if args.model:
        if args.ligand:
            args.ligand = sum(args.ligand, [])
        h_change = {"all":gemmi.HydrogenChange.ReAddButWater,
                    "full":gemmi.HydrogenChange.ReAdd,
                    "yes":gemmi.HydrogenChange.NoChange,
                    "no":gemmi.HydrogenChange.Remove}[args.hydrogen]
        refine_geom(model_in=args.model,
                    monomer_dir=args.monlib,
                    cif_files=args.ligand,
                    h_change=h_change,
                    ncycle=args.ncycle,
                    output_prefix=args.output_prefix,
                    randomize=args.randomize,
                    refmackwds=refmackwds,
                    refine_cfg=refine_cfg,
                    find_links=args.find_links,
                    use_ncsr=args.ncsr)
    else:
        if args.ligand:
            logger.writeln("WARNING: monlib and ligand are ignored in the dictionary updating mode")
        if args.keywords or args.keyword_file:
            logger.writeln("WARNING: refmac keywords are ignored in the dictionary updating mode")
        refine_and_update_dictionary(cif_in=args.update_dictionary,
                                     monomer_dir=args.monlib,
                                     output_prefix=args.output_prefix,
                                     refine_cfg=refine_cfg,
                                     randomize=args.randomize,
                                     ncycle1=args.ncycle_update[0],
                                     ncycle2=args.ncycle_update[1])

# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
