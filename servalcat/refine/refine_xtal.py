"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import gemmi
import numpy
import pandas
import os
import time
import shutil
import argparse
import types
from servalcat.utils import logger
from servalcat import utils
from servalcat.xtal.sigmaa import decide_mtz_labels, process_input
from servalcat.refine.xtal import LL_Xtal
from servalcat.refine.refine import Geom, Refine, RefineParams, update_meta, print_h_options, load_config
from servalcat.refine import tls as tls_mod
from servalcat.refine import occ_groups
from servalcat.refine import summary
from servalcat.refmac import refmac_keywords
from servalcat import ext
b_to_u = utils.model.b_to_u

_add = utils.cli.add_arg

def add_arguments(parser, after_required=None, hklin_required=True):
    """Arguments are in three tiers: (1) required, (2) optional with default values, (3) optional features that are
    off unless given. after_required(parser): called before tiers 2 and 3 so that a program built on this one
    (tls_select) can list its own options first."""
    utils.cli.setup_parser(parser, "Restrained maximum-likelihood refinement of a crystallographic model against X-ray data "
                           "(isotropic or anisotropic ADPs, optional TLS).",
                           "--hklin FILE --model FILE [--monlib DIR]", "--model model.pdb --hklin data.mtz -o refined")

    # (1) required
    g = parser.add_argument_group(utils.cli.REQUIRED)
    if hklin_required:
        _add(g, "--hklin", required=True, metavar="FILE", show="",
             help="Reflection data: MTZ or mmCIF structure factors, merged or unmerged")
    _add(g, "--model", required=True, metavar="FILE", show="",
         help="Atomic model: PDB or mmCIF (also small-molecule .res/.ins/.cif)")
    _add(g, "--monlib", metavar="DIR", show="$CLIBD_MON; required if that variable is not set",
         help="Monomer library: directory with the CCP4 monomer cif dictionaries")
    if after_required:
        after_required(parser)

    # (2) optional, with default values
    g = parser.add_argument_group(utils.cli.DEFAULTS)
    _add(g, "-s", "--source", choices=["electron", "xray", "neutron", "custom"], default="xray",
         help="Radiation source")
    _add(g, "--labin", metavar="COLUMNS", show="F,SIGF,FreeR_flag if present, otherwise chosen by column type",
         help="MTZ columns of observations, sigmas and test flags, e.g. FP,SIGFP,FreeR_flag or I,SIGI,FreeR_flag")
    _add(g, "--free", type=int, metavar="N", show="guessed from the flags",
         help="Value of the test-set flag")
    _add(g, "-d", "--d_min", type=float, metavar="A", show="all data",
         help="High-resolution limit in Angstrom")
    _add(g, "--d_max", type=float, metavar="A", show="all data",
         help="Low-resolution limit in Angstrom")
    _add(g, "--nbins", type=int, metavar="N", show="auto",
         help="Number of resolution bins for statistics")
    _add(g, "--nbins_ml", type=int, metavar="N", show="auto",
         help="Number of resolution bins for ML parameters")
    _add(g, "--use_in_est", choices=["all", "work", "test"], show="test; work with --twin",
         help="Reflections used to estimate the ML parameters")
    _add(g, "--ncycle", type=int, default=10, metavar="N",
         help="Number of refinement cycles")
    _add(g, "--adp", choices=["fix", "iso", "aniso"], default="iso",
         help="ADP model; with --tls the residual B stays isotropic")
    _add(g, "--weight", type=float, metavar="W", show="auto",
         help="Weight of the geometry restraints relative to the data")
    _add(g, "--target_bond_rmsz_range", nargs=2, type=float, default=[0.5, 1.], metavar=("MIN", "MAX"),
         help="Bond rmsZ range that the automatic weight adjustment aims for")
    _add(g, "--hydrogen", default="all", choices=["all", "yes", "no"],
         help="all: add riding hydrogen atoms; yes: use hydrogen atoms if present in the model; "
         "no: remove hydrogen atoms")
    _add(g, "--adpr_weight", type=float, default=1., metavar="W",
         help="Weight of the ADP restraints")
    _add(g, "--occr_weight", type=float, default=0., metavar="W",
         help="Weight of the occupancy restraints")
    _add(g, "--adp_restraint_mode", choices=["diff", "kldiv"], default="kldiv",
         help="Type of the ADP restraint")
    _add(g, "--max_dist_for_adp_restraint", type=float, default=4., metavar="A",
         help="Largest distance between atoms with an ADP restraint, in Angstrom")
    _add(g, "--adp_restraint_power", type=float, metavar="P", show="4",
         help="Power of the distance dependence of the ADP restraint weights")
    _add(g, "--adp_restraint_exp_fac", type=float, metavar="F", show="0.011271",
         help="Factor in the exponent of the distance dependence of the ADP restraint weights")
    _add(g, "--jellybody_params", nargs=2, type=float, metavar=("SIGMA", "DMAX"), default=[0.01, 4.2],
         help="Sigma and maximum distance of the jelly-body restraints, used with --jellybody")
    _add(g, "--occ_waters", choices=["none", "partial", "all"], default="partial",
         help="Waters whose occupancy is refined with --refine_occ: none; partial: waters with occupancy below 1; all")
    _add(g, "--occ_adpr_weight", type=float, default=5., metavar="W",
         help="Weight of the ADP restraints around the atoms with refined occupancy, with --refine_occ (0: no extra weight)")
    _add(g, "--tls_ncycle", type=int, metavar="N", show="1",
         help="Number of TLS steps in each refinement cycle, used with --tls")
    _add(g, "--seq_min_identity", type=float, default=0.8, metavar="X",
         help="Fraction of the modelled residues that must agree with a sequence for it to be assigned, "
         "used with --seq")
    _add(g, "--randomize", type=float, default=0, metavar="RMSD", show="0, no shaking",
         help="Shake the coordinates with this rmsd in Angstrom before refinement")
    _add(g, "-o", "--output_prefix", metavar="PREFIX", show="<model name>_refined",
         help="Prefix of the output files")

    # (3) optional features, off unless given
    g = parser.add_argument_group("Refinement options (off unless given)")
    _add(g, "--fix_xyz", action="store_true", help="Do not refine the coordinates")
    _add(g, "--refine_occ", action="store_true",
         help="Refine the occupancies of alternative conformations and partially occupied atoms: the occupancy groups "
         "and constraints are generated from the input model (written to PREFIX_occ.yaml; see docs/occ_refine.rst)")
    _add(g, "--refine_all_occ", action="store_true",
         help="Refine the occupancy of every atom individually (not grouped; usually unstable)")
    _add(g, "--refine_h", action="store_true", help="Refine hydrogen positions (otherwise restraints only)")
    _add(g, "--refine_dfrac", action="store_true", help="Refine the deuterium fraction (neutron data only)")
    _add(g, "--twin", action="store_true", help="Twin refinement")
    _add(g, "--twin_mlalpha", action="store_true", help="Optimise the twin fractions by ML")
    _add(g, "--unrestrained", action="store_true", help="No geometry restraints (no monomer library needed)")
    _add(g, "--ncsr", action="store_true", help="Local NCS restraints")
    _add(g, "--jellybody", action="store_true", help="Jelly-body restraints")
    _add(g, "--jellyonly", action="store_true", help="Jelly-body restraints instead of geometry restraints (experimental)")
    _add(g, "--adaptive_restraint", type=float, metavar="ALPHA", show="off",
         help="Scale the target of bond, angle, plane, torsion and vdW restraints by 1 + ALPHA * z^2")
    _add(g, "--no_weight_adjust", action="store_true", help="Do not adjust the restraint weight during refinement")
    _add(g, "--no_solvent", action="store_true", help="No bulk-solvent contribution")
    _add(g, "--bfactor", type=float, metavar="B", show="keep the B-values of the model",
         help="Reset all atomic B-values to this value before refinement")
    _add(g, "--find_links", action="store_true", help="Find and add links such as metal coordination")
    _add(g, "--hout", action="store_true", help="Write hydrogen atoms in the output model")
    _add(g, "--adp_restraint_no_long_range", action="store_true",
         help="ADP restraints between bonded atoms and 1-3 neighbours only")

    g = parser.add_argument_group("TLS refinement (off unless given)")
    _add(g, "--tls", action="store_true",
         help="TLS refinement with isotropic residual B; the groups come from refine.tls.groups in --config, "
         "or from the input model")
    _add(g, "--tls_per_chain", action="store_true", help="TLS refinement with one group per chain, if no groups are defined")
    _add(g, "--tls_file", metavar="FILE", show="none",
         help="TLS definition (yaml) from a previous run (PREFIX_tls.yaml) or from 'util tls_select'; "
         "use it to continue after editing the model")

    g = parser.add_argument_group("Sequence annotation (off unless given)")
    _add(g, "--seq", nargs="*", action="append", metavar="FASTA", show="none",
         help="Sequence file(s) in FASTA format: the best-matching sequence is assigned to each protein, DNA and RNA "
         "chain for SEQRES and label_seq_id, and residues that differ from it are reported")

    g = parser.add_argument_group("Additional inputs (off unless given)")
    if not hklin_required:
        _add(g, "--hklin", metavar="FILE", show="none",
             help="Reflection data: MTZ or mmCIF structure factors (merged or unmerged)")
    _add(g, "--ligand", nargs="*", action="append", metavar="CIF", show="none",
         help="Restraint dictionary file(s) for ligands that are not in the monomer library")
    _add(g, "--config", metavar="YAML", show="none", help="Configuration file with all refinement settings")
    _add(g, "--keywords", nargs="+", action="append", metavar="KEYWORD", show="none",
         help="REFMAC-style keyword(s), e.g. external restraints")
    _add(g, "--keyword_file", nargs="+", action="append", metavar="FILE", show="none",
         help="File(s) with REFMAC-style keywords")
    _add(g, "--hklin_free", metavar="FILE", show="the flags are taken from --hklin",
         help="MTZ file with the test-set flags")
    _add(g, "--labin_free", metavar="COLUMN", show="auto", help="Column of the flags in --hklin_free")
    _add(g, "--wavelength", type=float, metavar="A", show="from the data file if present",
         help="Wavelength for anomalous scattering factors f', f'' (X-ray); 0: ignore the wavelength in the file")

    g = parser.add_argument_group("Advanced options (off unless given)")
    _add(g, "--prefer_intensity", action="store_true", help="When choosing the columns automatically, prefer intensities")
    _add(g, "--prefer_anomalous", action="store_true", help="When choosing the columns automatically, prefer Friedel pairs")
    _add(g, "--keep_entities", action="store_true", help="Keep the entity and SEQRES information of the input")
    _add(g, "--keep_charges", action="store_true", help="Use scattering factors of charged atoms (use with care)")
    _add(g, "--allow_unusual_occupancies", action="store_true", help="Allow negative occupancies or occupancies above 1")
    _add(g, "--newligand_continue", action="store_true",
         help="Make ad-hoc restraints for unknown ligands (not recommended)")
    _add(g, "--write_trajectory", action="store_true", help="Write the model of every cycle")
    _add(g, "--vonmises", action="store_true", help="Von Mises angle restraints (experimental)")
    _add(g, "--use_fw", action="store_true",
         help="Debugging: use French-Wilson amplitudes, but intensities for the statistics")
    # not shown: testing options
    parser.add_argument('--labin_llweight', help=argparse.SUPPRESS)
    parser.add_argument("--non_binary_solvent_mask", action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--robust_function', help=argparse.SUPPRESS)
# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def set_prefix(args):
    if not args.output_prefix:
        args.output_prefix = utils.fileio.splitext(os.path.basename(args.model))[0] + "_refined"
# set_prefix()

def setup(args, use_tls=True):
    """Read data and model, prepare restraints, ML target and the Refine object (everything before the cycles)
    Returns a namespace; also used by util tls_select"""
    if args.refine_dfrac and args.source != "neutron": # TODO should check params.is_refined()
        raise SystemExit("--refine_dfrac can only be used for the neutron source")
    if args.labin_llweight and args.twin:
        raise SystemExit("--labin_llweight not supported for twin refinement")
    if args.wavelength is not None and args.source != "xray":
        raise SystemExit("Error: Wavelength is only available for X-ray source")
    if args.ligand: args.ligand = sum(args.ligand, [])
    set_prefix(args)
    # This could be confusing. Twinning may not be detected.
    if not args.use_in_est:
        args.use_in_est = "work" if args.twin else "test"
    refmackwds = refmac_keywords.RefmacKeywords(args.keywords, args.keyword_file)
    refine_cfg = load_config(args.config, args, refmackwds)
    refmackwds.dump_exte_blocks(args.output_prefix + "_exte.json")
    hklin = args.hklin
    labin = args.labin
    if labin is not None:
        labin = labin.split(",")
    elif utils.fileio.is_mmhkl_file(hklin):
        hklin = utils.fileio.read_mmhkl(hklin)
        labin = decide_mtz_labels(hklin, prefer_intensity=args.prefer_intensity,
                                  prefer_anomalous=args.prefer_anomalous)
    software_items = utils.fileio.software_items_from_mtz(hklin)
    try:
        hkldata, st, fc_labs, args.free, use_in_est = process_input(
            hklin=hklin,
            labin=labin,
            n_bins_ml=args.nbins_ml,
            n_bins_stat=args.nbins,
            free=args.free,
            xyzin=args.model,
            d_max=args.d_max,
            d_min=args.d_min,
            use=args.use_in_est,
            max_mlbins=30,
            keep_charges=args.keep_charges,
            allow_unusual_occupancies=args.allow_unusual_occupancies,
            hklin_free=args.hklin_free,
            labin_free=args.labin_free,
            labin_llweight=args.labin_llweight)
    except RuntimeError as e:
        raise SystemExit("Error: {}".format(e))

    if args.source != "xray":
        if hkldata.wavelength:
            logger.writeln(f"Ignoring wavelength in the input file")
        hkldata.wavelength = None
    elif args.wavelength == 0:
        logger.writeln(f"Ignoring wavelength as --wavelength=0 was given")
        hkldata.wavelength = None
    elif args.wavelength:
        if hkldata.wavelength and abs(hkldata.wavelength - args.wavelength) > 0.0001:
            logger.writeln(f"Updating wavelength using --wavelength={args.wavelength}")
        hkldata.wavelength = args.wavelength
    
    if "FREE" in hkldata.df:
        use_in_target = "work"
    else:
        use_in_target = "all"

    is_int = "I" in hkldata.df
    if args.use_fw:
        if not is_int:
            raise SystemExit("Error: need intensity input when -use_fw")
        logger.writeln("Converting intensities to amplitudes with the French & Wilson algorithm")
        from servalcat.xtal import french_wilson
        B_aniso = french_wilson.determine_Sigma_and_aniso(hkldata)
        french_wilson.french_wilson(hkldata, B_aniso, labout=["FP", "SIGFP"]) # TODO when anomalous
        del hkldata.binned_df["ml"]["S"]
        is_int = False
        
    utils.model.fix_deuterium_residues(st)
    seq_files = sum(args.seq, []) if args.seq else []
    seq_records = None
    seq_report = None
    if seq_files:
        try:
            seq_records = utils.model.read_sequence_records(seq_files)
        except (RuntimeError, OSError) as e:
            raise SystemExit("Error: {}".format(e))
        if not seq_records:
            raise SystemExit("Error: no sequences found in {}".format(", ".join(seq_files)))
        if not 0 < args.seq_min_identity <= 1:
            raise SystemExit("Error: --seq_min_identity must be in (0, 1]")
    use_nucleus = args.source in ("neutron", "electron")
    if args.unrestrained:
        monlib = gemmi.MonLib()
        topo = None
        if args.hydrogen == "yes":
            h_change = gemmi.HydrogenChange.NoChange
        else:
            h_change = gemmi.HydrogenChange.Remove
            st.remove_hydrogens()
            if args.hydrogen == "all":
                logger.writeln("\nWARNING: in unrestrained refinement hydrogen atoms are not generated.\n")
        for i, cra in enumerate(st[0].all()):
            cra.atom.serial = i + 1
        if seq_records and not args.keep_entities:
            seq_report = utils.model.setup_entities(st, clear=True, force_subchain_names=True, overwrite_entity_type=True,
                                                    fix_sequences=True, sequences=seq_records,
                                                    min_seq_identity=args.seq_min_identity)
            utils.model.log_sequence_report(seq_report, logger, seq_files)
    else:
        try:
            monlib = utils.restraints.load_monomer_library(st, monomer_dir=args.monlib, cif_files=args.ligand,
                                                           stop_for_unknowns=not args.newligand_continue,
                                                           refmackwds=refmackwds)
        except RuntimeError as e:
            raise SystemExit("Error: {}".format(e))
        if not args.keep_entities:
            seq_report = utils.model.setup_entities(st, clear=True, force_subchain_names=True, overwrite_entity_type=True,
                                                    fix_sequences=True, sequences=seq_records,
                                                    min_seq_identity=args.seq_min_identity)
        elif seq_records:
            seq_report = utils.model.assign_sequences(st, seq_records, args.seq_min_identity)
            for chain in st[0]:
                for res in chain:
                    res.label_seq = None
            st.assign_label_seq_id()
            utils.model.set_labels(st, seq_report)
            utils.model.analyse_sequences(st, seq_report)
        if seq_report is not None:
            utils.model.log_sequence_report(seq_report, logger, seq_files)
        utils.restraints.find_and_fix_links(st, monlib, find_metal_links=args.find_links,
                                            add_found=args.find_links)
        h_change = {"all":gemmi.HydrogenChange.ReAddKnown,
                    "yes":gemmi.HydrogenChange.NoChange,
                    "no":gemmi.HydrogenChange.Remove}[args.hydrogen]
        try:
            topo, _ = utils.restraints.prepare_topology(st, monlib, h_change=h_change,
                                                        check_hydrogen=(args.hydrogen=="yes"),
                                                        refmackwds=refmackwds)
        except RuntimeError as e:
            raise SystemExit("Error: {}".format(e))

        if h_change == gemmi.HydrogenChange.ReAddKnown and use_nucleus:
            topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.Nucleus,
                                           default_scale=utils.restraints.default_proton_scale)

    print_h_options(h_change, st[0].has_hydrogen(), args.refine_h, args.hout, geom_only=False)
    
    ccu = utils.model.CustomCoefUtil()
    if args.source == "custom":
        ccu.read_from_cif(st, args.model)
        ccu.show_info()
        addends, addends2 = None, None
    else:
        addends, addends2 = utils.model.check_atomsf(st, args.source, mott_bethe=(args.source=="electron"), wavelength=hkldata.wavelength)

    # initialize values
    utils.model.reset_adp(st[0], args.bfactor, args.adp)
    utils.model.initialize_values(st[0], refine_cfg.initialisation)

    # occupancy groups from the alternative conformations and partial occupancies of the input model
    occ_result = None
    if args.refine_occ:
        occ_result = occ_groups.find_occupancy_groups(st, waters=args.occ_waters)
        occ_groups.log_summary(occ_result)
        if occ_result.groups:
            occ_groups.apply_to_config(refine_cfg, occ_result, args.occ_adpr_weight)
            occ_yaml = args.output_prefix + "_occ.yaml"
            occ_groups.write_yaml(occ_yaml, occ_result, args.occ_adpr_weight)
            logger.writeln("Occupancy groups written to {} (can be edited and used with --config)".format(occ_yaml))
            if not args.no_solvent and not args.non_binary_solvent_mask:
                logger.writeln("NOTE: using the non-binary solvent mask, as advised when refining occupancies")
                args.non_binary_solvent_mask = True
            if args.ncycle < 30:
                logger.writeln("NOTE: group occupancies usually need 30 cycles or more to converge; --ncycle is {}".format(args.ncycle))
        else:
            logger.writeln("No alternative conformations or partially occupied atoms found: no occupancy refinement")

    # auto weight
    if args.weight is None:
        logger.writeln("Estimating weight using resolution")
        reso = hkldata.d_min_max()[0]
        args.weight = numpy.exp(reso * 0.9104 + 0.2162)
        logger.writeln(" Will use weight= {:.2f}".format(args.weight))
        
    if args.ncsr:
        ncslist = utils.restraints.prepare_ncs_restraints(st)
    else:
        ncslist = False
    if args.refine_dfrac:
        # needed to make Fc calculation expand H/D (ugh)
        st.has_d_fraction = True
    refine_params = RefineParams(st, refine_xyz=not args.fix_xyz,
                                 adp_mode=dict(fix=0, iso=1, aniso=2)[args.adp],
                                 refine_occ=args.refine_all_occ,
                                 refine_dfrac=args.refine_dfrac, cfg=refine_cfg,
                                 exclude_h_ll=not args.refine_h)
    geom = Geom(st, topo, monlib, refine_params, refine_cfg,
                shake_rms=args.randomize, adpr_w=args.adpr_weight, occr_w=args.occr_weight, refmackwds=refmackwds,
                unrestrained=args.unrestrained or args.jellyonly, use_nucleus=use_nucleus,
                ncslist=ncslist)
    geom.geom.angle_von_mises = args.vonmises
    geom.geom.adpr_max_dist = args.max_dist_for_adp_restraint
    if args.adp_restraint_power is not None: geom.geom.adpr_d_power = args.adp_restraint_power
    if args.adp_restraint_exp_fac is not None: geom.geom.adpr_exp_fac = args.adp_restraint_exp_fac
    if args.adp_restraint_no_long_range: geom.geom.adpr_long_range = False
    geom.geom.adpr_mode = args.adp_restraint_mode
    if args.jellybody or args.jellyonly:
        geom.geom.ridge_sigma, geom.geom.ridge_dmax = args.jellybody_params
    if args.jellyonly: geom.geom.ridge_exclude_short_dist = False
    if args.source == "custom":
        ccu.set_coeffs(st)
    if args.robust_function:
        assert is_int and not args.twin
        ft, p = args.robust_function.split(",")
        p = float(p)
        if ft == "huber":
            #p = 1.345
            robust_func = lambda r: numpy.where(r < p, 1, p / r)
            logger.writeln(f"Will use Huber (cH= {p})")
        elif ft == "tukey":
            #p = 4.685
            robust_func = lambda r: numpy.where(r < p, (1-(r/p)**2)**2, 0)
            logger.writeln(f"Will use Tukey (cT= {p})")
        else:
            raise ValueError("unknown function type")
    else:
        robust_func = None
    tls = None
    if use_tls and (args.tls or args.tls_per_chain or refine_cfg.tls.groups):
        if args.adp == "aniso":
            raise SystemExit("Error: TLS refinement uses an isotropic residual B; use --adp iso (or --adp fix) "
                             "instead of --adp aniso")
        tls = tls_mod.setup_tls(st, refine_cfg.tls, per_chain=args.tls_per_chain, ncycle=args.tls_ncycle,
                                specials=geom.geom.specials, exclude_h_ll=not args.refine_h,
                                model_path=args.model)
    ll = LL_Xtal(hkldata, args.free, st, monlib, source=args.source,
                 use_solvent=0 if args.no_solvent else 2 if args.non_binary_solvent_mask else 1,
                 use_in_est=use_in_est, use_in_target=use_in_target,
                 twin=args.twin, twin_mlalpha=args.twin_mlalpha,
                 addends=addends, addends2=addends2, is_int=is_int,
                 robust_func=robust_func)
    ll.tls = tls
    refiner = Refine(st, geom, refine_cfg, refine_params, ll=ll,
                     unrestrained=args.unrestrained, tls=tls)
    return types.SimpleNamespace(refiner=refiner, refine_cfg=refine_cfg, software_items=software_items,
                                 ll=ll, st=st, hkldata=hkldata, is_int=is_int, tls=tls, geom=geom,
                                 refine_params=refine_params, monlib=monlib, use_in_target=use_in_target,
                                 seq_report=seq_report, occ_result=occ_result)
# setup()

def main(args):
    t_start = time.time()
    ctx = setup(args)
    refiner, refine_cfg, software_items = ctx.refiner, ctx.refine_cfg, ctx.software_items
    ll, st, hkldata, is_int, tls = ctx.ll, ctx.st, ctx.hkldata, ctx.is_int, ctx.tls
    stats = refiner.run_cycles(args.ncycle, weight=args.weight,
                               weight_adjust=not args.no_weight_adjust,
                               weight_adjust_bond_rmsz_range=args.target_bond_rmsz_range,
                               stats_json_out=args.output_prefix + "_stats.json")
    if tls is not None:
        tls.finalize() # B values in the output are those of the total U
    update_meta(st, stats[-1], ll)
    if tls is not None:
        tls_mod.update_meta_tls(st, tls)
        tls_mod.write_tls_config(args.output_prefix + "_tls.yaml", st, tls)
    st.meta.software = software_items + st.meta.software
    refiner.st.name = args.output_prefix
    utils.fileio.write_model(refiner.st, args.output_prefix, pdb=True, cif=True, hout=args.hout)
    if st.has_d_fraction: # XXX make hout when neutron
        st_hd_expand = refiner.st.clone()
        st_hd_expand.store_deuterium_as_fraction(False)
        utils.fileio.write_model(st_hd_expand, args.output_prefix + "_hd_expand", pdb=True, cif=True, hout=True)
    if refine_cfg.write_trajectory:
        utils.fileio.write_model(refiner.st_traj, args.output_prefix + "_traj", cif=True)

    # Write mtz file
    if ll.twin_data:
        labs = ["F_est", "F_exp"]
    elif is_int:
        labs = ["I", "SIGI", "F_est"]
    else:
        labs = ["FP", "SIGFP"]
    labs.extend(["FOM", "FWT", "FWT_nofill", "DELFWT", "FC"])
    if "FAN" in hkldata.df:
        labs.append("FAN")
    if "DELFAN" in hkldata.df:
        labs.append("DELFAN")
    if not args.no_solvent:
        labs.append("FCbulk")
    if "FREE" in hkldata.df:
        labs.append("FREE")
    if args.labin_llweight:
        labs.append("llweight")
    labs += ll.D_labs + ["S"] # for debugging, for now
    if args.robust_function:
        labs += ["robustweight", "intensity_z"]
    mtz_out = args.output_prefix+".mtz"
    hkldata.write_mtz(mtz_out, labs=labs, types={"FOM": "W", "FP":"F", "SIGFP":"Q", "I":"J", "SIGI":"Q", "F_est": "F", "F_exp": "F", "llweight": "R"})

    # summary of the run (the log of the refinement above is unchanged)
    p = args.output_prefix
    summary.final_summary(args, refiner.st, stats, seq_report=ctx.seq_report,
                          seq_files=sum(args.seq, []) if args.seq else [], occ_result=ctx.occ_result,
                          files=[("model (pdb)", p + ".pdb"), ("model (mmcif)", p + ".mmcif"), ("map coefficients", mtz_out),
                                 ("statistics", p + "_stats.json"), ("TLS groups", p + "_tls.yaml"),
                                 ("occupancy groups", p + "_occ.yaml"), ("log", getattr(args, "logfile", None))],
                          elapsed=time.time() - t_start)

# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
