"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import argparse
import sys
import traceback
import platform
import gemmi
import numpy
import scipy
import pandas
import servalcat.xtal.sigmaa
import servalcat.xtal.french_wilson
import servalcat.utils.commands
import servalcat.refine.refine_geom
import servalcat.refine.refine_xtal

from servalcat.utils import logger

def test_installation():
    import packaging.version
    vers = logger.dependency_versions()
    pandas_ver = packaging.version.parse(vers["pandas"])
    numpy_ver = packaging.version.parse(vers["numpy"])
    msg_unknown = "Unexpected error occurred (related to numpy+pandas). Please report to authors with the result of servalcat -v."
    msg_skip = "If you want to ignore this error, please specify --skip_test."
    ret = True
    
    try:
        x = pandas.DataFrame(dict(x=[2j]))
        x.merge(x)
    except TypeError:
        ret = False
        if pandas_ver >= packaging.version.parse("1.3.0") and numpy_ver < packaging.version.parse("1.19.1"):
            print("There is a problem in pandas+numpy. Please update numpy to 1.19.1 or newer (or use pandas < 1.3.0).")
        else:
            print(traceback.format_exc())
            print(msg_unknown)
    except:
        print(traceback.format_exc())
        print(msg_unknown)
        ret = False

    if not ret:
        print(msg_skip)
        
    return ret
# test_installation()

def main():
    parser = argparse.ArgumentParser(prog="servalcat",
                                     description="X-ray crystallographic structure refinement (restrained ML refinement "
                                     "with optional TLS), map calculation and utilities.")
    parser.add_argument("--skip_test", action="store_true", help="Skip installation test")
    parser.add_argument("-v", "--version", action="store_true")
    parser.add_argument("--logfile")
    subparsers = parser.add_subparsers(dest="command", title="commands", metavar="<command>")

    modules = dict(refine_xtal=servalcat.refine.refine_xtal,
                   refine_geom=servalcat.refine.refine_geom,
                   sigmaa=servalcat.xtal.sigmaa,
                   fw=servalcat.xtal.french_wilson,
                   util=servalcat.utils.commands)
    aliases = dict(refine_xtal_norefmac="refine_xtal") # backward compatibility

    summaries = dict(refine_xtal="Restrained ML refinement against X-ray data (XYZ, ADP, occupancy, optional TLS)",
                     refine_geom="Geometry-only optimisation of a model or a ligand dictionary",
                     sigmaa="sigmaA parameter estimation and map calculation (2Fo-Fc, Fo-Fc)",
                     fw="French-Wilson conversion of intensities to amplitudes",
                     util="Utilities: TLS group selection, geometry/ADP analysis, model and file tools")
    for n in modules:
        p = subparsers.add_parser(n, help=summaries.get(n), description=summaries.get(n))
        for a, t in aliases.items(): # registered without showing up in -h
            if t == n:
                subparsers._name_parser_map[a] = p
        modules[n].add_arguments(p)

    args = parser.parse_args()
    args.command = aliases.get(args.command, args.command)

    if args.version:
        logger.write_header(head="", show_command=False)
        return
    
    if not args.skip_test and not test_installation():
        return
    
    if args.command == "util" and not args.subcommand:
        print("specify subcommand.")    
    elif args.command in modules:
        if hasattr(modules[args.command], "set_prefix"):
            modules[args.command].set_prefix(args)
        if not args.logfile:
            if getattr(args, "output_prefix", None):
                args.logfile = args.output_prefix + ".log"
            else:
                args.logfile = "servalcat.log"
        logger.set_file(args.logfile)
        logger.write_header()
        try:
            modules[args.command].main(args)
        except SystemExit as e:
            logger.error(str(e))
            sys.exit(1)
        logger.exit_success()
        logger.close()
    else:
        parser.print_help()

# main()

if __name__ == "__main__":
    main()

