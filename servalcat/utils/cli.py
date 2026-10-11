"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.

Layout of the command-line help shared by the programs: (1) required arguments, (2) optional arguments with
default values, (3) optional features that are off unless given. The default of every option is stated in its help.
"""
from __future__ import absolute_import, division, print_function, generators
import argparse
import functools
import textwrap

def add_arg(group, *names, show=None, **kw):
    """add_argument whose help always ends with the default. show: text of the default if it is not simply the
    argparse default (e.g. 'auto'); flags (store_true) show 'off'; '' adds nothing (required arguments)."""
    h = kw.get("help")
    if h and h is not argparse.SUPPRESS and show != "":
        if show is None:
            if kw.get("action") == "store_true": show = "off"
            elif kw.get("default") is not None: show = "%(default)s"
            else: show = "none"
        kw["help"] = "{} (default: {})".format(h, show)
    return group.add_argument(*names, **kw)

def setup_parser(parser, description, usage_args, example_args):
    """description, usage and example for the help of a (sub)command, and a formatter that keeps the layout.
    usage_args: required part of the usage line, e.g. '--hklin FILE --model FILE'"""
    parser.description = textwrap.fill(description, 78)
    parser.usage = "{} {} [options]".format(parser.prog, usage_args)
    parser.epilog = "Example: {} {}".format(parser.prog, example_args)
    parser.formatter_class = functools.partial(argparse.RawDescriptionHelpFormatter, max_help_position=34)
    if hasattr(parser, "_optionals"):
        parser._optionals.title = "help"

REQUIRED = "Required arguments"
DEFAULTS = "Optional arguments with default values"
