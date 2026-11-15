"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.

Summary printed at the end of refine_xtal: ADP analysis of the final model, the change of R-values and geometry
Z-scores during the refinement as sparklines (and PNG graph), the sequence assignment if --seq was used, the
occupancies if they were refined, and the output files. The full refinement log is still written as before; the
summary is only added at the end. Colours are used on a terminal only, never in the log file.
"""
from __future__ import absolute_import, division, print_function, generators
import os
import sys
from servalcat.utils import logger
from servalcat import utils

BLOCKS = "▁▂▃▄▅▆▇█"
RED, YELLOW, CYAN, MAGENTA, GREEN, BOLD, RESET = "\033[91m", "\033[93m", "\033[96m", "\033[95m", "\033[92m", "\033[1m", "\033[0m"
DIVIDER = "─" * 56

def emit(plain, colored=None):
    """Write a line to the terminal and the log. A coloured version goes to a terminal only; the log gets plain text."""
    lg = logger._logger
    if colored is not None and sys.stdout.isatty() and not lg.stopped:
        print(colored, flush=True)
        if lg.ofs is not None:
            lg.ofs.write(plain + "\n")
            lg.ofs.flush()
    else:
        logger.writeln(plain)

def sparkline(values):
    if not values:
        return ""
    lo, hi = min(values), max(values)
    rng = hi - lo
    return "".join(BLOCKS[int((v - lo) / rng * (len(BLOCKS) - 1) + 0.5)] if rng > 0 else BLOCKS[0] for v in values)

def series_from_stats(stats):
    """{name: [value per cycle]} for R-work, R-free, z-Bond and z-Angle (those that are available)"""
    out = {}
    if not stats:
        return out
    data0 = stats[0].get("data", {}).get("summary", {})
    for w, f in (("Rwork", "Rfree"), ("R1work", "R1free")):
        if w in data0:
            out["R-work"] = [s["data"]["summary"][w] for s in stats]
            if f in data0:
                out["R-free"] = [s["data"]["summary"][f] for s in stats]
            break
    else:
        for k in ("R", "R1"):
            if k in data0:
                out["R"] = [s["data"]["summary"][k] for s in stats]
                break
    for name, key in (("zBond", "Bond distances, non H"), ("zAngle", "Bond angles, non H")):
        try:
            vals = [s["geom"]["summary"]["r.m.s.Z"][key] for s in stats]
        except (KeyError, TypeError):
            continue
        out[name] = vals
    return out

COLORS = {"R-work": RED, "R-free": YELLOW, "R": RED, "zBond": CYAN, "zAngle": MAGENTA}

def sparkline_rows(series, ncyc):
    """(plain, coloured) lines: name, sparkline, first -> last, change"""
    rows = []
    for name, vals in series.items():
        first, last = vals[0], vals[-1]
        delta = last - first
        arrow = "↓" if delta < 0 else ("↑" if delta > 0 else "=")
        detail = "{:.4f} → {:.4f}  {} {:.4f}".format(first, last, arrow, abs(delta))
        plain = "  {:<8} {}  {}".format(name, sparkline(vals), detail)
        c = COLORS.get(name, "")
        colored = "  {}{}{:<8}{} {}  {}{}{}".format(c, BOLD, name, RESET, sparkline(vals), c, detail, RESET)
        rows.append((plain, colored))
    return rows

def write_png(path, series):
    """Graph of the series against the cycle number; returns False if matplotlib is not available"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    panels = [("R", [k for k in ("R-work", "R-free", "R") if k in series]),
              ("z-score of the bond lengths", ["zBond"] if "zBond" in series else []),
              ("z-score of the bond angles", ["zAngle"] if "zAngle" in series else [])]
    panels = [(t, ks) for t, ks in panels if ks]
    if not panels:
        return False
    fig, axes = plt.subplots(len(panels), 1, figsize=(6, 2.2 * len(panels) + 0.6), sharex=True, squeeze=False)
    cols = {"R-work": "tab:red", "R-free": "tab:orange", "R": "tab:red", "zBond": "tab:cyan", "zAngle": "tab:purple"}
    for ax, (title, ks) in zip(axes[:, 0], panels):
        for k in ks:
            ax.plot(range(len(series[k])), series[k], marker="o", ms=3, label=k, color=cols.get(k))
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        if len(ks) > 1:
            ax.legend(frameon=False)
    axes[-1, 0].set_xlabel("cycle")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True

def final_summary(args, st, stats, seq_report=None, seq_files=None, occ_result=None, files=(), elapsed=None):
    """Print the summary of a finished refine_xtal run (st: the final model)"""
    emit("")
    emit("=" * 56, BOLD + "=" * 56 + RESET)
    emit(" Refinement summary", BOLD + " Refinement summary" + RESET)
    emit("=" * 56, BOLD + "=" * 56 + RESET)

    # ADP analysis of the model that is written
    utils.model.adp_analysis(st)

    # R-values and geometry during the refinement
    series = series_from_stats(stats)
    if series:
        n = len(next(iter(series.values()))) - 1
        emit(" Refinement progress ({} cycle{}; cycle 0 = input model)".format(n, "" if n == 1 else "s"),
             BOLD + " Refinement progress ({} cycle{}; cycle 0 = input model)".format(n, "" if n == 1 else "s") + RESET)
        emit("  " + DIVIDER)
        for plain, colored in sparkline_rows(series, n):
            emit(plain, colored)
        emit("  " + DIVIDER)
        png = args.output_prefix + "_summary.png"
        if write_png(png, series):
            emit("  graph: {}".format(png))
        emit("")

    # sequence alignment, if --seq was used
    if seq_report is not None:
        emit(" Sequence alignment (--seq)", BOLD + " Sequence alignment (--seq)" + RESET)
        utils.model.log_sequence_report(seq_report, logger, seq_files)

    # occupancies
    if occ_result is not None and occ_result.groups:
        from servalcat.refine import occ_groups
        emit(" Refined occupancies", BOLD + " Refined occupancies" + RESET)
        for line in occ_groups.final_table(st, occ_result):
            emit(line)
        emit("")

    emit(" Output files")
    for label, path in files:
        if path and os.path.exists(path):
            emit("  {:<16} {}".format(label, path))
    if elapsed is not None:
        emit("  total time: {:.0f} s".format(elapsed))
    emit("")
