"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.

HTML report of a TLS group selection (util tls_select, tls_bfit and tls_nma): the groups of every number of groups
that was tried, drawn as boxes along the chains, the scan (R-values, -LL and the fit to the B-factors vs the number of
groups) and the groups of the recommendation. One standalone file (inline SVG, CSS and a few lines of JavaScript).
"""
from __future__ import absolute_import, division, print_function, generators
import datetime
import glob
import html
import math
import os
import re
import sys
import servalcat
from servalcat.utils import logger
from servalcat.refine import tls as tls_mod

e = html.escape
NCOLORS = 8 # categorical colours --s1 .. --s8; they repeat after 8 groups in a chain (the boxes are numbered)
_NUMC = ["#0b0b0b"] * 5 + ["#fff", "#fff", "#0b0b0b"] # group number on the box: the better contrast for each colour

# columns of the scan table: key, header, number format
_COLS = [("N", "N", "{:.0f}"), ("groups", "Groups", "{:.0f}"), ("params", "Parameters", "{:.0f}"),
         ("sizes", "Domain sizes", None), ("Rwork", "R-work", "{:.4f}"), ("Rfree", "R-free", "{:.4f}"),
         ("R1work", "R1-work", "{:.4f}"), ("R1free", "R1-free", "{:.4f}"), ("R", "R", "{:.4f}"), ("R1", "R1", "{:.4f}"),
         ("CCFworkavg", "CC work", "{:.4f}"), ("CCFfreeavg", "CC free", "{:.4f}"), ("-LL", "−LL", "{:,.1f}"),
         ("B_rms", "B fit r.m.s. (Å²)", "{:.2f}"), ("drop_pct", "Drop of B fit (%)", "{:.1f}"),
         ("seconds", "Time (s)", "{:.0f}")]

def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

def _fmt(v, f):
    if isinstance(v, str) and f is not None:
        return e(v)
    if f is None:
        return e(str(v)) if v not in (None, "") else "–"
    return f.format(v) if _finite(v) else "–"

def chain_tracks(st, scans):
    """[(chain name, [(residue key, residue label)])]: the polymer residues of every chain, plus residues that are in a
    group in any scan"""
    grouped = {(ch, str(r.seqid)) for _, _, items in scans for ch, r, g in items if g > 0}
    out = []
    for ch in st[0]:
        res = [r for r in ch if tls_mod.is_polymer_residue(r) or (ch.name, str(r.seqid)) in grouped]
        if res:
            out.append((ch.name, [(str(r.seqid), tls_mod._seqid_str(r.seqid)) for r in res]))
    return out

def group_boxes(tracks, scans, selected, unit):
    """SVG: one row per scan, the chains side by side, each group a box (residues of the chain in no group: track)"""
    tot = sum(len(r) for _, r in tracks)
    left, width, gap = 132, 840, 8
    scale = (width - gap * (len(tracks) - 1)) / max(tot, 1)
    rowh, boxh, top = 34, 24, 34
    H = top + rowh * len(scans) + 2
    s = [f'<svg class="gd" viewBox="0 0 {left + width + 4} {H}" role="img" aria-label="TLS groups along the chains">']
    xs, x, res_of = {}, left, dict(tracks)
    for name, res in tracks:
        w = len(res) * scale
        xs[name] = (x, w)
        lab = name if w < 70 else f"{name} ({len(res)})"
        s.append(f'<text class="ax" x="{x + w / 2:.1f}" y="{top - 12}" text-anchor="middle">{e(lab)}</text>')
        x += w + gap
    s.append(f'<text class="ax" x="0" y="{top - 12}">chain (residues)</text>')
    for i, (n, label, items) in enumerate(scans):
        y = top + i * rowh
        sel = n == selected and n is not None
        if sel:
            s.append(f'<rect class="selrow" x="0" y="{y - 4}" width="{left + width + 4}" height="{boxh + 8}" rx="6"/>')
        ngr = len({g for _, _, g in items if g > 0})
        s.append(f'<text class="rl" x="6" y="{y + boxh / 2 - 1}">{e(label)}</text>')
        s.append(f'<text class="rs" x="6" y="{y + boxh / 2 + 12}">{ngr} group{"s" if ngr != 1 else ""}'
                 f'{" · selected" if sel else ""}</text>')
        dom = {(ch, str(r.seqid)): g for ch, r, g in items if g > 0}
        gchains = {}
        for (ch, _), g in dom.items():
            gchains.setdefault(g, set()).add(ch)
        order = {} # colour of a group that spans chains: its order in the whole model
        for name, _ in tracks:
            x0, w = xs[name]
            s.append(f'<rect class="track" x="{x0:.1f}" y="{y}" width="{w:.1f}" height="{boxh}" rx="4"/>')
            runs = []
            res = res_of[name]
            for k, (key, _) in enumerate(res):
                g = dom.get((name, key))
                if runs and runs[-1][0] == g:
                    runs[-1][2] = k
                else:
                    runs.append([g, k, k])
            local = {}
            for g, a, b in runs:
                if g is None:
                    continue
                if len(gchains[g]) > 1:
                    ci = order.setdefault(g, len(order))
                else:
                    ci = local.setdefault(g, len(local))
                xa, wb = x0 + a * scale + 1, max((b - a + 1) * scale - 2, 1)
                tip = f"{label} · chain {name} {res[a][1]}–{res[b][1]} · group {g} · {b - a + 1} residues"
                s.append(f'<rect class="box" x="{xa:.1f}" y="{y + 1}" width="{wb:.1f}" height="{boxh - 2}" rx="3" '
                         f'style="fill:var(--s{ci % NCOLORS + 1})" data-tip="{e(tip)}"/>')
                if wb >= 18:
                    s.append(f'<text class="bn" x="{xa + wb / 2:.1f}" y="{y + boxh / 2 + 4}" text-anchor="middle" '
                             f'style="fill:{_NUMC[ci % NCOLORS]}">{g}</text>')
    s.append('</svg>')
    return "\n".join(s)

def _nice_step(raw):
    p = 10 ** math.floor(math.log10(raw))
    for k in (1, 2, 2.5, 5, 10):
        if raw <= k * p:
            return k * p
    return 10 * p

def line_chart(xs, series, xlabel, ylabel, selected, fmt):
    """SVG line chart of a scan. series: [(label, values, colour)]; a marker per point with a tooltip;
    the selected number of groups is marked by a dashed line"""
    W, H, L, R, T, B = 460, 250, 58, 70, 22, 40
    pw, ph = W - L - R, H - T - B
    ys = [v for _, vals, _ in series for v in vals if _finite(v)]
    if len(ys) < 2:
        return ""
    lo, hi = min(ys), max(ys)
    pad = max((hi - lo) * 0.1, abs(hi) * 1e-4, 1e-6)
    lo, hi = lo - pad, hi + pad
    step = _nice_step((hi - lo) / 4)
    x0, x1 = min(xs) - 0.4, max(xs) + 0.4
    X = lambda v: L + pw * (v - x0) / (x1 - x0)
    Y = lambda v: T + ph * (1 - (v - lo) / (hi - lo))
    s = [f'<svg class="lc" viewBox="0 0 {W} {H}" role="img" aria-label="{e(ylabel)} vs {e(xlabel)}">']
    t = math.ceil(lo / step) * step
    while t <= hi + 1e-12:
        s.append(f'<line class="grid" x1="{L}" x2="{L + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>')
        s.append(f'<text class="tick" x="{L - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{e(fmt.format(t))}</text>')
        t += step
    s.append(f'<line class="base" x1="{L}" x2="{L + pw}" y1="{T + ph}" y2="{T + ph}"/>')
    every = max(1, math.ceil(len(xs) / 12))
    for v in xs[::every]:
        s.append(f'<text class="tick" x="{X(v):.1f}" y="{T + ph + 16}" text-anchor="middle">{v}</text>')
    s.append(f'<text class="ax" x="{L + pw / 2}" y="{H - 4}" text-anchor="middle">{e(xlabel)}</text>')
    s.append(f'<text class="ax" x="4" y="{T - 8}">{e(ylabel)}</text>')
    if selected is not None and selected in xs:
        s.append(f'<line class="sel" x1="{X(selected):.1f}" x2="{X(selected):.1f}" y1="{T}" y2="{T + ph}"/>')
        s.append(f'<text class="tick" x="{X(selected) + 4:.1f}" y="{T + 10}">selected</text>')
    ends = []
    for label, vals, col in series:
        pts = [(x, v) for x, v in zip(xs, vals) if _finite(v)]
        if not pts:
            continue
        d = " ".join(f'{"M" if i == 0 else "L"}{X(x):.1f},{Y(v):.1f}' for i, (x, v) in enumerate(pts))
        s.append(f'<path class="ln" d="{d}" style="stroke:{col}"/>')
        for x, v in pts:
            s.append(f'<circle class="mk" cx="{X(x):.1f}" cy="{Y(v):.1f}" r="4" style="fill:{col}" '
                     f'data-tip="N = {x} · {e(label)} {e(fmt.format(v))}"/>')
        ends.append((Y(pts[-1][1]), X(pts[-1][0]), label))
    ends.sort()
    placed = []
    for yy, xx, label in ends: # direct labels at the line ends, pushed apart when they would overlap
        if placed and yy - placed[-1] < 12:
            yy = placed[-1] + 12
        placed.append(yy)
        s.append(f'<text class="el" x="{xx + 8:.1f}" y="{yy + 4:.1f}">{e(label)}</text>')
    s.append('</svg>')
    return "\n".join(s)

def scan_charts(rows, kw, kf, selected, unit):
    rows = [r for r in rows if _finite(r.get("N"))]
    if len(rows) < 2:
        return ""
    xs = [int(r["N"]) for r in rows]
    xlabel = f"N ({unit})"
    charts = []
    if kw:
        series = [(kw, [r.get(kw) for r in rows], "var(--s1)")]
        if kf:
            series.append((kf, [r.get(kf) for r in rows], "var(--s2)"))
        legend = "".join(f'<span><i style="background:{c}"></i>{e(lab)}</span>' for lab, _, c in series) if kf else ""
        charts.append(("R-values of the trial refinements", legend, line_chart(xs, series, xlabel, "R", selected, "{:.3f}")))
    if any(_finite(r.get("-LL")) for r in rows):
        charts.append(("Likelihood target (−LL)", "", line_chart(xs, [("−LL", [r.get("-LL") for r in rows], "var(--s1)")],
                                                                xlabel, "−LL", selected, "{:,.0f}")))
    if any(_finite(r.get("B_rms")) for r in rows):
        charts.append(("Fit of TLS to the B-factors", "", line_chart(xs, [("r.m.s.", [r.get("B_rms") for r in rows], "var(--s1)")],
                                                                    xlabel, "r.m.s. B(obs) − B(TLS) (Å²)", selected, "{:.2f}")))
    charts = [c for c in charts if c[2]]
    return "".join(f'<div><h3>{e(t)}</h3>{f"<div class=legend>{lg}</div>" if lg else ""}<figure>{svg}</figure></div>'
                   for t, lg, svg in charts)

def scan_table(rows, selected):
    keys = [k for k, _, _ in _COLS if any(k in r for r in rows)]
    dk = [k for k in rows[0] if k.startswith("d_")] if rows else []
    cols = [c for c in _COLS if c[0] in keys]
    if dk: # change of the selection metric from N-1 to N groups, after the R columns
        i = max([j for j, c in enumerate(cols) if c[0] in ("Rwork", "Rfree", "R1work", "R1free", "R", "R1")], default=len(cols) - 1)
        cols.insert(i + 1, (dk[0], f"Δ{dk[0][2:]} (N−1 → N)", "{:+.4f}"))
    head = "".join(f'<th class="{"n" if f else ""}">{e(h)}</th>' for _, h, f in cols)
    body = []
    for r in rows:
        cls = ' class="best"' if r.get("N") == selected else ""
        body.append(f"<tr{cls}>" + "".join(f'<td class="{"n" if f else ""}">{_fmt(r.get(k), f)}</td>' for k, _, f in cols)
                    + "</tr>")
    return f'<div class="scroll"><table class="t"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'

def groups_table(groups, items):
    nres = {}
    for ch, r, g in items:
        if g > 0:
            nres[g] = nres.get(g, 0) + 1
    rows = []
    for i, sels in enumerate(groups):
        chains = sorted({s.split("/")[2] for s in sels if s.count("/") >= 3})
        rows.append(f'<tr><td class="n">{i + 1}</td><td>{e(", ".join(chains))}</td><td class="n">{nres.get(i + 1, "–")}</td>'
                    f'<td><code>{"</code> <code>".join(e(s) for s in sels)}</code></td></tr>')
    return ('<div class="scroll"><table class="t fit"><thead><tr><th class="n">Group</th><th>Chain</th><th class="n">Residues</th>'
            f'<th>Selections</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')

def output_files(prefix):
    """[(file, description)] of the files of a TLS selection run with this output prefix, as written so far"""
    desc = [(r"_tls\.yaml", lambda m: "TLS definition of the recommendation; use it with refine_xtal --tls_file"),
            (r"_domains\.pdb", lambda m: "model with the group number as B-factor (colour by B in Coot or PyMOL)"),
            (r"_domains\.csv", lambda m: "group of every residue"),
            (r"_scan\.png", lambda m: "plot of the scan"),
            (r"_scan\.csv", lambda m: "scan table"), (r"_scan\.json", lambda m: "scan table, reason and groups of every N"),
            (r"_tls_N(\d+)\.yaml", lambda m: f"TLS definition with N = {m.group(1)}"),
            (r"_domains_N(\d+)\.csv", lambda m: f"group of every residue, N = {m.group(1)}")]
    found = []
    for f in glob.glob(glob.escape(prefix) + "_*"):
        for i, (pat, d) in enumerate(desc):
            m = re.fullmatch(pat, f[len(prefix):])
            if m: # the most useful files first, then by number
                found.append(((i, int(m.group(1)) if m.groups() else 0), f, d(m)))
                break
    out = [(f, d) for _, f, d in sorted(found)]
    if os.path.exists(prefix + ".log"):
        out.append((prefix + ".log", "log"))
    return out

def write_report(path, tool, st, scans, selected, reason, rows=(), kw=None, kf=None, groups=None, unit="groups per chain",
                 facts=(), files=()):
    """scans: [(N or None, row label, [(chain name, residue, group number)])], one box row each;
    selected: N of the recommendation (0: no TLS); rows: the scan table (dicts as in PREFIX_scan.json);
    groups: selections of the recommended groups; facts: [(label, text)]; files: [(file name, description)]"""
    rows = list(rows)
    reason = "\n".join(l.strip() for l in str(reason).splitlines())
    tracks = chain_tracks(st, scans)
    sel_items = next((items for n, _, items in scans if n == selected), [])
    ngr = len(groups) if groups else 0
    if selected == 0:
        headline = "Recommendation: no TLS"
    else:
        headline = f"Recommendation: {selected} {unit} ({ngr} TLS group{'s' if ngr != 1 else ''})" \
                   if selected is not None else f"{ngr} TLS group{'s' if ngr != 1 else ''}"
    cards = []
    if selected is not None:
        cards.append(("Selected N", str(selected), unit))
    cards.append(("TLS groups", str(ngr), "in the recommended definition"))
    r_sel = next((r for r in rows if r.get("N") == selected), None)
    r_0 = next((r for r in rows if r.get("N") == 0), None)
    key = kf or kw
    if key and r_sel is not None and r_0 is not None and _finite(r_sel.get(key)) and _finite(r_0.get(key)):
        cards.append((f"{key}, no TLS → selected", f"{r_0[key]:.4f} → {r_sel[key]:.4f}",
                      f"{(r_sel[key] - r_0[key]) * 100:+.2f} points in the trial refinement"))
    if r_sel is not None and _finite(r_sel.get("B_rms")):
        cards.append(("B fit r.m.s.", f"{r_sel['B_rms']:.2f} Å²", "TLS fitted to the B-factors"))
    p = [f'<!doctype html><html lang="en"><head><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         f'<title>TLS groups: {e(tool)}</title><style>{CSS}</style></head><body>',
         '<button class="theme" type="button" aria-label="Toggle light or dark theme">Theme</button><main>',
         f'<h1>TLS groups from {e(tool)}</h1>',
         f'<p class="sub">{e(os.path.basename(str(dict(facts).get("Model", ""))))} · servalcat {e(servalcat.__version__)} · '
         f'{datetime.datetime.now().strftime("%d %B %Y %H:%M")}</p>',
         f'<div class="rec"><b>{e(headline)}.</b> <span>{e(reason)}</span></div>',
         '<div class="cards">' + "".join(f'<div class="card"><div class="k">{e(k)}</div><div class="v">{e(v)}</div>'
                                          f'<div class="d">{e(d)}</div></div>' for k, v, d in cards) + '</div>']
    if facts:
        p.append('<dl class="facts">' + "".join(f'<div><dt>{e(k)}</dt><dd>{e(str(v))}</dd></div>' for k, v in facts) + '</dl>')
    p.append('<section><h2>TLS groups along the chains</h2>')
    p.append(f'<div class="scroll">{group_boxes(tracks, scans, selected, unit)}</div>')
    p.append('<p class="cap">One row per number of groups tried; the highlighted row is the recommendation. Each box is a TLS '
             f'group, numbered as in the yaml file and coloured by its order in the chain (colours repeat after {NCOLORS} '
             'groups in a chain). Grey: polymer residues in no group. Hover a box for its residues.</p></section>')
    charts = scan_charts(rows, kw, kf, selected, unit)
    if rows:
        p.append('<section><h2>Scan</h2>')
        if charts:
            p.append(f'<div class="charts">{charts}</div>')
        p.append(scan_table(rows, selected))
        cap = []
        if kw:
            cap.append("N = 0 is the refinement without TLS. R-values and −LL come from short trial refinements "
                       "(coordinates fixed, residual B and TLS refined).")
        if any(_finite(r.get("B_rms")) for r in rows):
            cap.append("The B fit is one TLS group per domain fitted directly to the B-factors of the model.")
        cap.append("The highlighted row is the recommendation.")
        p.append(f'<p class="cap">{e(" ".join(cap))}</p></section>')
    if groups:
        p.append(f'<section><h2>Groups of the recommendation</h2>{groups_table(groups, sel_items)}</section>')
    if files:
        p.append('<section><h2>Files</h2><table class="t files"><tbody>' + "".join(
            f'<tr><td><a href="{e(os.path.basename(f))}">{e(os.path.basename(f))}</a></td><td>{e(d)}</td></tr>'
            for f, d in files) + '</tbody></table></section>')
    p.append(f'<p class="cmd">servalcat {e(" ".join(sys.argv[1:]))}</p>')
    p.append(f'</main><div id="tip" role="tooltip"></div><script>{JS}</script></body></html>')
    with open(path, "w") as ofs:
        ofs.write("\n".join(p))
    logger.writeln(f"Report written: {path}")

CSS = r"""
:root{--page:#f9f9f7;--surface:#fcfcfb;--surface2:#f0efec;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--base:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;
--s7:#6250d6;--s8:#e34948;--hl:rgba(42,120,214,.10);color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--page:#0d0d0d;--surface:#1a1a19;--surface2:#262624;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--base:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;
--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;--hl:rgba(57,135,229,.16);color-scheme:dark}}
:root[data-theme="dark"]{--page:#0d0d0d;--surface:#1a1a19;--surface2:#262624;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--base:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;
--s6:#008300;--s7:#9085e9;--s8:#e66767;--hl:rgba(57,135,229,.16);color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:26px;line-height:1.2;margin:0 0 6px;font-weight:650}
h2{font-size:19px;margin:0 0 12px;font-weight:650}
h3{font-size:14px;margin:4px 0 6px;font-weight:650}
.sub{color:var(--ink2);margin:0 0 14px}
.rec{background:var(--surface);border:1px solid var(--border);border-left:4px solid var(--s1);border-radius:10px;padding:12px 16px;
white-space:pre-wrap}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:14px 0 0}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.card .k{font-size:12px;color:var(--muted)}.card .v{font-size:21px;font-weight:650;font-variant-numeric:tabular-nums}
.card .d{font-size:12.5px;color:var(--ink2)}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px 22px;margin:16px 0 0}
.facts div{min-width:0}.facts dt{font-size:12px;color:var(--muted)}.facts dd{margin:0;font-size:13.5px;overflow-wrap:anywhere}
section{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px 20px 16px;margin:18px 0}
.scroll{overflow-x:auto}
svg{display:block;width:100%;height:auto;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
svg.gd{min-width:720px}
.charts{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(420px,100%),1fr));gap:20px;margin:0 0 12px}.charts>div{min-width:0}
figure{margin:0}
.track{fill:var(--surface2)}.selrow{fill:var(--hl)}
.box{stroke:none}.bn{font-size:11px;font-weight:600;pointer-events:none}
.ax,.tick{fill:var(--muted);font-size:11.5px;font-variant-numeric:tabular-nums}
.rl{fill:var(--ink);font-size:13px;font-weight:600}.rs{fill:var(--ink2);font-size:11.5px}
line.grid{stroke:var(--grid);stroke-width:1}.base{stroke:var(--base);stroke-width:1}
.sel{stroke:var(--muted);stroke-width:1;stroke-dasharray:4 3}
.ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.mk{stroke:var(--surface);stroke-width:2}
.el{fill:var(--ink2);font-size:11.5px}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12.5px;color:var(--ink2);margin:0 0 4px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
table.t{border-collapse:collapse;font-size:13px;width:100%}
table.t th{text-align:left;font-weight:600;color:var(--ink2);border-bottom:1px solid var(--base);padding:5px 8px;font-size:12px;
vertical-align:bottom}
table.t td{border-bottom:1px solid var(--grid);padding:5px 8px;vertical-align:top}
table.t.fit{width:auto}table.t.fit td:last-child{overflow-wrap:anywhere}
table.t .n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
table.t tr.best td{background:var(--hl);font-weight:600}
code{font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--surface2);padding:1px 4px;border-radius:4px}
.cap{font-size:12.5px;color:var(--ink2);margin:8px 0 0;max-width:900px}
.cmd{font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:var(--muted);overflow-wrap:anywhere}
a{color:inherit}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--border);
box-shadow:0 4px 16px rgba(0,0,0,.12);border-radius:8px;padding:6px 9px;font-size:12.5px;max-width:340px;display:none;z-index:9;
font-variant-numeric:tabular-nums}
.theme{position:fixed;top:12px;right:12px;font-size:12px;background:var(--surface);color:var(--ink2);border:1px solid var(--border);
border-radius:8px;padding:4px 9px;cursor:pointer}
@media (max-width:600px){section{padding:14px 12px 12px}h1{font-size:22px}}
"""

JS = r"""
const tip=document.getElementById('tip');
document.addEventListener('mousemove',ev=>{const t=ev.target.closest('[data-tip]');
if(!t){tip.style.display='none';return;}
tip.textContent=t.getAttribute('data-tip');tip.style.display='block';const w=tip.offsetWidth,h=tip.offsetHeight;
let x=ev.clientX+14,y=ev.clientY+14;if(x+w>innerWidth-8)x=ev.clientX-w-14;if(y+h>innerHeight-8)y=ev.clientY-h-14;
tip.style.left=x+'px';tip.style.top=y+'px';});
document.querySelector('.theme').addEventListener('click',()=>{const r=document.documentElement;
const cur=r.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
r.dataset.theme=cur==='dark'?'light':'dark';});
"""
