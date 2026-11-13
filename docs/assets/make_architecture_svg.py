"""Generate docs/assets/architecture.svg: animated (CSS) diagram of PatchTransformer V1.

    python docs/assets/make_architecture_svg.py docs/assets/architecture.svg

Shares one time axis between the history (8 patches x 60 px = 256 steps shown) and the
192-step forecast (6 x 60 px), so the fan chart is drawn to the same scale as the input.
"""

import math
import random
import sys

OUT = sys.argv[1]
T = 12.0  # animation cycle, seconds
HIDE_AT, GONE_AT = 11.2, 11.7

random.seed(7)
X0, ORIGIN, XEND = 40, 520, 880
NP = 8
PW = 60
CY = 170
GAP = (296, 324)


def level(x):
    t = x - X0
    return CY - 0.02 * t + 16 * math.sin(2 * math.pi * t / 120) + 5 * math.sin(2 * math.pi * t / 40 + 1)


hist = {x: level(x) + random.gauss(0, 2.2) for x in range(X0, ORIGIN + 1, 4)}


def hist_path():
    parts, pen = [], False
    for x, y in hist.items():
        if GAP[0] < x < GAP[1]:
            pen = False
            continue
        parts.append(f"{'L' if pen else 'M'}{x},{y:.1f}")
        pen = True
    return " ".join(parts)


fut = list(range(ORIGIN, XEND + 1, 4))


def sigma(x):
    return 2.5 + 0.035 * (x - ORIGIN)


def band(z):
    up = [f"{x},{level(x) - z * sigma(x):.1f}" for x in fut]
    dn = [f"{x},{level(x) + z * sigma(x):.1f}" for x in reversed(fut)]
    return "M" + " L".join(up) + " L" + " L".join(dn) + " Z"


median = "M" + " L".join(f"{x},{level(x):.1f}" for x in fut)

# ---------------------------------------------------------------- animation
css_anims = []


def p(t):
    return f"{100 * t / T:.2f}%"


def anim(cls, kind, start, dur=0.4):
    end = start + dur
    if kind == "draw":
        frames = (f"0%,{p(start)}{{stroke-dashoffset:1;opacity:1}}"
                  f"{p(end)},{p(HIDE_AT)}{{stroke-dashoffset:0;opacity:1}}"
                  f"{p(GONE_AT)},100%{{stroke-dashoffset:0;opacity:0}}")
        base = "stroke-dasharray:1;"
    elif kind == "fade":
        frames = (f"0%,{p(start)}{{opacity:0;transform:translateY(-5px)}}"
                  f"{p(end)},{p(HIDE_AT)}{{opacity:1;transform:none}}"
                  f"{p(GONE_AT)},100%{{opacity:0;transform:none}}")
        base = ""
    elif kind == "grow":
        frames = (f"0%,{p(start)}{{transform:scaleX(0)}}"
                  f"{p(end)},{p(HIDE_AT)}{{transform:scaleX(1);opacity:1}}"
                  f"{p(GONE_AT)},100%{{transform:scaleX(1);opacity:0}}")
        base = f"transform-origin:{ORIGIN}px {CY}px;"
    elif kind == "pulse":  # a short dash running along the path once
        frames = (f"0%,{p(start)}{{stroke-dashoffset:0.12;opacity:0}}"
                  f"{p(start + 0.01)}{{opacity:1}}"
                  f"{p(end)}{{stroke-dashoffset:-1;opacity:1}}"
                  f"{p(end + 0.01)},100%{{stroke-dashoffset:-1;opacity:0}}")
        base = "stroke-dasharray:0.12 1.2;"
    else:
        raise ValueError(kind)
    css_anims.append(f".{cls}{{{base}animation:{cls} {T}s linear infinite}}@keyframes {cls}{{{frames}}}")
    return cls


# ---------------------------------------------------------------- drawing
o = []
add = o.append


def text(x, y, s, cls="lbl", anchor="start", extra=""):
    add(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}"{extra}>{s}</text>')


W, H = 960, 660
add(f'<rect width="{W}" height="{H}" rx="14" class="bg"/>')

# header
text(40, 46, "PatchTransformer V1", "title")
text(40, 66, "6.6M parameters · trained on synthetic data only · zero-shot forecaster", "sub")

# ---- row 1: the time axis (history | forecast)
for k in range(1, NP):
    x = X0 + k * PW
    add(f'<line x1="{x}" y1="116" x2="{x}" y2="226" class="grid"/>')
add(f'<rect x="{GAP[0]}" y="116" width="{GAP[1] - GAP[0]}" height="110" fill="url(#hatch)" class="{anim("gap", "fade", 1.0)}"/>')
text((GAP[0] + GAP[1]) / 2, 110, "missing", "tiny", "middle", f' class2=""')
add(f'<line x1="{X0}" y1="226" x2="{XEND}" y2="226" class="axis"/>')
add(f'<line x1="{ORIGIN}" y1="104" x2="{ORIGIN}" y2="226" class="origin"/>')
text(ORIGIN + 6, 240, "forecast origin", "tiny")
text(44, 128, "history · up to 1024 steps", "cap")
text(ORIGIN + 10, 128, "forecast · 192 steps × 9 quantiles", "cap fc")

add(f'<path d="{hist_path()}" pathLength="1" class="hist {anim("histline", "draw", 0.2, 1.4)}"/>')

fan_cls = anim("fan", "grow", 8.3, 1.3)
add(f'<g class="{fan_cls}">')
for z in (1.2816, 0.8416, 0.5244, 0.2533):
    add(f'<path d="{band(z)}" class="band"/>')
add("</g>")
add(f'<path d="{median}" pathLength="1" class="med {anim("median", "draw", 8.3, 1.3)}"/>')

# rollout arc (horizons beyond 192)
ro = anim("rollout", "fade", 9.8, 0.5)
add(f'<g class="{ro}">')
add(f'<path d="M{XEND - 6},112 Q{(XEND + ORIGIN) / 2},74 {ORIGIN + 8},112" pathLength="1" class="loop" marker-end="url(#ahead-fc)"/>')
text((XEND + ORIGIN) / 2, 84, "h &gt; 192: median fed back as history, run again", "tiny fc", "middle",
     f' class3=""')
add("</g>")

# ---- row 2: normalize + patchify -> tokens [32 values ++ 32 mask]
cell = 50 / 8
for i in range(NP):
    x0 = X0 + i * PW
    cx = x0 + PW / 2
    g = anim(f"chip{i}", "fade", 1.8 + 0.12 * i, 0.35)
    add(f'<g class="{g}">')
    add(f'<line x1="{cx}" y1="232" x2="{cx}" y2="250" class="conn" marker-end="url(#ahead)"/>')
    add(f'<rect x="{x0 + 5}" y="256" width="50" height="16" rx="2" class="chipv"/>')
    seg = [(x, y) for x, y in hist.items() if x0 <= x <= x0 + PW and not GAP[0] < x < GAP[1]]
    if seg:
        ys = [y for _, y in seg]
        lo, hi = min(ys), max(ys)
        pts = " ".join(f"{x0 + 5 + (x - x0) * 50 / PW:.1f},{270 - 12 * (hi - y) / max(hi - lo, 1e-6):.1f}" for x, y in seg)
        add(f'<polyline points="{pts}" class="mini"/>')
    for c in range(8):
        lx = x0 + 5 + c * cell
        mid = x0 + (c + 0.5) * PW / 8
        missing = GAP[0] < mid < GAP[1]
        add(f'<rect x="{lx + 0.6:.1f}" y="276" width="{cell - 1.2:.1f}" height="10" rx="1" class="{"mask0" if missing else "mask1"}"/>')
    add("</g>")
text(560, 268, "Normalize, then cut into patches", "lbl")
text(560, 286, "median / MAD · 32 values ++ 32 mask bits", "mono")

# ---- row 3: residual MLP embedding -> d_model 192
shades = [0.9, 0.35, 0.65, 0.2, 0.8, 0.5]
for i in range(NP):
    x0 = X0 + i * PW
    cx = x0 + PW / 2
    g = anim(f"vec{i}", "fade", 2.9 + 0.1 * i, 0.35)
    add(f'<g class="{g}">')
    add(f'<line x1="{cx}" y1="292" x2="{cx}" y2="308" class="conn" marker-end="url(#ahead)"/>')
    for r in range(6):
        op = shades[(r + i * 2) % 6]
        add(f'<rect x="{cx - 12}" y="{314 + r * 7}" width="24" height="6" rx="1" class="vecc" fill-opacity="{op}"/>')
    add(f'<line x1="{cx}" y1="358" x2="{cx}" y2="378" class="conn" marker-end="url(#ahead)"/>')
    add("</g>")
text(560, 334, "Residual MLP embedding", "lbl")
text(560, 352, "64 → 384 → 192 per patch token", "mono")

# ---- row 4: 8 pre-norm transformer blocks
add('<rect x="28" y="384" width="504" height="156" rx="10" class="block"/>')
pills = ["RMSNorm", "Attention · 3 heads · RoPE", "RMSNorm", "SwiGLU 768"]
px = 44
for j, s in enumerate(pills):
    w = 14 + 6.6 * len(s)
    add(f'<rect x="{px}" y="396" width="{w:.0f}" height="22" rx="11" class="pill"/>')
    text(px + w / 2, 411, s, "pilltxt", "middle")
    px += w
    if j < len(pills) - 1:
        add(f'<line x1="{px + 3:.0f}" y1="407" x2="{px + 13:.0f}" y2="407" class="conn" marker-end="url(#ahead)"/>')
        px += 18
text(518, 412, "× 8", "times", "end")

last_cx = X0 + (NP - 1) * PW + PW / 2
TY = 508
for i in range(NP):
    cx = X0 + i * PW + PW / 2
    add(f'<line x1="{cx}" y1="426" x2="{cx}" y2="{TY - 8}" class="stream"/>')
# faint bidirectional attention among history tokens
for i in range(NP - 2):
    a, b = X0 + i * PW + PW / 2, X0 + (i + 2) * PW + PW / 2
    add(f'<path d="M{a},{TY - 7} Q{(a + b) / 2},{TY - 44} {b},{TY - 7}" class="attn-bg"/>')
# every token -> last token (the one the head reads)
for i in range(NP - 1):
    a = X0 + i * PW + PW / 2
    hgt = (last_cx - a) * 0.36
    cls = anim(f"arc{i}", "draw", 4.4 + 0.15 * i, 0.6)
    add(f'<path d="M{a},{TY - 7} Q{(a + last_cx) / 2},{TY - 7 - hgt:.0f} {last_cx},{TY - 7}" pathLength="1" class="attn {cls}"/>')
for i in range(NP):
    cx = X0 + i * PW + PW / 2
    add(f'<circle cx="{cx}" cy="{TY}" r="6.5" class="tok {anim(f"tok{i}", "fade", 3.9 + 0.06 * i, 0.3)}"/>')
add(f'<circle cx="{last_cx}" cy="{TY}" r="11" class="ring {anim("ring", "fade", 6.9, 0.3)}"/>')

text(560, 420, "8 pre-norm Transformer blocks", "lbl")
text(560, 438, "d_model 192 · bidirectional over the history", "mono")
for k in range(8):
    x = 560 + k * 20
    add(f'<rect x="{x}" y="452" width="14" height="14" rx="3" class="pip"/>')
    add(f'<rect x="{x}" y="452" width="14" height="14" rx="3" class="pipon {anim(f"pip{k}", "fade", 4.6 + 0.3 * k, 0.15)}"/>')
text(726, 463, "layer 1 → 8", "tiny")

# ---- row 5: head on the last token -> 192 x 9, back up to the forecast
hp = f"M{last_cx},{TY + 12} L{last_cx},582 L556,582"
add(f'<path d="{hp}" class="conn" marker-end="url(#ahead)"/>')
add(f'<path d="{hp}" pathLength="1" class="sig {anim("sig1", "pulse", 7.0, 0.6)}"/>')
add('<rect x="560" y="560" width="300" height="44" rx="8" class="head"/>')
add(f'<rect x="560" y="560" width="300" height="44" rx="8" class="headon {anim("headon", "fade", 7.5, 0.3)}"/>')
text(574, 578, "Quantile head on the last patch token", "lbl")
text(574, 595, "RMSNorm → MLP 192 → 768 → 192 × 9", "mono")
up = f"M860,582 L908,582 L908,{CY} L{XEND + 6},{CY}"
add(f'<path d="{up}" class="conn fcs" marker-end="url(#ahead-fc)"/>')
add(f'<path d="{up}" pathLength="1" class="sig {anim("sig2", "pulse", 7.8, 0.6)}"/>')

# footer: inference trick + loss
foot = ('Inference: forecast x and −x in one batch, then q<tspan dy="3" class="subs">τ</tspan>'
        '<tspan dy="-3"> = ½ · (q</tspan><tspan dy="3" class="subs">τ</tspan><tspan dy="-3">(x) − q</tspan>'
        '<tspan dy="3" class="subs">1−τ</tspan><tspan dy="-3">(−x)). Trained with the pinball loss on 9 quantiles, 0.1 … 0.9.</tspan>')
text(40, 636, foot, "foot")

# ---------------------------------------------------------------- assemble
style = """
svg{--bg:#F5F7FA;--panel:#FFFFFF;--ink:#17202C;--muted:#5A6576;--rule:#D2D9E3;--hist:#245A9C;--fc:#C25E14;
--sans:-apple-system,BlinkMacSystemFont,"Segoe UI","Helvetica Neue",Arial,sans-serif;
--mono:ui-monospace,"SF Mono","Cascadia Mono",Consolas,Menlo,monospace}
@media (prefers-color-scheme:dark){svg{--bg:#0D1219;--panel:#141B25;--ink:#E5EAF1;--muted:#93A0B3;--rule:#2A3443;--hist:#7AAAE6;--fc:#F09A4A}}
.bg{fill:var(--bg)}
text{fill:var(--ink);font-family:var(--sans)}
.title{font-size:20px;font-weight:700;letter-spacing:-.01em}
.sub{font-size:12.5px;fill:var(--muted)}
.lbl{font-size:13px;font-weight:600}
.cap{font-size:11.5px;fill:var(--muted);font-family:var(--mono)}
.mono{font-size:11.5px;fill:var(--muted);font-family:var(--mono)}
.tiny{font-size:10.5px;fill:var(--muted);font-family:var(--mono)}
.fc{fill:var(--fc)}
.foot{font-size:11.5px;fill:var(--muted)}
.subs{font-size:8.5px;fill:var(--muted)}
.times{font-size:15px;font-weight:700;fill:var(--muted);font-family:var(--mono)}
.pilltxt{font-size:10.5px;font-family:var(--mono)}
.grid{stroke:var(--rule);stroke-width:1;stroke-dasharray:2 3}
.axis{stroke:var(--ink);stroke-width:1;opacity:.5}
.origin{stroke:var(--ink);stroke-width:1.2;stroke-dasharray:4 3}
.hist{fill:none;stroke:var(--hist);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.band{fill:var(--fc);fill-opacity:.16}
.med{fill:none;stroke:var(--fc);stroke-width:2;stroke-linecap:round}
.loop{fill:none;stroke:var(--fc);stroke-width:1.3}
.conn{fill:none;stroke:var(--muted);stroke-width:1.2}
.fcs{stroke:var(--fc)}
.chipv{fill:var(--hist);fill-opacity:.12;stroke:var(--hist);stroke-opacity:.5;stroke-width:1}
.mini{fill:none;stroke:var(--hist);stroke-width:1.2}
.mask1{fill:var(--ink);fill-opacity:.75}
.mask0{fill:none;stroke:var(--muted);stroke-width:1}
.vecc{fill:var(--hist)}
.block{fill:var(--panel);stroke:var(--rule);stroke-width:1.2}
.pill{fill:var(--bg);stroke:var(--rule);stroke-width:1}
.stream{stroke:var(--rule);stroke-width:1.5}
.attn-bg{fill:none;stroke:var(--muted);stroke-opacity:.35;stroke-width:1}
.attn{fill:none;stroke:var(--hist);stroke-width:1.5;stroke-opacity:.85}
.tok{fill:var(--panel);stroke:var(--hist);stroke-width:2}
.ring{fill:none;stroke:var(--fc);stroke-width:2.2}
.pip{fill:none;stroke:var(--rule);stroke-width:1.2}
.pipon{fill:var(--hist)}
.head{fill:var(--panel);stroke:var(--rule);stroke-width:1.2}
.headon{fill:none;stroke:var(--fc);stroke-width:2}
.sig{fill:none;stroke:var(--fc);stroke-width:3.5;stroke-linecap:round;opacity:0}
.hatch{stroke:var(--muted);stroke-opacity:.45;stroke-width:1}
""" + "\n".join(css_anims) + """
@media (prefers-reduced-motion:reduce){*{animation:none!important}.sig{opacity:0}}
"""

defs = """<defs>
<marker id="ahead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 Z" style="fill:var(--muted)"/></marker>
<marker id="ahead-fc" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 Z" style="fill:var(--fc)"/></marker>
<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" class="hatch"/></pattern>
</defs>"""

body = "\n".join(o).replace(' class2=""', "").replace(' class3=""', "")
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
       f'aria-labelledby="t d">\n<title id="t">TSFM PatchTransformer V1 architecture</title>\n'
       f'<desc id="d">A univariate history is normalized and cut into 32-step patches (values plus observation mask), '
       f'embedded to 192 dimensions, passed through 8 bidirectional pre-norm Transformer blocks, and the last patch token '
       f'is decoded into 192 future steps times 9 quantiles.</desc>\n'
       f'<style>{style}</style>\n{defs}\n{body}\n</svg>\n')
open(OUT, "w", encoding="utf-8").write(svg)
print(len(svg), "bytes")
