"""Lipstick shade library for the live mirror's lipstick task: each shade (name, hex color, finish) is rendered as a
swatch on white (a color chip by default, or drawn lips), which becomes the <Picture 1> reference instead of a garment image. The prompt never
names the shade, so every shade shares one token count and switching shades never recompiles."""
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

FINISHES = ("matte", "satin", "gloss")

DEFAULT_SHADES = [
    {"name": "classic_red", "hex": "#B3121F", "finish": "matte"},
    {"name": "nude_beige", "hex": "#C08470", "finish": "satin"},
    {"name": "rosewood", "hex": "#9C4A58", "finish": "satin"},
    {"name": "berry", "hex": "#7A1D3B", "finish": "matte"},
    {"name": "coral", "hex": "#EE6352", "finish": "gloss"},
    {"name": "fuchsia", "hex": "#C2186B", "finish": "gloss"},
    {"name": "plum", "hex": "#5A1E40", "finish": "matte"},
    {"name": "brick", "hex": "#8E3B2B", "finish": "satin"},
]


def load_shades(path=None):
    """Shades from a JSON list of {"name", "hex", "finish"} (finish: matte | satin | gloss), or the defaults."""
    shades = json.loads(Path(path).read_text()) if path else DEFAULT_SHADES
    for s in shades:
        if s.get("finish", "satin") not in FINISHES:
            raise ValueError(f"shade {s['name']}: finish must be one of {FINISHES}")
    return shades


def _hex(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _lip_outlines(cx, cy, half_w, n=120):
    """Upper and lower lip polygons around the mouth line at (cx, cy): cupid's bow on top, fuller lower lip."""
    xs = [-1 + 2 * i / n for i in range(n + 1)]
    top = [-0.42 * (1 - x * x) ** 0.75 + 0.14 * math.exp(-(x / 0.16) ** 2) for x in xs]  # bow dip at the center
    mid = [0.04 * (1 - x * x) for x in xs]  # mouth line, slightly curved
    bot = [0.50 * (1 - x * x) ** 0.85 for x in xs]
    pt = lambda x, y: (cx + x * half_w, cy + y * half_w)  # noqa: E731
    upper = [pt(x, y) for x, y in zip(xs, top)] + [pt(x, y) for x, y in zip(reversed(xs), reversed(mid))]
    lower = [pt(x, y) for x, y in zip(xs, mid)] + [pt(x, y) for x, y in zip(reversed(xs), reversed(bot))]
    return upper, lower


def render_swatch(hex_color, finish="satin", size=1024):
    """Lips filled with the shade on white; the finish shows as the highlight (none / soft sheen / sharp gloss)."""
    ss = 2  # supersample for smooth edges
    S = size * ss
    base = _hex(hex_color)
    img = Image.new("RGB", (S, S), (255, 255, 255))
    upper, lower = _lip_outlines(S / 2, S / 2, S * 0.40)
    mask = Image.new("L", (S, S), 0)
    md = ImageDraw.Draw(mask)
    md.polygon(upper, fill=255)
    md.polygon(lower, fill=255)
    # soft shading: slightly darker toward the mouth line and the corners, so it reads as lips, not a flat logo
    shade = Image.new("RGB", (S, S), base)
    dark = Image.new("RGB", (S, S), tuple(int(v * 0.72) for v in base))
    ramp = Image.new("L", (S, S), 0)
    rd = ImageDraw.Draw(ramp)
    rd.line([(S * 0.12, S / 2 + S * 0.016), (S * 0.88, S / 2 + S * 0.016)], fill=200, width=int(S * 0.03))
    ramp = ramp.filter(ImageFilter.GaussianBlur(S * 0.02))
    shade = Image.composite(dark, shade, ramp)
    img.paste(shade, (0, 0), mask)
    if finish != "matte":
        hl = Image.new("L", (S, S), 0)
        hd = ImageDraw.Draw(hl)
        cx, cy, w = S / 2, S / 2 + S * 0.095, S * 0.40
        if finish == "gloss":  # sharp, bright specular on the lower lip and a small one on the upper lip
            hd.ellipse([cx - w * 0.30, cy - w * 0.045, cx + w * 0.22, cy + w * 0.045], fill=235)
            hd.ellipse([cx + w * 0.18, S / 2 - w * 0.20, cx + w * 0.42, S / 2 - w * 0.15], fill=170)
            hl = hl.filter(ImageFilter.GaussianBlur(S * 0.006))
        else:  # satin: wide, faint sheen
            hd.ellipse([cx - w * 0.38, cy - w * 0.07, cx + w * 0.30, cy + w * 0.07], fill=90)
            hl = hl.filter(ImageFilter.GaussianBlur(S * 0.025))
        hl = Image.composite(hl, Image.new("L", (S, S), 0), mask)
        img = Image.composite(Image.new("RGB", (S, S), (255, 255, 255)), img, hl)
    return img.resize((size, size), Image.Resampling.LANCZOS)


def render_chip(hex_color, finish="satin", size=1024):
    """Shape-free swatch: a rounded color chip on white. Drawn lips as the reference made the model paste those
    lips (shape, size, open mouth) onto the face (B300 test 2026-10-27), so the chip carries only color + finish."""
    ss = 2
    S = size * ss
    base = _hex(hex_color)
    img = Image.new("RGB", (S, S), (255, 255, 255))
    mask = Image.new("L", (S, S), 0)
    m = S * 0.12
    ImageDraw.Draw(mask).rounded_rectangle([m, m, S - m, S - m], radius=S * 0.08, fill=255)
    img.paste(Image.new("RGB", (S, S), base), (0, 0), mask)
    if finish != "matte":
        hl = Image.new("L", (S, S), 0)
        hd = ImageDraw.Draw(hl)
        if finish == "gloss":  # sharp diagonal specular band
            hd.polygon([(S * 0.25, m), (S * 0.40, m), (m, S * 0.40), (m, S * 0.25)], fill=220)
            hl = hl.filter(ImageFilter.GaussianBlur(S * 0.008))
        else:  # satin: broad faint sheen
            hd.ellipse([S * 0.18, S * 0.18, S * 0.62, S * 0.50], fill=80)
            hl = hl.filter(ImageFilter.GaussianBlur(S * 0.05))
        hl = Image.composite(hl, Image.new("L", (S, S), 0), mask)
        img = Image.composite(Image.new("RGB", (S, S), (255, 255, 255)), img, hl)
    return img.resize((size, size), Image.Resampling.LANCZOS)


SWATCH_STYLES = {"chip": render_chip, "lips": render_swatch}


def write_swatches(shades, out_dir, size=1024, style="chip"):
    """Render every shade to <out_dir>/<style>/<name>.png; returns {name: path} in library order."""
    out = Path(out_dir) / style
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for s in shades:
        p = out / f"{s['name']}.png"
        SWATCH_STYLES[style](s["hex"], s.get("finish", "satin"), size).save(p)
        paths[s["name"]] = str(p)
    return paths


if __name__ == "__main__":  # preview: python shades.py [shades.json] [out_dir]
    import sys
    paths = write_swatches(load_shades(sys.argv[1] if len(sys.argv) > 1 else None),
                           sys.argv[2] if len(sys.argv) > 2 else "shades_preview")
    print("\n".join(paths.values()))
