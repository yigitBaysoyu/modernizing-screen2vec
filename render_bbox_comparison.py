"""
render_bbox_comparison.py  --  Screen2Vec-style layout figure of a reconstructed
screen next to its ground truth.

Both sides are the *cleaned* screen representation with binned 1-100 coordinates:
    - GROUND TRUTH  : raw Rico layout, cleaned in memory via utils.clean_screen
    - RECONSTRUCTION: the {index}_reconstructed.json written by 3_json_reconstruction.py

Encoding (Screen2Vec-style solid blocks):
    - only LEAF components are drawn (containers would just cover their children)
    - fill color = --color_by, by default the component family read off the class
      name (Text / Image / Button / Input / Toggle / Media / List / Other), i.e.
      exactly what Class Accuracy scores: the same color in the same place on both
      panels means class *and* position were recovered.
The two screens are rendered as separate panels side by side, so each panel shows
one layout on its own and the colors carry component semantics -- not which side
of the comparison a box belongs to.

Usage:
    python render_bbox_comparison.py 10004
    python render_bbox_comparison.py 10004 -o cmp.png --width 420 --height 760
    python render_bbox_comparison.py 10004 --only gt          # single panel
    python render_bbox_comparison.py 10004 --color_by clickable
"""

import os
import json
import argparse

from PIL import Image, ImageDraw, ImageFont

from utils import clean_screen


# Component families, matched against the class-name suffix. Order matters: the
# first match wins, so specific families come before generic ones (an
# "AppCompatImageButton" is a Button, not an Image).
FAMILIES = [
    ("Input",  ("EditText", "AutoCompleteTextView", "SearchView"), (147, 51, 234)),   # purple
    ("Toggle", ("CheckBox", "CheckedTextView", "Switch", "RadioButton", "ToggleButton",
                "SeekBar", "RatingBar", "ProgressBar"), (13, 148, 136)),              # teal
    ("Button", ("Button", "ImageButton", "ActionMenuItemView", "OverflowMenuButton"),
     (234, 88, 12)),                                                                  # orange
    ("Image",  ("ImageView",), (22, 163, 74)),                                        # green
    ("Text",   ("TextView",), (37, 99, 235)),                                         # blue
    ("Media",  ("WebView", "VideoView", "SurfaceView", "MapView"), (192, 38, 211)),   # magenta
    ("List",   ("ListView", "RecyclerView", "GridView", "ViewPager"), (180, 130, 60)),# tan
]
OTHER = ("Other", (148, 163, 184))   # unmatched / obfuscated class names - gray

# Alternative encodings for --color_by; they answer different questions about the
# same screen, so the figure keeps them one flag away.
CLICKABLE_COLORS = {"Clickable": (234, 88, 12), "Not clickable": (100, 116, 139)}
TEXT_COLORS = {"With text": (37, 99, 235), "Without text": (220, 38, 38)}

BASE_HEIGHT = 760              # geometry/fonts are tuned for this height, then scaled
GAP = 0.06                     # gap between panels, as a fraction of one panel's width


def _font(px):
    try:
        return ImageFont.load_default(size=px)
    except TypeError:  # older Pillow: fixed-size bitmap fallback
        return ImageFont.load_default()


def _iter_nodes(components):
    """Yield every component in the forest (depth-first)."""
    def rec(node):
        yield node
        for child in node.get("children") or []:
            yield from rec(child)
    for comp in components or []:
        yield from rec(comp)


def _iter_leaves(components):
    """Yield only the leaf components -- the ones that actually draw pixels."""
    for node in _iter_nodes(components):
        if not node.get("children"):
            yield node


def _to_pixels(bounds, width, height):
    p = [
        bounds[0] / 100 * width,
        bounds[1] / 100 * height,
        bounds[2] / 100 * width,
        bounds[3] / 100 * height,
    ]
    return [min(p[0], p[2]), min(p[1], p[3]), max(p[0], p[2]), max(p[1], p[3])]


def classify(node, color_by):
    """Map a component to its (legend label, fill color) under the chosen encoding."""
    if color_by == "clickable":
        key = "Clickable" if node.get("clickable") else "Not clickable"
        return key, CLICKABLE_COLORS[key]
    if color_by == "text":
        key = "With text" if node.get("text") else "Without text"
        return key, TEXT_COLORS[key]
    cls = (node.get("class") or "").split(".")[-1]
    for name, suffixes, color in FAMILIES:
        if any(cls.endswith(suffix) for suffix in suffixes):
            return name, color
    return OTHER


def render_screen(screen, width, height, color_by, seen):
    """One layout panel: filled leaf boxes on white, framed.

    Records every label it draws in `seen`, so the legend can list only the
    categories that actually occur in this figure.
    """
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    for node in _iter_leaves(screen.get("components")):
        bounds = node.get("bounds")
        if not (isinstance(bounds, list) and len(bounds) == 4):
            continue
        label, color = classify(node, color_by)
        seen[label] = color
        # the hairline white outline keeps adjacent or stacked blocks of the same
        # family from melting into one shape
        draw.rectangle(_to_pixels(bounds, width, height), fill=color, outline=(255, 255, 255))
    draw.rectangle([0, 0, width - 1, height - 1], outline=(160, 160, 160))
    return panel


def _caption(draw, x, y, w, title, screen, font_title, font_sub):
    """Centered panel caption: title plus the screen's app / activity."""
    app, activity = screen.get("app", ""), screen.get("activity", "")
    sub = f"{app} / {activity}".strip(" /")
    for text, font, fill, dy in ((title, font_title, (20, 20, 20), 0),
                                 (sub, font_sub, (120, 120, 120), font_title.size + 3)):
        if not text:
            continue
        tw = draw.textlength(text, font=font)
        draw.text((x + (w - tw) / 2, y + dy), text, fill=fill, font=font)


def _legend(draw, x0, y, font, seen, max_x):
    """Color key, listing only the categories present in this figure."""
    box = font.size
    x = x0
    for label, color in seen.items():
        w = box + 6 + draw.textlength(label, font=font) + 14
        if x + w > max_x:                     # wrap onto the next line
            x, y = x0, y + box + 6
        draw.rectangle([x, y, x + box, y + box], fill=color)
        draw.text((x + box + 6, y), label, fill=(60, 60, 60), font=font)
        x += w


def render_comparison(panels, width, height, color_by="family"):
    """Lay out captioned panels side by side and add the legend underneath.

    `panels` is a list of (title, screen); a single entry renders a single screen.
    """
    scale = height / BASE_HEIGHT
    f_title = _font(max(12, round(14 * scale)))
    f_sub = _font(max(10, round(11 * scale)))
    f_legend = _font(max(11, round(12 * scale)))

    head_h = round(38 * scale)
    legend_h = round(46 * scale)
    margin = round(14 * scale)
    gap = round(width * GAP)

    canvas_w = len(panels) * width + (len(panels) - 1) * gap + 2 * margin
    canvas = Image.new("RGB", (canvas_w, head_h + height + legend_h), "white")
    draw = ImageDraw.Draw(canvas)

    seen = {}
    for i, (title, screen) in enumerate(panels):
        x = margin + i * (width + gap)
        _caption(draw, x, round(4 * scale), width, title, screen, f_title, f_sub)
        canvas.paste(render_screen(screen, width, height, color_by, seen), (x, head_h))

    # legend in the fixed family order, not in drawing order, so the key looks the
    # same from figure to figure
    order = [name for name, _suffixes, _color in FAMILIES] + [OTHER[0]]
    seen = dict(sorted(seen.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99))
    _legend(draw, margin, head_h + height + round(12 * scale), f_legend, seen, canvas_w - margin)
    return canvas


def load_reconstruction(path):
    """Read a reconstructed screen. Raises if the generation is not valid JSON."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            screen = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Unparsable reconstruction {os.path.abspath(path)}: {e}") from e
    if not isinstance(screen, dict):
        return {"app": "", "activity": "", "components": screen}
    return screen


def _stats(screen, color_by):
    """Per-category leaf counts, so the console mirrors what the figure shows."""
    leaves = list(_iter_leaves(screen.get("components")))
    counts = {}
    for node in leaves:
        label, _color = classify(node, color_by)
        counts[label] = counts.get(label, 0) + 1
    breakdown = ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
    return f"{len(leaves)} leaves: {breakdown}" if leaves else "0 leaves"


def main():
    parser = argparse.ArgumentParser(
        description="Render ground truth and reconstruction side by side as Screen2Vec-style layout figures.")
    parser.add_argument("index", help="Item index, e.g. 0 -> 0.json / 0_reconstructed.json")
    parser.add_argument("--recon_dir", default="reconstructions_4x4096_avg__via__embeddings_4x4096_avg_full_qwen2.5-1.5b-new",
                        help="Directory holding the {index}_reconstructed.json files")
    parser.add_argument("--layouts_dir", default="../rico-data/data/layouts",
                        help="Directory holding the raw Rico layouts")
    parser.add_argument("--only", choices=("gt", "pred"), default=None,
                        help="Render a single panel instead of both")
    parser.add_argument("--color_by", choices=("family", "clickable", "text"), default="family",
                        help="What the fill color encodes (default: component family)")
    parser.add_argument("-o", "--out", default=None, help="Output PNG (default: {index}_bbox.png)")
    parser.add_argument("--width", type=int, default=420, help="Width of one panel in px")
    parser.add_argument("--height", type=int, default=760, help="Height of one panel in px")
    args = parser.parse_args()

    gt_path = os.path.join(args.layouts_dir, f"{args.index}.json")
    pred_path = os.path.join(args.recon_dir, f"{args.index}_reconstructed.json")

    panels = []
    if args.only != "pred":
        assert os.path.exists(gt_path), f"Could not find ground truth: {gt_path}"
        gt = clean_screen(gt_path)      # raw Rico -> binned, cleaned representation
        panels.append(("Ground Truth", gt))
        print(f"GT    : {os.path.abspath(gt_path)}  ({_stats(gt, args.color_by)})")
    if args.only != "gt":
        assert os.path.exists(pred_path), f"Could not find reconstruction: {pred_path}"
        pred = load_reconstruction(pred_path)
        panels.append(("Reconstruction", pred))
        print(f"PRED  : {os.path.abspath(pred_path)}  ({_stats(pred, args.color_by)})")

    img = render_comparison(panels, args.width, args.height, args.color_by)
    out = os.path.abspath(args.out or f"{args.index}_bbox.png")
    img.save(out)
    print(f"SAVED : {out}")


if __name__ == "__main__":
    main()
