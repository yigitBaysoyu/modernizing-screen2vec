"""
render_layout.py  --  visual sanity-check tool for the cleaning stage.

Renders a cleaned screen (the {app, activity, components} object produced by
utils.clean_screen) into a labeled wireframe image, so we can eyeball whether
the cleaned JSON still captures the screen's layout.

No cleaned JSON is ever read from / written to disk: this tool takes a *raw*
Rico .json file, cleans it in memory via utils.clean_screen, and renders the
result straight to a PNG.

Encoding ("typeclick"):
    - outline COLOR  -> coarse component-family hint only (PALETTE; unknown -> gray)
    - outline WEIGHT + corner triangle -> clickable (independent of color)
    - header strip   -> screen function: activity (dark) + app package (gray)
    - label          -> the component's actual class name only (content text is
                        omitted to keep busy screens readable; it stays in the JSON).
                        Detail-bearing nodes get label priority; collisions are
                        resolved by nudging labels down a line so all stay legible.

Coordinates in the cleaned screen are binned 1-100 per axis, so the render canvas
size is arbitrary; the true device aspect ratio was normalized away in cleaning.

Usage:
    python render_layout.py data/rico-examples/4.json            # -> 4.layout.png
    python render_layout.py data/rico-examples/                  # all *.json in dir
    python render_layout.py data/rico-examples/4.json -o out.png
    python render_layout.py data/rico-examples/4.json --width 600 --height 1000
"""

import os
import json
import glob
import argparse

from PIL import Image, ImageDraw, ImageFont

from utils import clean_screen


# Outline color by component family (matched against the class-name suffix).
PALETTE = {
    "TextView": (70, 130, 180),     # text        - steel blue
    "Button": (200, 60, 60),        # button      - red
    "ImageView": (46, 160, 90),     # image/icon  - green
    "ImageButton": (230, 120, 0),   # image button- orange
    "EditText": (120, 80, 180),     # input       - purple
    "CardView": (150, 150, 150),    # card region - gray
    "Toolbar": (160, 140, 185),     # toolbar     - violet
    "ListView": (200, 150, 80),     # list/grid   - tan
    "GridView": (200, 150, 80),
    "RecyclerView": (200, 150, 80),
    "WebView": (0, 150, 150),       # web view    - teal
    "VideoView": (180, 60, 120),    # video       - magenta
}
DEFAULT_COLOR = (150, 150, 150)

# Geometry/fonts are tuned for this reference height and scaled up linearly for
# larger canvases, so higher resolution stays crisp (not just bigger).
BASE_HEIGHT = 960


def _font(px):
    """A crisp, scalable default font (Pillow >= 10.1 supports a size argument)."""
    try:
        return ImageFont.load_default(size=px)
    except TypeError:  # older Pillow: fixed-size bitmap fallback
        return ImageFont.load_default()


def color_for(cls):
    s = (cls or "").split(".")[-1]
    for suffix in sorted(PALETTE, key=len, reverse=True):  # most specific suffix wins
        if s.endswith(suffix):
            return PALETTE[suffix]
    return DEFAULT_COLOR


def _to_pixels(bounds, width, height):
    return [
        bounds[0] / 100 * width,
        bounds[1] / 100 * height,
        bounds[2] / 100 * width,
        bounds[3] / 100 * height,
    ]


def _iter_nodes(components):
    """Yield (node, depth) for every component in the forest."""
    def rec(node, depth):
        yield node, depth
        for child in node.get("children") or []:
            yield from rec(child, depth + 1)
    for comp in components:
        yield from rec(comp, 0)


def render(components, width=1080, height=1920):
    scale = height / BASE_HEIGHT
    font = _font(max(10, round(11 * scale)))
    line_h = round(12 * scale)
    pad = round(3 * scale)
    mark = round(9 * scale)
    w_click = max(2, round(3 * scale))
    w_plain = max(1, round(scale))

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Pass 1: draw boxes (color = type, weight + corner triangle = clickable),
    # collect labels for placement in pass 2.
    labels = []
    for node, depth in _iter_nodes(components):
        bounds = node.get("bounds")
        if not (isinstance(bounds, list) and len(bounds) == 4):
            continue
        cls = node.get("class", "")
        clickable = node.get("clickable") is True
        col = color_for(cls)
        p = _to_pixels(bounds, width, height)

        draw.rectangle(p, outline=col, width=w_click if clickable else w_plain)
        if clickable:
            _, y1, x2, _ = p
            draw.polygon([(x2 - mark, y1), (x2, y1), (x2, y1 + mark)], fill=col)

        # Label = just the component's class. Content text is omitted to keep busy
        # screens readable; the actual text/content-desc/resource-id stays in the JSON.
        detail = node.get("text") or node.get("content-desc") or node.get("resource-id")
        label = cls.split(".")[-1] if cls else ""
        if label:
            labels.append((depth, bool(detail), p[0] + pad, p[1] + pad, label, col))

    # Pass 2: place labels, text-bearing first, nudging down on collision.
    def overlaps(a, b):
        return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

    placed = []
    labels.sort(key=lambda L: (not L[1], L[0]))  # detail-bearing labels first, then shallower
    for _depth, _has_detail, x, y, text, col in labels:
        yy = y
        for _ in range(14):
            box = draw.multiline_textbbox((x, yy), text, font=font)
            if not any(overlaps(box, q) for q in placed):
                break
            yy += line_h
        draw.multiline_text((x, yy), text, fill=col, font=font)
        placed.append(draw.multiline_textbbox((x, yy), text, font=font))

    return img


def _add_header(wireframe, app, activity, width, height):
    """Prepend a header strip showing the screen function (activity) and app package."""
    if not (app or activity):
        return wireframe
    scale = height / BASE_HEIGHT
    f_act = _font(max(11, round(13 * scale)))
    f_app = _font(max(10, round(11 * scale)))
    header_h = round(34 * scale)
    canvas = Image.new("RGB", (width, height + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((round(6 * scale), round(4 * scale)), activity or "(unknown activity)", fill=(20, 20, 20), font=f_act)
    draw.text((round(6 * scale), round(18 * scale)), app, fill=(120, 120, 120), font=f_app)
    draw.line([(0, header_h - 1), (width, header_h - 1)], fill=(200, 200, 200))
    canvas.paste(wireframe, (0, header_h))
    return canvas


def render_screen(screen, width=1080, height=1920):
    """Render an in-memory cleaned screen to a PIL Image.

    `screen` is the {app, activity, components} dict returned by utils.clean_screen
    (a bare components list is also tolerated). Returns the labeled wireframe image;
    the caller decides whether/where to save it.
    """
    if isinstance(screen, dict):
        components = screen.get("components", [])
        app, activity = screen.get("app", ""), screen.get("activity", "")
    else:
        components = screen  # tolerate a bare components list
        app, activity = "", ""

    wireframe = render(components, width, height)
    return _add_header(wireframe, app, activity, width, height)


def output_path(in_path, out_arg):
    base = os.path.splitext(os.path.basename(in_path))[0] + ".layout.png"
    if out_arg is None:
        return os.path.join(os.path.dirname(in_path), base)
    if os.path.isdir(out_arg) or out_arg.endswith(os.sep):
        os.makedirs(out_arg, exist_ok=True)
        return os.path.join(out_arg, base)
    return out_arg


def render_file(in_path, out_arg, width, height):
    # clean the raw Rico screen in memory, render it, and save the PNG preview
    screen = clean_screen(in_path)
    img = render_screen(screen, width, height)
    out_path = output_path(in_path, out_arg)
    img.save(out_path)
    print(f"{os.path.basename(in_path):>22}  ->  {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Render a raw Rico screen (cleaned on the fly) to a labeled layout image.")
    parser.add_argument("input", help="A raw Rico .json file or a directory containing them")
    parser.add_argument("-o", "--out", default=None, help="Output PNG file or directory")
    parser.add_argument("--width", type=int, default=1080, help="Canvas width in px")
    parser.add_argument("--height", type=int, default=1920, help="Canvas height in px")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*.json")))
        files = [f for f in files if not f.endswith(".clean.json")]
        if not files:
            print(f"No .json files found in {args.input}")
            return
        for f in files:
            render_file(f, args.out, args.width, args.height)
    else:
        render_file(args.input, args.out, args.width, args.height)


if __name__ == "__main__":
    main()
