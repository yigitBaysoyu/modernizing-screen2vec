# This file can draw JSONs from the dataset.

import os
import json
import argparse
from PIL import Image, ImageDraw


def parse_and_draw(node, draw):
    # Rico bounds format is usually [left, top, right, bottom]
    if 'bounds' in node:
        b = node['bounds']
        # If bounds are in string format (e.g. "[0,0][1440,2560]"), parse them:
        if isinstance(b, str):
            b = b.replace('[', '').replace(']', ',').strip(',')
            b = [int(x) for x in b.split(',') if x]

        if len(b) == 4:
            # Draw the bounding box for the UI element
            draw.rectangle(b, outline="red", width=2)

            # Label with node class and text if present
            labels = []
            node_class = node.get('class')
            if node_class:
                # Keep class name short (remove package prefix)
                node_class = node_class.split('.')[-1]
                labels.append(node_class)

            node_text = node.get('text')
            if node_text:
                labels.append(f"'{node_text}'")

            if labels:
                label_text = " | ".join(labels)
                draw.text((b[0] + 5, b[1] + 5), label_text, fill="blue")

    # Recursively traverse child nodes
    if 'children' in node and isinstance(node['children'], list):
        for child in node['children']:
            parse_and_draw(child, draw)


def create_ui_preview(json_path, output_image_path, screen_size=(1440, 2560)):
    # 1. Load JSON data
    if not os.path.exists(json_path):
        print(f"Error: File not found at {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON: {e}")
            return

    # 2. Extract screen size from the root bounds if available
    # Locate actual UI root in RICO format
    if 'activity' in data and 'root' in data['activity']:
        ui_root = data['activity']['root']
    else:
        ui_root = data

    if isinstance(ui_root, dict) and 'bounds' in ui_root:
        b = ui_root['bounds']
        if isinstance(b, str):
            b = b.replace('[', '').replace(']', ',').strip(',')
            b = [int(x) for x in b.split(',') if x]
        if isinstance(b, list) and len(b) == 4:
            screen_size = (int(b[2]) - int(b[0]), int(b[3]) - int(b[1]))

    # Ensure valid dimensions
    if screen_size[0] <= 0 or screen_size[1] <= 0:
        screen_size = (1440, 2560)

    # 3. Create a blank canvas
    image = Image.new("RGB", screen_size, color="white")
    draw = ImageDraw.Draw(image)

    # 4. Traverse tree and draw bounding boxes
    parse_and_draw(ui_root, draw)

    # 5. Save output image
    image.save(output_image_path, "JPEG")
    print(f"Preview successfully saved to: {output_image_path}")


def main():
    parser = argparse.ArgumentParser(description="Render a wireframe of a RICO UI JSON layout.")
    parser.add_argument("--json", type=str, required=True, help="Path to the input RICO JSON file.")
    parser.add_argument("--output", type=str,
                        help="Path to save the generated preview image (default: <json_name>_preview.jpg).")
    parser.add_argument("--width", type=int, default=1440, help="Default canvas width if not specified in root bounds.")
    parser.add_argument("--height", type=int, default=2560,
                        help="Default canvas height if not specified in root bounds.")

    args = parser.parse_args()

    output_path = args.output
    if not output_path:
        base_name = os.path.splitext(os.path.basename(args.json))[0]
        output_path = f"{base_name}_preview.jpg"

    create_ui_preview(args.json, output_path, screen_size=(args.width, args.height))


if __name__ == "__main__":
    main()
