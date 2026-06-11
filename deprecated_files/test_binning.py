import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_max_coordinates(filepath):
    """
    Parses a single RICO JSON file and finds the absolute maximum
    X and Y coordinates across all elements in the UI tree.
    """
    if not os.path.exists(filepath):
        return None, None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return None, None

    max_x = 0
    max_y = 0

    def traverse(node):
        nonlocal max_x, max_y
        if not isinstance(node, dict):
            return

        if 'bounds' in node:
            b = node['bounds']
            # Bounds are [x1, y1, x2, y2]. We care about the maximum reach (x2, y2).
            if len(b) == 4:
                try:
                    max_x = max(max_x, int(b[2]))
                    max_y = max(max_y, int(b[3]))
                except (ValueError, TypeError):
                    pass

        if 'children' in node:
            for child in node['children']:
                traverse(child)

    # Handle the RICO wrapping structure we fixed earlier
    if 'activity' in data and 'root' in data['activity']:
        ui_root = data['activity']['root']
    else:
        ui_root = data

    traverse(ui_root)

    return max_x, max_y


def main():
    parser = argparse.ArgumentParser(description="Analyze RICO dataset screen sizes for Binning.")
    parser.add_argument("--dir", type=str, default="../rico-data/data/combined/",
                        help="Path to the directory containing RICO JSON files.")
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print(f"Error: Directory '{args.dir}' not found.")
        return

    print(f"Scanning directory: {args.dir}")
    json_files = [f for f in os.listdir(args.dir) if f.endswith('.json')]

    if not json_files:
        print("No JSON files found in the specified directory.")
        return

    results = []

    # Process all files
    for filename in tqdm(json_files, desc="Parsing JSONs"):
        filepath = os.path.join(args.dir, filename)
        w, h = get_max_coordinates(filepath)

        if w is not None and h is not None and (w > 0 or h > 0):
            results.append({"filename": filename, "max_x": w, "max_y": h})

    if not results:
        print("Could not extract valid coordinates from the files.")
        return

    # Create DataFrame for statistics
    df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("                 SCREEN SIZE DESCRIPTIVE STATISTICS")
    print("=" * 60)

    # Display the Pandas descriptive stats (count, mean, std, min, 25%, 50%, 75%, max)
    print("\n--- Statistics for Width (Max X) ---")
    print(df['max_x'].describe(percentiles=[.05, .25, .5, .75, .95, .99]).to_string())

    print("\n--- Statistics for Height (Max Y) ---")
    print(df['max_y'].describe(percentiles=[.05, .25, .5, .75, .95, .99]).to_string())

    print("\n" + "=" * 60)
    print("                         SUMMARY FOR BINNING")
    print("=" * 60)
    absolute_max_x = df['max_x'].max()
    absolute_max_y = df['max_y'].max()

    print(f"Absolute Largest X found : {absolute_max_x} pixels")
    print(f"Absolute Largest Y found : {absolute_max_y} pixels")
    print(f"99th Percentile X        : {df['max_x'].quantile(0.99):.0f} pixels")
    print(f"99th Percentile Y        : {df['max_y'].quantile(0.99):.0f} pixels")

    # Check for extreme outliers
    outliers_y = df[df['max_y'] > df['max_y'].quantile(0.99)]
    if not outliers_y.empty:
        print(f"\nWarning: Found {len(outliers_y)} files with extreme Y coordinates (likely scrollable lists).")
        print("Top 3 largest outliers:")
        print(outliers_y.sort_values(by='max_y', ascending=False).head(3).to_string(index=False))

    # Generate a visual distribution plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(df['max_x'], bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Max X Coordinates (Width)')
    plt.xlabel('Pixels')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)

    plt.subplot(1, 2, 2)
    plt.hist(df['max_y'], bins=50, color='salmon', edgecolor='black')
    plt.title('Distribution of Max Y Coordinates (Height)')
    plt.xlabel('Pixels')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)

    plt.tight_layout()
    plot_filename = "screen_sizes_distribution.png"
    plt.savefig(plot_filename)
    print(f"\nSaved distribution histograms to '{plot_filename}'")
    print("=" * 60)


if __name__ == "__main__":
    main()