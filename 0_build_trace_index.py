"""
0_build_trace_index.py

Builds the persistent join between RICO interaction-trace screens and the
RICO layouts corpus that the embeddings are keyed by. Run this once, before
any of the trace-based stages (4_, 5_, 6_, 8_, 9_).

The two RICO downloads number screens independently - trace `gestures.json`
keys are per-trace indices, `data/layouts/<id>.json` ids are global - so they
must be joined on content. See trace_index.py for the join key and the
measurements behind it.

Outputs (next to the chosen --output path):
    trace_layout_index.csv           app,trace,screen_key,layout_id,match_method
    trace_layout_index.meta.json     counts, coverage, hash scheme, provenance
    trace_layout_index_dropped.csv   screens that could not be joined, with reason

Usage:
    python 0_build_trace_index.py
    python 0_build_trace_index.py --min_coverage 0.995
"""

import argparse
import sys
import time

import trace_index


def main():
    parser = argparse.ArgumentParser(
        description="Build the RICO trace-screen -> layouts-id index"
    )
    parser.add_argument(
        "--layouts_dir", type=str, default=trace_index.DEFAULT_LAYOUTS_DIR,
        help="Directory of RICO layout JSONs (the corpus the embeddings "
             "were generated from)",
    )
    parser.add_argument(
        "--traces_dir", type=str, default=trace_index.DEFAULT_TRACES_DIR,
        help="RICO filtered_traces directory",
    )
    parser.add_argument(
        "--output", type=str, default=trace_index.DEFAULT_INDEX_PATH,
        help="Output CSV path (default: trace_layout_index.csv)",
    )
    parser.add_argument(
        "--min_coverage", type=float, default=0.99,
        help="Fail if a smaller fraction of trace screens could be joined. "
             "Guards against a re-downloaded or mismatched RICO copy silently "
             "degrading the join (default: 0.99)",
    )
    parser.add_argument(
        "--workers", type=int, default=16,
        help="Threads used for hashing (default: 16)",
    )
    args = parser.parse_args()

    start = time.time()

    def progress(msg):
        print(msg, flush=True)

    rows, stats, dropped = trace_index.build_index(
        layouts_dir=args.layouts_dir,
        traces_dir=args.traces_dir,
        workers=args.workers,
        progress=progress,
    )

    print("\nJoin results")
    print(f"  layout files                : {stats['layout_files']}")
    print(f"    distinct full-doc hashes  : {stats['layout_distinct_full_hashes']}")
    print(f"    distinct root hashes      : {stats['layout_distinct_root_hashes']}")
    print(f"  trace gesture screens       : {stats['trace_screens']}")
    print(f"  mapped                      : {stats['mapped']} "
          f"({100 * stats['coverage']:.2f}%)")
    print(f"    via full-document hash    : {stats['mapped_full_doc_hash']}")
    print(f"    via activity.root hash    : {stats['mapped_root_hash']}")
    print(f"  dropped                     : {stats['dropped']}")
    print(f"    unhashable trace file     : {stats['dropped_unhashable']}")
    print(f"    no layout match           : {stats['dropped_no_match']}")
    print(f"  distinct layout ids used    : {stats['layout_ids_consumed']}")

    if stats["coverage"] < args.min_coverage:
        print(f"\nERROR: coverage {100 * stats['coverage']:.2f}% is below the "
              f"required {100 * args.min_coverage:.2f}%. Index NOT written.",
              file=sys.stderr)
        print("Check that --layouts_dir and --traces_dir come from the same "
              "RICO release.", file=sys.stderr)
        return 1

    csv_path, meta_path, dropped_path = trace_index.write_index(
        rows, stats, dropped,
        out_path=args.output,
        layouts_dir=args.layouts_dir,
        traces_dir=args.traces_dir,
    )

    print(f"\nWrote {csv_path}")
    print(f"      {meta_path}")
    print(f"      {dropped_path}")
    print(f"Done in {time.time() - start:.1f}s")
    print("\nNext: the trace-based stages now resolve screens through this "
          "index automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
