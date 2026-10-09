# examples/project/slim_campaign.py
"""Copies raw campaign output into ``data/`` without the ``front`` column.

The per-iteration CSVs written by ``ExperimentRunner`` end with a ``front``
column holding the whole archive at every iteration. That column is what the
plotting scripts read, and it is also what makes the raw output too large to
version. Every table in ``results/`` is computed from the other columns and
from the per-run summaries, so the stripped copies are enough to reproduce
them.

Usage (from the repository root):

    python examples/project/slim_campaign.py              # every raw/out_* folder
    python examples/project/slim_campaign.py out_static   # named folders only
"""

import argparse
import csv
import os
import shutil
import sys

from paths import DATA_DIR, RAW_DIR

csv.field_size_limit(10 ** 9)


def slim_folder(name: str) -> int:
    """Writes the stripped copy of ``raw/<name>`` to ``data/<name>``."""
    src, dst = os.path.join(RAW_DIR, name), os.path.join(DATA_DIR, name)
    os.makedirs(dst, exist_ok=True)
    count = 0
    for filename in sorted(os.listdir(src)):
        if not filename.endswith(".out.csv"):
            continue
        src_path = os.path.join(src, filename)
        dst_path = os.path.join(dst, filename)
        if filename.endswith(".summary.out.csv"):
            shutil.copyfile(src_path, dst_path)
        else:
            with open(src_path, newline="") as fin, \
                    open(dst_path, "w", newline="") as fout:
                reader, writer = csv.reader(fin), csv.writer(fout)
                header = next(reader)
                keep = len(header)
                if header[-1].split("[")[0].strip() == "front":
                    keep -= 1
                writer.writerow(header[:keep])
                for row in reader:
                    writer.writerow(row[:keep])
        count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folders", nargs="*",
                    help="folders inside raw/ (default: every out_* folder)")
    args = ap.parse_args()

    if not os.path.isdir(RAW_DIR):
        print(f"No raw campaign output found at {RAW_DIR}")
        sys.exit(1)
    folders = args.folders or sorted(
        d for d in os.listdir(RAW_DIR)
        if d.startswith("out_") and os.path.isdir(os.path.join(RAW_DIR, d)))
    for name in folders:
        print(f"{name}: {slim_folder(name)} files -> "
              f"{os.path.join(DATA_DIR, name)}")


if __name__ == "__main__":
    main()
