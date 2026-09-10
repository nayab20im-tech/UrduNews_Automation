"""
avatar_engine/prepare_anchors.py
===================================
Derive the two reusable anchor assets (male.png / female.png) from the
canonical two-anchor studio photo and verify them.

    python -m avatar_engine.prepare_anchors [--force]
"""

from __future__ import annotations

import argparse
import sys

from .anchors import prepare_assets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="avatar_engine.prepare_anchors")
    parser.add_argument("--force", action="store_true",
                        help="Re-derive assets even if they exist")
    args = parser.parse_args(argv)
    assets = prepare_assets(force=args.force)
    for anchor_id, path in assets.items():
        print(f"{anchor_id}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
