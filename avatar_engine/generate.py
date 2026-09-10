"""
avatar_engine/generate.py
============================
CLI for the avatar subsystem:

    python -m avatar_engine.generate \
        --script generated/script.txt \
        --anchor female \
        --output output/news.mp4

    python -m avatar_engine.generate --anchor male --text "السلام علیکم۔"
    python -m avatar_engine.generate --config config/avatar.yaml ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import AvatarEngineError
from .pipeline import generate_anchor_video


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="avatar_engine.generate",
        description="Generate a lip-synced Urdu news anchor video.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--script", help="Path to a UTF-8 Urdu script file")
    src.add_argument("--text", help="Urdu script text on the command line")
    parser.add_argument("--anchor", choices=("male", "female"),
                        default="female", help="Anchor identity")
    parser.add_argument("--scene", choices=("dual", "single"), default=None,
                        help="dual = both anchors alternate (default); "
                             "single = one centred anchor")
    parser.add_argument("--first-speaker", choices=("male", "female"),
                        default=None, help="Who opens the broadcast (dual)")
    parser.add_argument("--output", help="Output MP4 path")
    parser.add_argument("--config", help="Optional YAML config file")
    args = parser.parse_args(argv)

    config = args.config
    overrides = {k: v for k, v in (
        ("scene_mode", args.scene),
        ("first_speaker", args.first_speaker),
    ) if v}
    if overrides:
        from . import config as cfgmod
        if config:
            config = cfgmod.AvatarConfig.from_yaml(config, **overrides)
        else:
            config = cfgmod.AvatarConfig.from_dict(overrides)

    try:
        report = generate_anchor_video(
            script_path=args.script,
            script=args.text,
            anchor_id=args.anchor,
            output_path=args.output,
            config=args.config,
        )
    except AvatarEngineError as exc:
        print(f"ERROR [{exc.stage}]: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
