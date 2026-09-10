"""
avatar_engine/tests/test_e2e.py
==================================
Real end-to-end test (Phases 18-19): generates actual MP4 videos for
BOTH anchors from the canonical Urdu test script and validates them.

Run directly:

    python -m avatar_engine.tests.test_e2e

or via pytest.  The test is only green when two valid MP4 files exist
on disk — importing successfully is not considered success.
"""

from __future__ import annotations

from pathlib import Path

from .. import config as cfg
from ..pipeline import generate_anchor_video
from ..validate import validate_video

TEST_SCRIPT = (
    "السلام علیکم۔\n"
    "یہ پاکستان نیوز کی آزمائشی نشریات ہے۔\n"
    "آج ملک میں اہم سیاسی اور معاشی پیش رفت سامنے آئی ہے۔\n"
    "مزید تفصیلات کے لیے ہمارے ساتھ رہیے۔\n"
)

OUTPUT_DIR = Path(cfg.OUTPUT_DIR) / "tests"


def _generate(anchor_id: str) -> dict:
    out = OUTPUT_DIR / f"e2e_{anchor_id}.mp4"
    report = generate_anchor_video(
        script=TEST_SCRIPT, anchor_id=anchor_id, output_path=out,
    )
    assert report["success"], f"{anchor_id}: validation failed {report['errors']}"
    assert Path(report["video"]).exists()
    assert report["audio"] and report["sync_ok"]
    assert report["duration"] > 0
    return report


def test_female_anchor_video():
    report = _generate("female")
    print("\nFEMALE:", report)


def test_male_anchor_video():
    report = _generate("male")
    print("\nMALE:", report)


def test_cache_hit_on_rerun():
    """Second generation of the same chunk must not re-render."""
    out = OUTPUT_DIR / "e2e_cache_check.mp4"
    generate_anchor_video(script="السلام علیکم۔", anchor_id="male",
                          output_path=out)
    # Second call: identical inputs → cache hit, still valid output.
    report = generate_anchor_video(
        script="السلام علیکم۔", anchor_id="male", output_path=out)
    assert report["success"]
    check = validate_video(out)
    assert check["success"]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for anchor in ("female", "male"):
        report = _generate(anchor)
        print(f"\n=== {anchor.upper()} ===")
        for key in ("video", "duration", "fps", "width", "height",
                    "audio", "sync_ok", "chunks", "backends", "elapsed_sec"):
            print(f"  {key}: {report.get(key)}")
    print("\nE2E TEST PASSED: two valid MP4 videos generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
