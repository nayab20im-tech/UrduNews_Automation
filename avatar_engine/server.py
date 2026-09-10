"""
avatar_engine/server.py
==========================
Optional FastAPI wrapper (Phase 17).  NOT a hard dependency of the
core CLI — import fails gracefully when FastAPI is not installed.

    POST /generate  {"script": "...", "anchor_id": "female"}
    →  {"success": true, "output": "..."}

Run:  python -m avatar_engine.server   (uvicorn, port 8100)
"""

from __future__ import annotations

from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "FastAPI not installed; optional server skipped "
        "(pip install fastapi uvicorn)"
    ) from exc

from . import config as cfg
from .errors import AvatarEngineError
from .pipeline import generate_anchor_video

app = FastAPI(title="avatar_engine", version="1.0.0")


class GenerateRequest(BaseModel):
    script: str
    anchor_id: str = "female"
    output_path: str | None = None


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    out = req.output_path or str(
        cfg.OUTPUT_DIR / f"api_{req.anchor_id}.mp4")
    try:
        report = generate_anchor_video(
            script=req.script, anchor_id=req.anchor_id, output_path=out)
    except AvatarEngineError as exc:
        raise HTTPException(status_code=500, detail=f"{exc.stage}: {exc}")
    return {"success": report["success"], "output": report["video"],
            "report": report}


def main() -> None:
    import uvicorn
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8100)


if __name__ == "__main__":
    main()
