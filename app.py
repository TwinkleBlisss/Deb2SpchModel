"""Flask entry‑point housing the single public route `/separate`."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from flask import Flask, abort, jsonify, request

from config import SEPARATED_SUBDIR
from preprocess import check_audio_file, ffmpeg_convert
from separator import separator


def _json_error(status: int, message: str):
    return jsonify({"error": message}), status


def create_app() -> Flask:
    app = Flask(__name__)

    @app.errorhandler(400)
    def _bad_request(e):
        return _json_error(400, str(e))

    @app.errorhandler(500)
    def _server_error(e):
        return _json_error(500, str(e))

    @app.post("/separate")
    def separate_route():
        # Parse & validate request body
        data = request.get_json(force=True, silent=True)
        if not data or "path" not in data:
            abort(400, "JSON payload must contain 'path'")
        in_path = Path(data["path"]).expanduser().resolve()
        check_audio_file(in_path)

        # Pre‑processing
        preproc_path = ffmpeg_convert(in_path)

        # Inference
        out_dir = in_path.parent / SEPARATED_SUBDIR / preproc_path.stem
        try:
            stem_paths: List[Path] = separator.separate(preproc_path, out_dir)
        except Exception as exc:  # pragma: no cover — model error path
            logging.exception("Inference failed: %s", exc)
            abort(500, "Model inference failed – see server logs")
        finally:
            preproc_path.unlink(missing_ok=True)  # tidy up temp file

        return jsonify({"separated_paths": [str(p) for p in stem_paths]})

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_app().run(host="0.0.0.0", port=8000, threaded=True)
