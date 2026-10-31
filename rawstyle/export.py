"""Produce an import file for the Lightroom Classic Lua plug-in."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from .catalog import TARGETS


def create_predictions_file(pairs, output_dir) -> Path:
    """Export a sequence of (RAW path, predicted settings) pairs.

    UTF-8 paths use percent-encoding so TSV fields cannot be broken by tabs,
    newlines, or unusual filenames; Lua plug-in decodes the path bytes.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = path / f"RAWStyle_Predictions_{stamp}.rawstyle"
    targets = [key for key in TARGETS if any(key in s for _, s in pairs)]
    with out.open("w", encoding="utf-8", newline="\n") as fp:
        fp.write("#RAWSTYLE\t2\n")
        fp.write("path\t" + "\t".join(targets) + "\n")
        for raw_path, settings in pairs:
            escaped = quote(str(Path(raw_path).resolve()), safe="/:.")
            values = [escaped] + [str(settings.get(key, "")) for key in targets]
            fp.write("\t".join(values) + "\n")
    return out
