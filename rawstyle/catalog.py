"""Read final develop settings from a Lightroom Classic catalog (read-only)."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
import sqlite3
from urllib.parse import quote

# The targets are Lightroom develop values, NOT white-balance modes or heuristics.
# Only apply settings that this model actually learned.
TARGETS = (
    "Temperature", "Tint", "Exposure2012", "Contrast2012", "Highlights2012",
    "Shadows2012", "Whites2012", "Blacks2012", "Texture", "Clarity2012",
    "Dehaze", "Vibrance", "Saturation", "Sharpness", "SharpenDetail",
    "SharpenEdgeMasking", "ColorNoiseReduction", "LuminanceSmoothing",
    "PostCropVignetteAmount",
)
DISPLAY = {
    "Temperature": "Temperature (K)", "Tint": "Tint", "Exposure2012": "Exposure",
    "Contrast2012": "Contrast", "Highlights2012": "Highlights", "Shadows2012": "Shadows",
    "Whites2012": "Whites", "Blacks2012": "Blacks", "Clarity2012": "Clarity",
}
LIMITS = {
    "Temperature": (2000, 50000), "Tint": (-150, 150), "Exposure2012": (-5, 5),
    "Sharpness": (0, 150), "SharpenDetail": (0, 100),
    "SharpenEdgeMasking": (0, 100), "ColorNoiseReduction": (0, 100),
    "LuminanceSmoothing": (0, 100),
}


def _skip_string(text: str, offset: int) -> tuple[str, int]:
    out = []
    i = offset + 1
    while i < len(text):
        c = text[i]
        if c == '"':
            return "".join(out), i + 1
        if c == "\\" and i + 1 < len(text):
            out.append(text[i + 1])
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out), i


def _skip_table(text: str, offset: int) -> int:
    depth, i = 0, offset
    while i < len(text):
        if text[i] == '"':
            _, i = _skip_string(text, i)
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return i


def parse_develop_table(text: str) -> dict[str, float]:
    """Extract top-level numeric sliders from `s = { ... }` WITHOUT executing Lua.

    The nested Look, profile and mask tables must not overwrite top-level values.
    """
    if not text:
        return {}
    m = re.search(r"\bs\s*=\s*\{", text)
    if not m:
        return {}
    result = {}
    i = m.end()
    while i < len(text):
        while i < len(text) and text[i] in " \t\r\n,":
            i += 1
        if i >= len(text) or text[i] == "}":
            break
        key_match = re.match(r"([A-Za-z_][A-Za-z_0-9]*)\s*=", text[i:])
        if not key_match:
            i = text.find(",", i)
            if i < 0:
                break
            i += 1
            continue
        key = key_match.group(1)
        i += key_match.end()
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        if text[i] == "{":
            i = _skip_table(text, i)
            continue
        if text[i] == '"':
            _, i = _skip_string(text, i)
            continue
        start = i
        while i < len(text) and text[i] not in ",}\r\n":
            i += 1
        if key in TARGETS:
            try:
                result[key] = float(text[start:i].strip())
            except ValueError:
                pass
    return result


def read_catalog(path: str | Path) -> list[dict]:
    """Produce one label record per filename. Never write to the .lrcat file."""
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != ".lrcat" or not path.is_file():
        raise ValueError("Select an existing Lightroom Classic .lrcat file.")
    uri = "file:" + quote(str(path), safe="/") + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    try:
        rows = conn.execute("""
            SELECT f.baseName, f.extension, d.text
            FROM Adobe_images AS i
            JOIN Adobe_imageDevelopSettings AS d ON d.image = i.id_local
            JOIN AgLibraryFile AS f ON f.id_local = i.rootFile
            WHERE d.text IS NOT NULL
        """).fetchall()
    except sqlite3.Error as exc:
        raise ValueError("This catalog does not have the expected Lightroom Classic tables.") from exc
    finally:
        conn.close()
    result = []
    for stem, extension, lua in rows:
        settings = parse_develop_table(lua)
        if stem and settings.get("Temperature") and "Tint" in settings:
            result.append({
                "name": f"{stem}.{extension}".casefold(),
                "settings": settings,
            })
    if not result:
        raise ValueError("No numeric temperature/tint labels were found in this catalog.")
    return result


def pair_raws(catalog: list[dict], paths: list[Path]) -> tuple[list[Path], list[dict], dict]:
    """Match by unique filename (including extension), and skip any ambiguity."""
    files = defaultdict(list)
    for path in paths:
        files[path.name.casefold()].append(path)
    labels = defaultdict(list)
    for record in catalog:
        labels[record["name"]].append(record["settings"])
    matched_paths, matched_labels = [], []
    ambiguous = 0
    for name in sorted(set(files).intersection(labels)):
        if len(files[name]) != 1 or len(labels[name]) != 1:
            ambiguous += 1
            continue
        matched_paths.append(files[name][0])
        matched_labels.append(labels[name][0])
    return matched_paths, matched_labels, {
        "catalog_photos": len(catalog),
        "raw_photos": len(paths),
        "matched": len(matched_paths),
        "ambiguous_names": ambiguous,
    }
