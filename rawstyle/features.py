"""Extract features from RAW sensor values and camera metadata."""
from __future__ import annotations

import io
import math
import json
import subprocess


def extract_raw_features(file_data: bytes | memoryview):
    import numpy as np
    import rawpy

    # rawpy.imread accepts a file-like object; it never reopens the original drive.
    # BytesIO makes a transient copy of the RAW bytes in the worker's RAM.
    with rawpy.imread(io.BytesIO(file_data)) as raw:
        mosaic = raw.raw_image_visible
        pattern = raw.raw_pattern
        if pattern is None or pattern.shape != (2, 2):
            raise ValueError("A 2×2 Bayer RAW sensor pattern is required.")
        h, w = mosaic.shape
        if h < 64 or w < 64:
            raise ValueError("RAW sensor is too small to analyse.")
        # 48 x 48 independent sensor blocks, each collecting RGB from the original
        # RAW Bayer values. No JPEG thumbnail or Lightroom preview is involved.
        ys = np.linspace(0, (h - 2) // 2, 48).astype("int64") * 2
        xs = np.linspace(0, (w - 2) // 2, 48).astype("int64") * 2
        desc = raw.color_desc
        blacks = raw.black_level_per_channel
        white = float(raw.white_level or 16383)
        channels = {"R": [], "G": [], "B": []}
        for dy in range(2):
            for dx in range(2):
                idx = int(pattern[dy, dx])
                letter = chr(desc[idx]).upper() if isinstance(desc, bytes) else str(desc[idx]).upper()
                if letter not in channels:
                    continue
                black = float(blacks[idx] if idx < len(blacks) else 0)
                pixel = mosaic[np.ix_(ys + dy, xs + dx)].astype("float32", copy=False)
                channels[letter].append(np.clip((pixel - black) / max(1, white - black), 0, 1))
        if any(not v for v in channels.values()):
            raise ValueError("Unsupported sensor colour filter layout.")
        cube = np.stack([np.mean(channels[color], axis=0) for color in ("R", "G", "B")], axis=-1)
        # Use camera WB gains only as an INPUT FEATURE, never as the target,
        # a forced output, or a substitute for the editor's final Kelvin.
        wb = raw.camera_whitebalance
        gains = list(wb) if wb is not None else []
        # rawpy does not expose raw.metadata. Read the intended camera metadata
        # from the same in-memory RAW bytes; never reopen the source HDD.
        proc = subprocess.run(["exiftool", "-json", "-n", "-ISO", "-ExposureTime", "-FNumber", "-FocalLength", "-"], input=bytes(file_data), capture_output=True, timeout=60)
        if proc.returncode != 0:
            raise ValueError("ExifTool could not read camera metadata")
        meta = json.loads(proc.stdout)[0]
        metadata = [float(meta.get(key, 0) or 0) for key in ("ISO", "ExposureTime", "FNumber", "FocalLength")]
    # Features focus on exposure, RGB balance, tonal distribution and scene layout.
    v = cube.reshape(-1, 3)
    lum = v @ np.array([0.213, 0.715, 0.072], dtype="float32")
    total = np.maximum(v.sum(axis=1, keepdims=True), 0.00005)
    chroma = v / total
    feats = []
    for channel in (v[:, 0], v[:, 1], v[:, 2], lum):
        feats.extend(np.quantile(channel, [.01, .05, .15, .3, .5, .7, .85, .95, .99]).tolist())
        feats.extend(np.histogram(channel, bins=12, range=(0, 1))[0].astype("float32").__truediv__(len(channel)).tolist())
    for channel in (chroma[:, 0], chroma[:, 1], chroma[:, 2]):
        feats.extend(np.quantile(channel, [.05, .25, .5, .75, .95]).tolist())
    # Demosaic-free, coarse spatial fingerprint; allow subject/scene differences.
    # Downsample 48x48 to 6x6 using sensor data, not 48x48 expanded pixels.
    coarse = cube.reshape(6, 8, 6, 8, 3).mean(axis=(1, 3))
    feats.extend(np.log1p(coarse * 8).ravel().tolist())
    ratio = np.maximum(v.mean(axis=0), 0.00001)
    feats.extend(np.log(ratio / ratio[1]).tolist())
    for gain in (gains + [0, 0, 0, 0])[:4]:
        feats.append(math.log1p(max(0.0, float(gain))))
    iso, shutter, aperture, focal = metadata
    feats.extend([math.log1p(max(0.0, iso)), math.log1p(max(0.0, shutter)),
                  math.log1p(max(0.0, aperture)), math.log1p(max(0.0, focal))])
    return np.nan_to_num(np.asarray(feats, dtype="float32"), nan=0, posinf=0, neginf=0)
