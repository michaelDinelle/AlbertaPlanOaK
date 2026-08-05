"""Record format v4 — Python decoder of the FROZEN codec.

Spec: frontend/src/sim/RECORD_V4_SPEC.md (the single source of truth; the Swift
encoder in Sources/Main.swift and the TS decoder in frontend/src/sim/record4.ts
implement the same document). This module decodes a v4 `actions.bin` back to the
EXACT v3 row shape — np.float32[steps, 16]: ctrl[6], done, pose[9] — so every
Python consumer (fixture generators, physics_parity) runs unchanged on either
format via `load_rows()`.

Torn tails (killed runs) are dropped chunk-by-chunk exactly as the spec says:
short header, missing compBytes, crc mismatch, or inflated size != rawBytes ends
the file at the previous chunk.
"""
from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np

FILE_MAGIC = 0x344B_414F
CHUNK_MAGIC = 0x344B_4843
ROW_F32 = 16


def _decode_chunk(payload: bytes, step_count: int, codec: dict) -> np.ndarray:
    """One inflated chunk payload -> f32[stepCount, 16] (spec 'Payload' order)."""
    rows = np.zeros((step_count, ROW_F32), dtype=np.float32)
    off = 0

    hdr = np.frombuffer(payload, dtype=np.uint8, count=step_count, offset=off)
    off += step_count
    rows[:, 6] = (hdr >> 4) & 1

    ctrl = codec.get("ctrl")
    if ctrl == "index":
        rows[:, 0] = (hdr & 0x0F).astype(np.float32)
    else:
        for i in range(int(ctrl["n"])):
            q = np.frombuffer(payload, dtype=np.uint8, count=step_count, offset=off)
            off += step_count
            rows[:, i] = (q.astype(np.float64) / 127.5 - 1.0).astype(np.float32)

    has_dx = any(f["enc"] == "dx32" for f in codec.get("poseFields", []))
    chunk_base_x = 0.0
    if has_dx:
        (chunk_base_x,) = struct.unpack_from("<d", payload, off)
        off += 8

    for f in codec.get("poseFields", []):
        idx = 7 + int(f["idx"])
        if f["enc"] == "u8":
            q = np.frombuffer(payload, dtype=np.uint8, count=step_count, offset=off)
            off += step_count
            rows[:, idx] = q.astype(np.float32)
        elif f["enc"] == "i16":
            q = np.frombuffer(payload, dtype=np.int16, count=step_count, offset=off)
            off += 2 * step_count
            rows[:, idx] = (q.astype(np.float64) * float(f["scale"])).astype(np.float32)
        elif f["enc"] == "dx32":
            q = np.frombuffer(payload, dtype=np.int32, count=step_count, offset=off)
            off += 4 * step_count
            scale = float(f["scale"])
            recon = chunk_base_x + np.cumsum(q.astype(np.float64) * scale)
            rows[:, idx] = recon.astype(np.float32)
        else:
            raise ValueError(f"unknown pose encoding {f['enc']!r}")

    for k, v in (codec.get("poseConst") or {}).items():
        rows[:, 7 + int(k)] = np.float32(v)

    (n_esc,) = struct.unpack_from("<I", payload, off)
    off += 4
    for _ in range(n_esc):
        (row_idx,) = struct.unpack_from("<I", payload, off)
        off += 4
        rows[row_idx] = np.frombuffer(payload, dtype=np.float32, count=ROW_F32, offset=off)
        off += 64
    return rows


def read_v4(path: Path | str, codec: dict) -> np.ndarray:
    """Decode a v4 actions.bin -> f32[steps, 16] via a sequential header scan."""
    data = Path(path).read_bytes()
    if len(data) < 16:
        return np.zeros((0, ROW_F32), dtype=np.float32)
    magic, version, _env_kind, _codec_version = struct.unpack_from("<4I", data, 0)
    if magic != FILE_MAGIC or version != 4:
        raise ValueError(f"{path}: not a record-v4 file (magic {magic:#x} v{version})")
    chunks: list[np.ndarray] = []
    off = 16
    while off + 32 <= len(data):
        (cmagic, _step_start, step_count, raw_bytes, comp_bytes, crc,
         _done_count) = struct.unpack_from("<7I", data, off)
        if cmagic != CHUNK_MAGIC or off + 32 + comp_bytes > len(data):
            break
        comp = data[off + 32: off + 32 + comp_bytes]
        if zlib.crc32(comp) != crc:
            break
        payload = zlib.decompress(comp, wbits=-15)
        if len(payload) != raw_bytes:
            break
        chunks.append(_decode_chunk(payload, step_count, codec))
        off += 32 + comp_bytes
    if not chunks:
        return np.zeros((0, ROW_F32), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def load_rows(lane_dir: Path | str, count: int | None = None) -> np.ndarray:
    """Load a lane's actions.bin as f32[steps, 16], dispatching on the lane's
    meta.json `recordVersion` (3 = raw f32[16] rows, 4 = the frozen codec)."""
    lane_dir = Path(lane_dir)
    meta = json.loads((lane_dir / "meta.json").read_text())
    if meta.get("recordVersion") == 4:
        rows = read_v4(lane_dir / "actions.bin", meta["codec"])
    else:
        flat = np.fromfile(lane_dir / "actions.bin", dtype=np.float32)
        rows = flat[: (flat.size // ROW_F32) * ROW_F32].reshape(-1, ROW_F32)
    return rows if count is None else rows[:count]
