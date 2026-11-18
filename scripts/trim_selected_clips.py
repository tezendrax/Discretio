#!/usr/bin/env python3
"""Frame-accurately trim selected IDD-PeD videos and rebase their clip JSON."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from tqdm import tqdm


def probe_fps(path: Path) -> str:
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate", "-of", "default=nw=1:nk=1",
        str(path),
    ], text=True).strip()
    if not raw or raw == "0/0":
        raise RuntimeError(f"Could not determine frame rate for {path}")
    return raw


def trim_one(clip: dict, source_dir: Path, output_dir: Path) -> dict:
    video_id = clip["video_id"]
    candidates = [p for p in source_dir.rglob("*")
                  if p.is_file() and p.stem == video_id]
    if not candidates:
        raise FileNotFoundError(f"No downloaded source video for {video_id}")
    source = candidates[0]
    start = int(clip["start_frame"])
    end = int(clip["end_frame"])
    t0 = int(clip["t0_frame"])
    if not start <= t0 <= end:
        raise ValueError(f"Invalid frame window for {clip['clip_id']}: {start}, {t0}, {end}")

    fps = probe_fps(source)
    output = output_dir / f"{video_id}.mp4"
    # FFmpeg's select filter decodes by source frame index, avoiding timestamp
    # rounding from -ss. The end frame is inclusive, matching the manifest.
    vf = f"select=between(n\\,{start}\\,{end}),setpts=N/({fps})/TB"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-vf", vf, "-fps_mode", "vfr",
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        str(output),
    ], check=True)

    adjusted = dict(clip)
    adjusted["start_frame"] = 0
    adjusted["t0_frame"] = t0 - start
    adjusted["end_frame"] = end - start
    adjusted["source_start_frame"] = start
    adjusted["source_end_frame"] = end
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_json", type=Path, required=True)
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    clips = json.loads(args.clips_json.read_text())

    adjusted = []
    for clip in tqdm(clips, desc="Trimming selected clips", unit="clip"):
        adjusted.append(trim_one(clip, args.source_dir, args.out_dir))

    out_json = args.out_dir / "selected_clips.json"
    out_json.write_text(json.dumps(adjusted, indent=2))
    print(f"\nWrote {len(adjusted)} trimmed clips and rebased manifest: {out_json}")


if __name__ == "__main__":
    main()
