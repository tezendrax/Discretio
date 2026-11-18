#!/usr/bin/env python3
"""Extract short IDD-PeD clips by HTTP byte-range seeking inside official tar files.

Only the selected frame windows are encoded locally. The multi-gigabyte source
videos are never saved to disk; FFmpeg reads ranges through a small local proxy.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import http.server
import json
import re
import subprocess
import tarfile
import threading
import time
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import requests
from tqdm import tqdm

BASE_URL = "https://cvit.iiit.ac.in/images/datasets/IDDPed/Videos"
VIDEO_INDEX: dict[str, dict[str, tuple[str, int, int]]] = {}


class RangeFile:
    def __init__(self, url: str):
        self.url = url
        self.session = requests.Session()
        response = None
        for attempt in range(8):
            try:
                response = self.session.head(url, allow_redirects=True, timeout=60)
                break
            except requests.RequestException:
                if attempt == 7:
                    raise
                time.sleep(attempt + 1)
        response.raise_for_status()
        self.size = int(response.headers["Content-Length"])
        if "bytes" not in response.headers.get("Accept-Ranges", "").lower():
            raise RuntimeError(f"No byte-range support advertised by {url}")
        self.pos = 0
        self.cache_start = -1
        self.cache = b""
        self.block = 1024 * 1024

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if self.pos < 0:
            raise ValueError("Negative seek")
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, size: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        if size < 0:
            size = self.size - self.pos
        size = min(size, self.size - self.pos)
        parts = bytearray()
        while size:
            if not self.cache_start <= self.pos < self.cache_start + len(self.cache):
                start = self.pos // self.block * self.block
                end = min(self.size - 1, start + self.block - 1)
                r = None
                for attempt in range(8):
                    try:
                        r = self.session.get(
                            self.url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                            timeout=(30, 180),
                        )
                        break
                    except requests.RequestException:
                        if attempt == 7:
                            raise
                        time.sleep(attempt + 1)
                if r.status_code != 206:
                    raise RuntimeError(f"Expected HTTP 206, got {r.status_code}; will not fetch full archive")
                self.cache_start, self.cache = start, r.content
            offset = self.pos - self.cache_start
            n = min(size, len(self.cache) - offset)
            if n <= 0:
                raise IOError("Empty range response")
            parts.extend(self.cache[offset:offset + n])
            self.pos += n
            size -= n
        return bytes(parts)

    def close(self):
        self.session.close()


def index_video(set_id: str, video_id: str) -> tuple[str, int, int]:
    if set_id not in VIDEO_INDEX:
        url = f"{BASE_URL}/{set_id}.tar"
        remote = RangeFile(url)
        entries = {}
        try:
            with tarfile.open(fileobj=remote, mode="r:") as archive:
                for member in archive:
                    if member.isfile():
                        entries[PurePosixPath(member.name).stem] = (url, member.offset_data, member.size)
        finally:
            remote.close()
        VIDEO_INDEX[set_id] = entries
    if video_id in VIDEO_INDEX[set_id]:
        return VIDEO_INDEX[set_id][video_id]
    raise FileNotFoundError(f"{video_id} not found in {set_id}.tar")


class RangeProxy(http.server.BaseHTTPRequestHandler):
    mapping: dict[str, tuple[str, int, int]] = {}
    transferred = 0
    transferred_lock = threading.Lock()

    def log_message(self, *_args):
        pass

    def do_HEAD(self):
        item = self.mapping.get(urlparse(self.path).path)
        if item is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(item[2]))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):
        item = self.mapping.get(urlparse(self.path).path)
        if item is None:
            self.send_error(404)
            return
        upstream, base, size = item
        match = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
        if not match:
            self.send_error(416, "A byte range is required")
            return
        start = int(match.group(1))
        end = min(size - 1, int(match.group(2)) if match.group(2) else size - 1)
        if start >= size or end < start:
            self.send_error(416)
            return
        length = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        cursor = base + start
        absolute_end = base + end
        for attempt in range(8):
            if cursor > absolute_end:
                return
            r = None
            try:
                r = requests.get(
                    upstream,
                    headers={"Range": f"bytes={cursor}-{absolute_end}", "Connection": "close", "Accept-Encoding": "identity"},
                    stream=True,
                    timeout=(30, 180),
                )
                if r.status_code != 206:
                    raise RuntimeError(f"Upstream range returned {r.status_code}")
                for chunk in r.iter_content(256 * 1024):
                    if chunk:
                        self.wfile.write(chunk)
                        cursor += len(chunk)
                        with self.transferred_lock:
                            type(self).transferred += len(chunk)
            except (requests.RequestException, RuntimeError):
                if attempt == 7:
                    return
                time.sleep(attempt + 1)
            except (BrokenPipeError, ConnectionResetError):
                return
            finally:
                if r is not None:
                    r.close()


def fps_for(url: str) -> str:
    result = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
        "-of", "default=nw=1:nk=1", url,
    ], text=True, stderr=subprocess.STDOUT, timeout=300).strip()
    if not result or result == "0/0":
        raise RuntimeError(f"Could not read source frame rate ({result!r})")
    return result


def extract_one(clip: dict, output_dir: Path, port: int) -> dict:
    video_id = str(clip["video_id"])
    m = re.match(r"^(gp_set_\d+)_vid_", video_id)
    if not m:
        raise ValueError(f"Unexpected video ID: {video_id}")
    start, end = int(clip["start_frame"]), int(clip["end_frame"])
    t0 = int(clip["t0_frame"])
    if not start <= t0 <= end:
        raise ValueError(f"Invalid frame range for {clip['clip_id']}")
    count = end - start + 1
    output = output_dir / f"{clip['clip_id']}.mp4"
    if output.exists():
        try:
            existing_frames = int(subprocess.check_output([
                "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", str(output),
            ], text=True, stderr=subprocess.DEVNULL).strip())
            if existing_frames == count:
                return rebased_clip(clip, fps_for(str(output)), output)
        except (subprocess.CalledProcessError, ValueError):
            pass
    upstream, offset, size = index_video(m.group(1), video_id)
    route = f"/v/{video_id}"
    RangeProxy.mapping[route] = (upstream, offset, size)
    url = f"http://127.0.0.1:{port}{route}"
    fps = fps_for(url)
    rate = Fraction(fps)
    start_seconds = f"{start * rate.denominator / rate.numerator:.9f}"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", start_seconds, "-i", url, "-frames:v", str(count),
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-movflags", "+faststart",
        str(output),
    ], check=True, timeout=900)
    if not output.exists() or output.stat().st_size == 0:
        raise IOError(f"FFmpeg produced no clip: {output}")
    actual = int(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", str(output),
    ], text=True).strip())
    if actual != count:
        raise RuntimeError(f"Frame-count mismatch for {clip['clip_id']}: expected {count}, got {actual}")
    return rebased_clip(clip, fps, output)


def rebased_clip(clip: dict, fps: str, output: Path) -> dict:
    start, end, t0 = int(clip["start_frame"]), int(clip["end_frame"]), int(clip["t0_frame"])
    adjusted = dict(clip)
    adjusted["start_frame"] = 0
    adjusted["t0_frame"] = t0 - start
    adjusted["end_frame"] = end - start
    adjusted["source_start_frame"] = start
    adjusted["source_end_frame"] = end
    adjusted["source_fps"] = fps
    adjusted["local_video"] = output.name
    return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_json", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--first_n", type=int, default=None, help="Optional small initial trial")
    args = parser.parse_args()
    clips = json.loads(args.clips_json.read_text())
    if args.first_n:
        clips = clips[:args.first_n]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeProxy)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output_clips = []
    try:
        for clip in tqdm(clips, desc="Extracting selected frame windows", unit="clip"):
            before = RangeProxy.transferred
            record = extract_one(clip, args.out_dir, port)
            output_clips.append(record)
            transferred = RangeProxy.transferred - before
            local_path = args.out_dir / record["local_video"]
            print(f"\n{clip['clip_id']}: fetched {transferred / 1e6:.1f} MB; wrote {local_path.stat().st_size / 1e6:.1f} MB")
            manifest = args.out_dir / "selected_clips.json"
            manifest.write_text(json.dumps(output_clips, indent=2, default=str))
            import pandas as pd
            pd.DataFrame(output_clips).to_csv(args.out_dir / "selected_clips.csv", index=False)
    finally:
        server.shutdown()
        server.server_close()
    manifest = args.out_dir / "selected_clips.json"
    manifest.write_text(json.dumps(output_clips, indent=2, default=str))
    import pandas as pd
    pd.DataFrame(output_clips).to_csv(args.out_dir / "selected_clips.csv", index=False)
    print(f"\nWrote {len(output_clips)} clips and manifests under {args.out_dir}")


if __name__ == "__main__":
    main()
