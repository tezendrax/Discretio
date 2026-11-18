#!/usr/bin/env python3
"""Fetch only selected video members from the official uncompressed IDD-PeD tars.

The CVIT server supports HTTP byte ranges. This reader lets tarfile scan headers
and seek over unrelated video payloads without downloading those payloads.
Only selected full videos are written locally; trim them to manifest windows
after download.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import BinaryIO

import requests


BASE_URL = "https://cvit.iiit.ac.in/images/datasets/IDDPed/Videos"


class HttpRangeReader(io.RawIOBase):
    """Seekable read-only file backed by cached HTTP Range blocks."""

    def __init__(self, url: str, block_size: int = 1024 * 1024):
        super().__init__()
        self.url = url
        self.block_size = block_size
        self.session = requests.Session()
        head = self.session.head(url, allow_redirects=True, timeout=60)
        head.raise_for_status()
        self.size = int(head.headers["Content-Length"])
        if "bytes" not in head.headers.get("Accept-Ranges", "").lower():
            raise RuntimeError(f"Server does not advertise byte ranges: {url}")
        self.pos = 0
        self.cache_start = -1
        self.cache = b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_pos = offset
        elif whence == io.SEEK_CUR:
            new_pos = self.pos + offset
        elif whence == io.SEEK_END:
            new_pos = self.size + offset
        else:
            raise ValueError("Invalid whence")
        if new_pos < 0:
            raise ValueError("Negative seek position")
        self.pos = new_pos
        return self.pos

    def _load_block(self, position: int) -> None:
        start = (position // self.block_size) * self.block_size
        end = min(self.size - 1, start + self.block_size - 1)
        response = self.session.get(
            self.url,
            headers={"Range": f"bytes={start}-{end}"},
            timeout=(30, 120),
        )
        if response.status_code != 206:
            raise RuntimeError(
                f"Range request returned HTTP {response.status_code}; "
                "refusing to download the full archive."
            )
        self.cache_start = start
        self.cache = response.content

    def read(self, size: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        if size is None or size < 0:
            size = self.size - self.pos
        size = min(size, self.size - self.pos)
        out = bytearray()
        while size:
            if not (self.cache_start <= self.pos < self.cache_start + len(self.cache)):
                self._load_block(self.pos)
            in_cache = self.pos - self.cache_start
            take = min(size, len(self.cache) - in_cache)
            if take <= 0:
                raise IOError("Empty HTTP range response")
            out.extend(self.cache[in_cache:in_cache + take])
            self.pos += take
            size -= take
        return bytes(out)

    def close(self) -> None:
        self.session.close()
        super().close()


def load_video_ids(clips_json: Path) -> dict[str, set[str]]:
    clips = json.loads(clips_json.read_text())
    groups: dict[str, set[str]] = {}
    for clip in clips:
        video_id = str(clip["video_id"])
        match = re.match(r"^(gp_set_\d+)_vid_", video_id)
        if not match:
            raise ValueError(f"Not an IDD-PeD full-video ID: {video_id}")
        groups.setdefault(match.group(1), set()).add(video_id)
    return groups


def extract_wanted_set(set_name: str, wanted: set[str], out_dir: Path) -> None:
    url = f"{BASE_URL}/{set_name}.tar"
    print(f"\n{set_name}: opening remote tar index ({len(wanted)} wanted video(s))")
    reader = HttpRangeReader(url)
    found: dict[str, object] = {}
    import tarfile
    from tqdm import tqdm

    with reader, tarfile.open(fileobj=reader, mode="r:") as archive:
        members = archive.getmembers()
        for member in members:
            if member.isfile() and PurePosixPath(member.name).stem in wanted:
                found[PurePosixPath(member.name).stem] = member

        missing = wanted - set(found)
        if missing:
            examples = [m.name for m in members if m.isfile()][:8]
            raise FileNotFoundError(
                f"{set_name}: missing {sorted(missing)}. Archive examples: {examples}"
            )

        for video_id in sorted(wanted):
            member = found[video_id]
            suffix = PurePosixPath(member.name).suffix or ".mp4"
            destination = out_dir / f"{video_id}{suffix}"
            if destination.exists() and destination.stat().st_size == member.size:
                print(f"Already complete: {destination}")
                continue
            source = archive.extractfile(member)
            if source is None:
                raise IOError(f"Cannot read archive member {member.name}")
            print(f"Extracting {member.name} ({member.size / 1e6:.1f} MB)")
            with source, destination.open("wb") as target, tqdm(
                total=member.size, unit="B", unit_scale=True, desc=video_id
            ) as bar:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
                    bar.update(len(chunk))
            if destination.stat().st_size != member.size:
                raise IOError(f"Size mismatch after extraction: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_json", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    groups = load_video_ids(args.clips_json)
    print("Selected sets:", {k: sorted(v) for k, v in groups.items()})
    for set_name, video_ids in groups.items():
        extract_wanted_set(set_name, video_ids, args.out_dir)
    print(f"\nDone. Selected videos saved under {args.out_dir}")


if __name__ == "__main__":
    main()
