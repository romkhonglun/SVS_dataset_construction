#!/usr/bin/env python3
"""
Chạy MFA align từng bài hát riêng biệt thay vì toàn bộ dataset.

Usage:
    python run_mfa_align.py dataset/mfa_corpus dataset/mfa_aligned

Hoặc chạy trực tiếp MFA CLI:
    mfa align dataset/mfa_corpus/SongName vietnamese_mfa vietnamese_mfa dataset/mfa_aligned/SongName --verbose --num_jobs 1
"""
import argparse
import subprocess
import sys
from pathlib import Path


def align_one_song(corpus_song_dir: Path, output_song_dir: Path, dictionary: str, model: str) -> bool:
    """Chạy MFA align cho 1 bài hát."""
    output_song_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mfa", "align",
        str(corpus_song_dir),
        dictionary,
        model,
        str(output_song_dir),
        "--verbose",
        "--num_jobs", "1",
    ]

    print(f"\n  Running MFA align: {corpus_song_dir.name}", flush=True)
    print(f"    Output: {output_song_dir}", flush=True)

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [FAIL] {corpus_song_dir.name}")
        print(f"    stderr: {result.stderr[:200]}")
        return False

    print(f"    [OK] {corpus_song_dir.name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run MFA align per song")
    parser.add_argument("corpus_dir", type=Path, help="Thư mục corpus MFA (chứa sub-dir từng bài)")
    parser.add_argument("output_dir", type=Path, help="Thư mục output TextGrid")
    parser.add_argument("--dictionary", default="vietnamese_mfa", help="Từ điển MFA")
    parser.add_argument("--model", default="vietnamese_mfa", help="Acoustic model MFA")
    args = parser.parse_args()

    if not args.corpus_dir.exists():
        print(f"[ERROR] Corpus không tồn tại: {args.corpus_dir}")
        sys.exit(1)

    song_dirs = sorted([d for d in args.corpus_dir.iterdir() if d.is_dir()])
    print(f"Tổng số bài: {len(song_dirs)}")

    success = 0
    fail = 0
    for i, song_dir in enumerate(song_dirs, 1):
        output_song = args.output_dir / song_dir.name
        if (output_song / "done.txt").exists():
            print(f"[{i}/{len(song_dirs)}] Skip (đã xử lý): {song_dir.name}")
            continue

        ok = align_one_song(song_dir, output_song, args.dictionary, args.model)
        if ok:
            (output_song / "done.txt").touch()
            success += 1
        else:
            fail += 1

    print(f"\n=== Hoàn thành: {success} thành công, {fail} lỗi ===")


if __name__ == "__main__":
    main()
