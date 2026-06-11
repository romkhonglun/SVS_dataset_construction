"""
Batch Forced Alignment runner với multiprocessing.

Mỗi worker process tạo ForcedAligner instance riêng —
an toàn với multiprocessing (không dùng global state).

Usage:
    pixi run python src/svs_dataset_contruction/run_batch_fa.py --workers 4
    pixi run python src/svs_dataset_contruction/run_batch_fa.py --dataset_dir my_dataset --workers 2
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from svs_dataset_contruction.aligners.mfa import ForcedAligner
from svs_dataset_contruction.config import AlignerConfig


def _worker(args: tuple[Path, Path, Path]) -> tuple[str, bool, str | None]:
    """
    Worker function chạy trong subprocess riêng.

    Mỗi worker tạo ForcedAligner instance riêng — tránh global state,
    an toàn khi chạy song song với ProcessPoolExecutor.

    Args:
        args: (audio_path, transcript_path, output_path)

    Returns:
        (filename, success, error_message_or_None)
    """
    audio_path, transcript_path, output_path = args
    try:
        transcript = transcript_path.read_text(encoding="utf-8").strip()
        aligner = ForcedAligner()  # instance riêng cho mỗi worker process
        aligner.align_to_file(audio_path, transcript, output_path)
        return (audio_path.name, True, None)
    except Exception as e:
        return (audio_path.name, False, str(e))


def main(dataset_dir: Path = Path("dataset"), max_workers: int = 4) -> None:
    vocals_dir    = dataset_dir / "vocals"
    raw_lyric_dir = dataset_dir / "raw_lyric"
    output_dir    = dataset_dir / "textgrid"
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(vocals_dir.glob("*.wav"))
    tasks: list[tuple[Path, Path, Path]] = []

    for audio_path in audio_files:
        transcript_path = raw_lyric_dir / f"{audio_path.stem}.txt"
        output_path = output_dir / f"{audio_path.stem}.TextGrid"

        if not transcript_path.exists():
            print(f"[SKIP] Không có transcript: {transcript_path.name}")
            continue
        if output_path.exists():
            print(f"[SKIP] Đã có output: {output_path.name}")
            continue

        tasks.append((audio_path, transcript_path, output_path))

    if not tasks:
        print("Không có file nào cần xử lý.")
        return

    print(f"Tổng cộng {len(tasks)} file cần xử lý với {max_workers} workers...")
    completed = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, t): t for t in tasks}
        for future in as_completed(futures):
            name, success, err = future.result()
            if success:
                completed += 1
                print(f"  [{completed}/{len(tasks)}] ✓ {name}")
            else:
                failed += 1
                print(f"  [FAIL] {name}: {err}")

    print(f"\nHoàn thành: {completed} file thành công, {failed} file thất bại.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch FA runner với multiprocessing")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Thư mục dataset")
    parser.add_argument("--workers", type=int, default=4, help="Số workers song song")
    args = parser.parse_args()
    main(dataset_dir=Path(args.dataset_dir), max_workers=args.workers)
