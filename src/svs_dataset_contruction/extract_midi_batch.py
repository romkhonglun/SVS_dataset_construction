import argparse
import json
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from functools import partial
from .note_extractor import NoteExtractor
from .config import DatasetPaths

def process_song(subdir, method: str, overwrite: bool):
    """Xử lý trích xuất MIDI cho một thư mục bài hát (các segments)."""
    paths = DatasetPaths()
    # Khởi tạo extractor bên trong worker để tránh vấn đề share resource
    extractor = NoteExtractor(method=method)
    audio_files = sorted(subdir.glob("*.wav"))
    
    for audio_path in audio_files:
        song_id = subdir.name
        seg_name = audio_path.stem
        output_name = f"{song_id}_{seg_name}.midi.json"
        output_path = paths.final_midis / output_name
        
        if output_path.exists() and not overwrite:
            continue
            
        try:
            notes = extractor.extract_midi_notes(audio_path)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(notes, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  -> [ERROR] Failed to extract {audio_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Batch MIDI extraction")
    parser.add_argument("--method", type=str, default="game", choices=["game", "basic-pitch"], help="MIDI extraction method")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing MIDI files")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers for multiprocessing")
    args = parser.parse_args()

    paths = DatasetPaths()
    paths.mkdir()
    
    corpus_dir = paths.mfa_corpus
    subdirs = sorted([d for d in corpus_dir.iterdir() if d.is_dir()])
    
    if not subdirs:
        corpus_dir = Path("dataset/mfa_corpus")
        subdirs = sorted([d for d in corpus_dir.iterdir() if d.is_dir()])
        if not subdirs:
            print(f"Không tìm thấy thư mục con nào trong {corpus_dir}")
            return

    print(f"Sử dụng phương pháp: {args.method}")
    print(f"Ghi đè file cũ: {args.overwrite}")
    print(f"Số workers: {args.workers}")
    print(f"Tìm thấy {len(subdirs)} bài hát. Bắt đầu xử lý...")

    if args.workers > 1:
        worker_func = partial(process_song, method=args.method, overwrite=args.overwrite)
        with multiprocessing.Pool(args.workers) as pool:
            list(tqdm(pool.imap_unordered(worker_func, subdirs), total=len(subdirs), desc="Đang trích xuất MIDI"))
    else:
        for subdir in tqdm(subdirs, desc="Đang trích xuất MIDI"):
            process_song(subdir, args.method, args.overwrite)

    print("\nHoàn thành trích xuất MIDI cho toàn bộ dataset.")

if __name__ == "__main__":
    main()
