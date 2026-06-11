import argparse
from pathlib import Path
import subprocess
import sys
from svs_dataset_contruction.config import DatasetPaths
from svs_dataset_contruction.aligners.midi_phoneme import MidiPhonemeAligner

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Tạo aligned_ds cho nhiều phương pháp trích xuất MIDI")
    parser.add_argument("--methods", type=str, default="rmvpe,game,basic-pitch", help="Danh sách phương pháp (phẩy giữa)")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Thư mục dataset gốc")
    parser.add_argument("--workers", type=int, default=1, help="Số workers cho MIDI extraction")
    parser.add_argument("--overwrite", action="store_true", help="Ghi đè file MIDI cũ")
    
    args = parser.parse_args()
    methods = [m.strip() for m in args.methods.split(",")]
    paths = DatasetPaths(root=Path(args.dataset_dir))
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Đang xử lý phương pháp: {method.upper()}")
        print(f"{'='*60}")
        
        # 1. Trích xuất MIDI (Batch)
        # Gọi trực tiếp module để đảm bảo context đúng
        cmd = [
            "pixi", "run", "python", "-m", "svs_dataset_contruction.extract_midi_batch",
            "--method", method,
            "--workers", str(args.workers)
        ]
        if args.overwrite:
            cmd.append("--overwrite")
            
        if not run_command(cmd):
            print(f"Bỏ qua bước align cho {method} do lỗi trích xuất MIDI.")
            continue
            
        # 2. Đồng bộ Phonemes & MIDI (MidiPhonemeAligner)
        print(f"\nĐang tạo aligned_ds cho {method}...")
        try:
            aligner = MidiPhonemeAligner(paths=paths, method=method)
            aligner.process_dataset()
        except Exception as e:
            print(f"Lỗi khi đồng bộ cho {method}: {e}")

    print("\nToàn bộ quá trình hoàn tất!")

if __name__ == "__main__":
    main()
