import shutil
from pathlib import Path
from svs_dataset_contruction.config import DatasetPaths

def migrate():
    paths = DatasetPaths()
    paths.mkdir()
    
    mapping = {
        "dataset/vocals": paths.vocals,
        "dataset/raw_lyric": paths.lyrics,
        "dataset/textgrid": paths.textgrids_full,
        "dataset/mfa_corpus": paths.mfa_corpus,
        "dataset/mfa_aligned": paths.mfa_aligned,
    }
    
    print("Bắt đầu di chuyển dữ liệu sang cấu trúc mới...")
    
    for old_path_str, new_path in mapping.items():
        old_path = Path(old_path_str)
        if old_path.exists() and old_path.is_dir():
            print(f"  Di chuyển {old_path} -> {new_path}")
            # Di chuyển nội dung bên trong
            for item in old_path.iterdir():
                dest = new_path / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
                else:
                    print(f"    [SKIP] {dest} đã tồn tại")
            # Xóa thư mục cũ nếu trống
            try:
                old_path.rmdir()
            except OSError:
                print(f"    [INFO] Thư mục {old_path} không trống, không thể xóa.")
        else:
            print(f"  [SKIP] Không tìm thấy thư mục cũ: {old_path}")

    # Đặc biệt: Di chuyển các file .midi.json từ mfa_corpus sang final_midis (namespaced)
    print("\nTrích xuất và đổi tên file MIDI sang final/midis/...")
    paths.final_midis.mkdir(parents=True, exist_ok=True)
    
    for song_dir in paths.mfa_corpus.iterdir():
        if song_dir.is_dir():
            song_id = song_dir.name
            for midi_file in song_dir.glob("*.midi.json"):
                seg_name = midi_file.name.replace(".midi.json", "")
                target_name = f"{song_id}_{seg_name}.midi.json"
                target_path = paths.final_midis / target_name
                
                if not target_path.exists():
                    shutil.move(str(midi_file), str(target_path))
                else:
                    midi_file.unlink() # Xóa bản copy cũ nếu đã có ở final
    
    print("\nHoàn thành di chuyển dữ liệu.")

if __name__ == "__main__":
    migrate()
