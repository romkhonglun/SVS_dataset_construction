"""
Module tách vocal khỏi audio đầu vào sử dụng audio-separator.

Sử dụng model mặc định vocals_mel_band_roformer để tách vocal.
Chỉ giữ lại file vocal, xóa các file khác (instrumental, drums, ...).
"""

import argparse
import shutil
from pathlib import Path

from audio_separator.separator import Separator

from ..config import SeparatorConfig


class VocalSeparator:
    """
    Tách vocal khỏi audio dùng MelBand RoFormer (audio-separator).

    Separator được khởi tạo lazy khi gọi separate() lần đầu.

    Example:
        sep = VocalSeparator()
        vocal_files = sep.separate_dir(input_dir=Path("input"), output_dir=Path("dataset/vocals"))

        # Tách một file đơn lẻ
        vocal_path = sep.separate(audio_file=Path("song.mp3"), output_dir=Path("dataset/vocals"))
    """

    def __init__(self, config: SeparatorConfig = SeparatorConfig()):
        self.config = config
        self._separator: Separator | None = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_separator(self, output_dir: Path) -> Separator:
        """Khởi tạo (hoặc tái sử dụng) Separator với output_dir chỉ định."""
        # Nếu output_dir đổi hoặc chưa có, tạo mới
        if self._separator is None or str(self._separator.output_dir) != str(output_dir):
            self._separator = Separator(
                output_dir=str(output_dir),
                output_format=self.config.output_format,
                model_file=self.config.model_filename,
            )
            self._separator.load_model()
        return self._separator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def separate(self, audio_file: Path, output_dir: Path) -> Path | None:
        """
        Tách vocal từ một file audio đơn lẻ.

        File vocal được đổi tên thành ``<stem>_vocals.<ext>`` và lưu vào output_dir.
        Các file phụ (instrumental, drums, ...) bị xóa.

        Args:
            audio_file: Đường dẫn file audio đầu vào.
            output_dir: Thư mục lưu file vocal output.

        Returns:
            Đường dẫn file vocal đã lưu, hoặc None nếu không tìm thấy.
        """
        if audio_file.suffix.lower() not in self.config.audio_extensions:
            print(f"  Bỏ qua (không phải audio): {audio_file.name}")
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        separator = self._get_separator(output_dir)

        print(f"Đang xử lý: {audio_file.name}")
        output_files = separator.separate(str(audio_file))

        if not output_files:
            print(f"  Không có file đầu ra cho: {audio_file.name}")
            return None

        # Tìm file vocal
        vocal_file: Path | None = None
        for out_file in output_files:
            out_path = Path(out_file)
            if "vocal" in out_path.stem.lower():
                vocal_file = out_path
                break

        # Fallback: lấy file đầu tiên nếu không tìm thấy
        if vocal_file is None:
            vocal_file = Path(output_files[0])

        # Lưu file vocal, xóa file phụ
        dest_path: Path | None = None
        for out_file in output_files:
            out_path = Path(out_file)
            if out_path == vocal_file:
                dest_name = f"{audio_file.stem}_vocals{out_path.suffix}"
                dest_path = output_dir / dest_name
                shutil.move(str(out_path), str(dest_path))
                print(f"  -> Vocal lưu tại: {dest_path}")
            else:
                if out_path.exists():
                    out_path.unlink()
                print(f"  -> Xóa file phụ: {out_path.name}")

        return dest_path

    def separate_dir(self, input_dir: Path, output_dir: Path) -> list[Path]:
        """
        Tách vocal cho tất cả file audio trong ``input_dir``.

        Args:
            input_dir:  Thư mục chứa file audio đầu vào.
            output_dir: Thư mục lưu các file vocal output.

        Returns:
            List các đường dẫn file vocal đã tạo.
        """
        if not input_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục: {input_dir}")

        vocal_files: list[Path] = []
        for audio_file in sorted(input_dir.iterdir()):
            if not audio_file.is_file():
                continue
            result = self.separate(audio_file, output_dir)
            if result is not None:
                vocal_files.append(result)

        return vocal_files


def main():
    parser = argparse.ArgumentParser(description="Tách vocal từ audio đầu vào.")
    parser.add_argument(
        "input_dir",
        default="input",
        nargs="?",
        help="Thư mục chứa folder 'audio' (mặc định: input)",
    )
    parser.add_argument(
        "--model",
        default="vocals_mel_band_roformer",
        help="Tên model sử dụng (mặc định: vocals_mel_band_roformer)",
    )
    parser.add_argument(
        "--output-dir",
        default="vocals",
        help="Tên thư mục output trong input_dir (mặc định: vocals)",
    )

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    audio_dir = input_path / "audio"
    output_dir = input_path / args.output_dir

    print(f"Bắt đầu tách vocal từ: {audio_dir}/")
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}/")
    print("-" * 50)

    config = SeparatorConfig(model_filename=args.model)
    sep = VocalSeparator(config=config)
    vocal_files = sep.separate_dir(input_dir=audio_dir, output_dir=output_dir)

    print("-" * 50)
    print(f"Hoàn thành! Đã tách {len(vocal_files)} file vocal.")
    for vf in vocal_files:
        print(f"  - {vf}")


if __name__ == "__main__":
    main()
