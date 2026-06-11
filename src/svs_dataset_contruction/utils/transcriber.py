"""
Module lyrics transcription.

Sử dụng ChunkFormer RNNT model để transcribe audio → text (lyrics).
"""

from pathlib import Path

from chunkformer import ChunkFormerModel

from ..config import TranscriberConfig


class Transcriber:
    """
    Transcribe audio thành lyrics dùng ChunkFormer RNNT.

    Model được lazy-load lần đầu tiên khi gọi transcribe().
    Mỗi instance quản lý model của riêng mình — an toàn với multiprocessing.

    Example:
        transcriber = Transcriber()
        text = transcriber.transcribe(Path("vocals/song.wav"))

        # Tuỳ chỉnh config
        cfg = TranscriberConfig(model_id="khanhld/chunkformer-rnnt-large-vie", chunk_size=32)
        transcriber = Transcriber(config=cfg)
    """

    def __init__(self, config: TranscriberConfig = TranscriberConfig()):
        self.config = config
        self._model: ChunkFormerModel | None = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self) -> ChunkFormerModel:
        """Lazy-load model vào bộ nhớ (chỉ load một lần)."""
        if self._model is None:
            print(f"Loading ChunkFormer RNNT model: {self.config.model_id} on {getattr(self.config, 'device', 'cpu')}...")
            self._model = ChunkFormerModel.from_pretrained(self.config.model_id)
            if hasattr(self.config, 'device') and self.config.device != "cpu":
                self._model.to(self.config.device)
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: Path) -> str:
        """
        Transcribe một file audio thành text (lyrics).

        Args:
            audio_path: Đường dẫn file audio.

        Returns:
            Text transcription (chuỗi đã ghép các chunks).
        """
        model = self._load_model()
        result = model.endless_decode(
            audio_path=str(audio_path),
            chunk_size=self.config.chunk_size,
            left_context_size=self.config.left_context_size,
            right_context_size=self.config.right_context_size,
            total_batch_duration=self.config.total_batch_duration,
            return_timestamps=True,
        )
        # Ghép các chunks thành 1 đoạn text
        chunks = [item["decode"] for item in result]
        return " ".join(chunks)

    def transcribe_to_file(self, audio_path: Path, output_path: Path) -> str:
        """
        Transcribe audio và lưu kết quả vào file .txt.

        Nếu output_path đã tồn tại, đọc lại và trả về nội dung cũ (skip).

        Args:
            audio_path:  Đường dẫn file audio.
            output_path: Đường dẫn file .txt sẽ được tạo.

        Returns:
            Text transcription.
        """
        if output_path.exists():
            print(f"  -> Đã có transcript: {output_path}")
            return output_path.read_text(encoding="utf-8").strip()

        transcript = self.transcribe(audio_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(transcript, encoding="utf-8")
        print(f"  -> Đã lưu transcript: {output_path}")
        return transcript

    def transcribe_batch(self, audio_paths: list[Path]) -> list[str]:
        """
        Transcribe nhiều file audio cùng lúc.

        Args:
            audio_paths: List các file audio.

        Returns:
            List các text transcription theo đúng thứ tự đầu vào.
        """
        model = self._load_model()
        results = model.batch_decode(
            audio_paths=[str(p) for p in audio_paths],
            chunk_size=self.config.chunk_size,
            left_context_size=self.config.left_context_size,
            right_context_size=self.config.right_context_size,
            total_batch_duration=self.config.batch_total_duration,
        )
        return results
