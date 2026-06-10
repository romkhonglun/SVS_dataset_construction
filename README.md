# SVS_dataset_contruction

Pipeline xây dựng dataset cho **Singing Voice Synthesis (SVS)** — hỗ trợ các model như DiffSinger, TCSinger, và SoulX-Singer.
Project được thiết kế tối ưu cho **tiếng Việt**, từ tách vocal, transcribe lời, forced alignment đến cắt segment.

---

## Tính năng đã hoàn thành

| Bước | Mô tả | Module |
|------|-------|--------|
| **1. Tách Vocal** | Tách vocal khỏi nhạc nền bằng MelBand RoFormer (`audio-separator`). Chỉ giữ lại vocal, xóa instrumental/drums. | `separator.py` |
| **2. Transcribe** | Chuyển audio tiếng Việt thành lyrics (lời bài hát) bằng **ChunkFormer RNNT** (`khanhld/chunkformer-rnnt-large-vie`). | `transcriber.py` |
| **3. Forced Alignment** | Căn chỉnh word-level giữa audio và lyrics bằng **Wav2Vec2 Việt Nam** (`nguyenvulebinh/wav2vec2-base-vietnamese-250h`), xuất file **TextGrid** với tier `words` và `confidence`. | `aligner.py` |
| **4. Trích xuất MIDI** | Trích xuất note/pitch từ audio vocal bằng model **GAME** (ONNX) hoặc **basic-pitch** (Spotify). | `note_extractor.py` |
| **5. Cắt Segment** | Cắt audio thành các đoạn segment (0.5s – 15s) dựa trên word timestamps từ FA. | `segmenter.py` |
| **6. Chuẩn bị Dataset** | Validate `metadata.csv`, tạo cấu trúc thư mục canonical cho dataset. | `preparator.py` |
| **7. Batch Runner** | Chạy toàn bộ pipeline hoặc từng phần cho cả dataset; hỗ trợ GPU và multiprocessing. | `run_full_pipeline.py`, `extract_midi_batch.py` |
| **8. CLI** | Giao diện dòng lệnh (`svs-pipeline`) để chạy từng bước hoặc full pipeline. | `pipeline.py` |

---

## Yêu cầu

- [pixi](https://pixi.sh) — công cụ quản lý môi trường và dependencies
- Python 3.11 hoặc 3.12

---

## Thiết lập môi trường

```bash
# Cài đặt tất cả dependencies
pixi install

# Kích hoạt shell trong môi trường pixi
pixi shell
```

---

## Cách sử dụng

### 1. Tách vocal từ audio đầu vào

```bash
pixi run python -m svs_dataset_contruction.separator input/
```

- Input: `input/audio/*.wav` (hoặc mp3, flac, …)
- Output: `input/vocals/*_vocals.wav`

### 2. Chạy full pipeline (Transcribe + FA)

```bash
pixi run python -m svs_dataset_contruction.run_full_pipeline
```

### 3. Trích xuất MIDI cho toàn bộ dataset (GPU)

```bash
# Sử dụng GAME (mặc định)
pixi run svs-extract-midi --method game

# Sử dụng Spotify basic-pitch
pixi run svs-extract-midi --method basic-pitch
```

- Sử dụng `onnxruntime-gpu` (cho GAME) hoặc `tensorflow` (cho basic-pitch) để tăng tốc độ xử lý.
- Tự động lưu file `.midi.json` vào `dataset/3_final/midis/`.

### 4. Chạy Forced Alignment batch

```bash
pixi run python -m svs_dataset_contruction.run_mfa_align dataset/mfa_corpus dataset/mfa_aligned
```

### 5. CLI từng bước (`svs-pipeline`)

```bash
# Transcribe một file
pixi run svs-pipeline transcribe dataset/vocals/song.wav

# Forced Alignment một file (tự động tìm transcript trong raw_lyric/)
pixi run svs-pipeline align dataset/vocals/song.wav

# Transcribe + Align một file
pixi run svs-pipeline run dataset/vocals/song.wav
```

### 5. Tổng hợp dataset cuối cùng (Finalize)

```bash
pixi run svs-pipeline finalize
```
- Gom toàn bộ dữ liệu (wav, TextGrid, midi) từ workspace và phẳng hoá vào thư mục `dataset/3_final/`.
- Tự động tạo `metadata.csv` tổng hợp cho huấn luyện.

---

## Cấu trúc dataset (Scientific Layout)

Dữ liệu được quản lý tự động thông qua `svs_dataset_contruction.config.DatasetPaths` theo từng giai đoạn xử lý:

```text
dataset/
├── 0_input/                # Dữ liệu nguyên thủy & metadata.csv khởi tạo
├── 1_interim/              # Kết quả sau các bước xử lý toàn bộ bài hát
│   ├── vocals/             # Tracks vocal (đã tách)
│   ├── lyrics/             # Transcript thô (.txt)
│   └── textgrids_full/     # Alignment thô bằng Wav2Vec2 (.TextGrid)
├── 2_mfa_workspace/        # Không gian làm việc cho MFA
│   ├── corpus/             # Segments thô (cắt từ TextGrid thô)
│   └── aligned/            # Segments đã align lại bằng MFA (.TextGrid)
└── 3_final/                # Dataset đầu ra (Sẵn sàng cho SVS models)
    ├── wavs/               # Segments âm thanh
    ├── textgrids/          # Segments alignment
    ├── midis/              # Segments cao độ/note (tuỳ chọn)
    ├── aligned_ds/         # Dữ liệu đã đồng bộ MIDI + Phonemes (chuẩn DiffSinger)
    └── metadata.csv        # File ánh xạ dùng để train (item_name, paths, etc)
```

---

## Cấu trúc project

```
SVS_dataset_contruction/
├── src/svs_dataset_contruction/    # Toàn bộ mã nguồn nằm ở đây
│   ├── __init__.py
│   ├── pipeline.py                 # CLI chính (run / batch / finalize)
│   ├── config.py                   # Quản lý đường dẫn (DatasetPaths) & settings
│   ├── transcriber.py              # Transcribe audio → lyrics
│   ├── aligner.py                  # Forced Alignment (Wav2Vec2)
│   ├── separator.py                # Tách vocal (MelBand RoFormer)
│   ├── segmenter.py                # Cắt audio thành segment
│   ├── game_note_extractor.py      # Trích xuất MIDI (GAME model ONNX)
│   ├── note_extractor.py           # NoteExtractor wrapper
│   ├── extract_midi_batch.py       # Batch trích xuất MIDI (GPU)
│   ├── run_full_pipeline.py        # Batch transcribe + FA
│   └── run_mfa_align.py            # Batch MFA alignment
├── pyproject.toml                  # Pixi workspace & dependencies
├── GEMINI.md                       # Định hướng môi trường & AI
├── README.md                       # File này
└── dataset/                        # Thư mục dữ liệu (như cấu trúc trên)
```

---

## Các lệnh Pixi thường dùng

| Lệnh | Mô tả |
|------|-------|
| `pixi install` | Cài đặt các dependencies từ `pyproject.toml` |
| `pixi shell` | Kích hoạt môi trường ảo của pixi |
| `pixi run <cmd>` | Chạy một command trong môi trường ảo |
| `pixi add <package>` | Thêm package mới vào dependencies |
| `pixi task add <name> <cmd>` | Định nghĩa một task mới |
| `pixi run <name>` | Chạy task đã định nghĩa |

---

## Thông tin kỹ thuật

- **Build system**: [hatchling](https://hatch.pypa.io/)
- **Channels**: conda-forge
- **Platform**: linux-64
- **Python**: 3.11 hoặc 3.12

## License

Apache 2.0
