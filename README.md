# SVS_dataset_contruction

Pipeline xây dựng dataset cho **Singing Voice Synthesis (SVS)** — hỗ trợ các model như DiffSinger, TCSinger, và SoulX-Singer.
Project được thiết kế tối ưu cho **tiếng Việt**, từ tách vocal, transcribe lời, forced alignment đến đồng bộ hóa MIDI.

---

## Tính năng nổi bật

| Bước | Mô tả | Công nghệ |
|------|-------|-----------|
| **1. Tách Vocal** | Tách vocal khỏi nhạc nền, chỉ giữ lại giọng hát sạch. | MelBand RoFormer (`audio-separator`) |
| **2. Transcribe** | Tự động nhận diện lời bài hát tiếng Việt từ audio. | **ChunkFormer RNNT** |
| **3. Forced Alignment** | Căn chỉnh thời gian cấp độ từ (word-level) và âm vị (phoneme). | Wav2Vec2 VN & **Montreal Forced Aligner (MFA)** |
| **4. Trích xuất Pitch/MIDI** | Trích xuất cao độ với độ chính xác cực cao, hỗ trợ nhiều phương pháp. | **RMVPE (chuẩn DiffSinger)**, GAME, Basic-Pitch |
| **5. Đồng bộ Dataset** | Tạo dữ liệu `.ds` đồng bộ hoàn hảo giữa Phonemes và MIDI. | **MidiPhonemeAligner** |
| **6. Batch Processing** | Xử lý hàng nghìn bài hát cùng lúc, hỗ trợ GPU và Multiprocessing. | `spawn` mode (GPU-safe) |

---

## Yêu cầu

- [pixi](https://pixi.sh) — công cụ quản lý môi trường và dependencies
- Python 3.11+
- CUDA (khuyên dùng để tăng tốc trích xuất Pitch)

---

## Thiết lập môi trường

```bash
# Cài đặt tất cả dependencies
pixi install

# Kích hoạt môi trường
pixi shell
```

---

## Cách sử dụng

### 1. Trích xuất MIDI & Tạo Dataset đồng bộ (Khuyên dùng)

Bạn có thể tạo các bộ dữ liệu `aligned_ds` riêng biệt cho từng phương pháp trích xuất để so sánh:

```bash
# Tạo dữ liệu cho RMVPE và GAME (chạy đa luồng 4 workers)
pixi run svs-batch-aligned --methods rmvpe,game --workers 4
```

- `--methods`: Hỗ trợ `rmvpe`, `game`, `basic-pitch`, `rosvot`.
- Kết quả sẽ được lưu vào `dataset/3_final/aligned_ds_{method}/`.

### 2. So sánh trực quan Pitch

Để kiểm tra xem Pitch trích xuất có khớp với audio hay không:

```bash
pixi run python -m svs_dataset_contruction.scripts.compare_pitch_rmvpe path/to/audio.wav
```

### 3. Chạy Pipeline từng bước (`svs-pipeline`)

```bash
# Transcribe + Align + Segment cho 1 file
pixi run svs-pipeline run dataset/0_input/song.wav

# Chạy batch cho cả thư mục input
pixi run svs-pipeline batch --workers 4

# Tổng hợp dữ liệu cuối cùng
pixi run svs-pipeline finalize
```

---

## Cấu trúc Dataset (Phát triển bởi Pipeline)

```text
dataset/
├── 0_input/                # Audio gốc đầu vào
├── 1_interim/              # Kết quả trung gian (vocals, lyrics, full textgrids)
├── 2_mfa_workspace/        # Không gian làm việc cho Montreal Forced Aligner
└── 3_final/                # Dữ liệu đích cho việc huấn luyện
    ├── wavs/               # Audio segments (0.5s - 15s)
    ├── textgrids/          # Alignment segments
    ├── midis_rmvpe/        # MIDI trích xuất bằng RMVPE
    ├── aligned_ds_rmvpe/   # Dữ liệu .ds (Phoneme + MIDI) dùng train DiffSinger
    └── metadata.csv        # Metadata tổng hợp
```

---

## Cấu trúc Project (Refactored)

```text
src/svs_dataset_contruction/
├── extractors/      # Logic trích xuất Pitch (RMVPE, GAME, BasicPitch, Rosvot)
├── aligners/        # Logic đồng bộ hóa (MFA, MIDI-Phoneme)
├── utils/           # Tiện ích (Transcriber, Segmenter, Separator, Visualization)
├── scripts/         # Các script chạy Batch và CLI tools
├── config.py        # Quản lý đường dẫn và cấu hình tập trung
└── pipeline.py      # Logic pipeline chính
```

---

## Thông tin kỹ thuật

- **MFA Model**: Vietnamese (MFA)
- **RMVPE**: Sử dụng Mel HTK và Weighted Average Decoding (độ chính xác cấp độ cent).
- **Multiprocessing**: Sử dụng `spawn` method để đảm bảo an toàn khi khởi tạo CUDA trong subprocess.

## License

Apache 2.0
