#!/bin/bash
echo "Bắt đầu trích xuất MIDI bằng basic-pitch..."
pixi run svs-extract-midi --method basic-pitch --overwrite --workers 6
echo "Trích xuất MIDI xong. Bắt đầu tổng hợp dataset..."
pixi run svs-pipeline finalize
echo "Hoàn thành toàn bộ tiến trình."
