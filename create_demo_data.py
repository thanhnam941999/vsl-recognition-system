#!/usr/bin/env python3
"""
Tạo dữ liệu demo ngôn ngữ ký hiệu Việt Nam
Mô phỏng 30 ký hiệu từ 10 người, mỗi ký hiệu 20 lần
"""

import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm

# Cấu hình
NUM_PEOPLE = 10
NUM_SIGNS = 30
SAMPLES_PER_SIGN = 20
NUM_FRAMES = 30  # 30 frames mỗi video (1 giây @ 30fps)
NUM_LANDMARKS = 21  # 21 điểm mốc bàn tay
NUM_COORDS = 3  # x, y, z

def create_synthetic_sign_data(sign_id, person_id, variation=0.1):
    """
    Tạo dữ liệu ký hiệu giả
    Mỗi ký hiệu có pattern riêng dựa trên sign_id
    """
    # Base pattern cho mỗi ký hiệu
    np.random.seed(sign_id * 1000 + person_id)

    # Tạo chuyển động cơ bản cho ký hiệu
    # Pattern khác nhau cho mỗi sign_id
    base_movement = np.zeros((NUM_FRAMES, NUM_LANDMARKS, NUM_COORDS))

    for frame_idx in range(NUM_FRAMES):
        # Tạo hình dạng bàn tay đặc trưng cho mỗi ký hiệu
        t = frame_idx / NUM_FRAMES

        for landmark_idx in range(NUM_LANDMARKS):
            # X coordinate - chuyển động ngang
            base_movement[frame_idx, landmark_idx, 0] = np.sin(2 * np.pi * t + sign_id * 0.1 + landmark_idx * 0.05)

            # Y coordinate - chuyển động dọc
            base_movement[frame_idx, landmark_idx, 1] = np.cos(2 * np.pi * t + sign_id * 0.15 + landmark_idx * 0.03)

            # Z coordinate - độ sâu
            base_movement[frame_idx, landmark_idx, 2] = np.sin(np.pi * t + sign_id * 0.2) * 0.3

    # Thêm biến thiên cá nhân
    noise = np.random.normal(0, variation, base_movement.shape)
    landmarks = base_movement + noise

    # Chuẩn hóa về [-1, 1]
    landmarks = landmarks / (np.max(np.abs(landmarks)) + 1e-8)

    return landmarks

def create_dataset():
    """Tạo toàn bộ dataset"""

    # Tạo thư mục
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("TẠO DỮ LIỆU DEMO NGÔN NGỮ KÝ HIỆU VIỆT NAM")
    print("="*60)
    print(f"Số người: {NUM_PEOPLE}")
    print(f"Số ký hiệu: {NUM_SIGNS}")
    print(f"Mẫu mỗi ký hiệu: {SAMPLES_PER_SIGN}")
    print(f"Tổng mẫu: {NUM_PEOPLE * NUM_SIGNS * SAMPLES_PER_SIGN}")
    print()

    # Tạo danh sách ký hiệu
    sign_names = []
    for i in range(NUM_SIGNS):
        if i < 10:
            sign_names.append(f"số_{i}")
        else:
            sign_names.append(f"ký_hiệu_{i}")

    # Lưu mapping
    sign_mapping = {i: name for i, name in enumerate(sign_names)}
    with open(data_dir / "sign_mapping.json", 'w', encoding='utf-8') as f:
        json.dump(sign_mapping, f, ensure_ascii=False, indent=2)

    # Tạo dữ liệu
    all_samples = []

    total_iterations = NUM_PEOPLE * NUM_SIGNS * SAMPLES_PER_SIGN

    with tqdm(total=total_iterations, desc="Tạo dữ liệu") as pbar:
        for person_id in range(NUM_PEOPLE):
            for sign_id in range(NUM_SIGNS):
                for sample_id in range(SAMPLES_PER_SIGN):
                    # Tạo dữ liệu landmarks
                    landmarks = create_synthetic_sign_data(sign_id, person_id)

                    # Tạo tên file
                    filename = f"person_{person_id:02d}_sign_{sign_id:03d}_sample_{sample_id:02d}.npy"
                    filepath = processed_dir / filename

                    # Lưu file
                    np.save(filepath, landmarks)

                    # Thêm vào danh sách
                    all_samples.append({
                        'filepath': str(filepath),
                        'person_id': person_id,
                        'sign_id': sign_id,
                        'sign_name': sign_names[sign_id],
                        'sample_id': sample_id,
                        'num_frames': NUM_FRAMES
                    })

                    pbar.update(1)

    # Lưu metadata
    import pandas as pd
    df = pd.DataFrame(all_samples)
    df.to_csv(data_dir / "metadata.csv", index=False)

    print()
    print("="*60)
    print("HOÀN THÀNH TẠO DỮ LIỆU")
    print("="*60)
    print(f"✅ Tổng số file: {len(all_samples)}")
    print(f"✅ Thư mục: {processed_dir}")
    print(f"✅ Metadata: {data_dir / 'metadata.csv'}")
    print(f"✅ Sign mapping: {data_dir / 'sign_mapping.json'}")
    print()

    # Thống kê
    print("THỐNG KÊ:")
    print(f"  - Số người: {NUM_PEOPLE}")
    print(f"  - Số ký hiệu: {NUM_SIGNS}")
    print(f"  - Mẫu/ký hiệu/người: {SAMPLES_PER_SIGN}")
    print(f"  - Tổng mẫu: {len(all_samples)}")
    print(f"  - Kích thước mỗi mẫu: {NUM_FRAMES}×{NUM_LANDMARKS}×{NUM_COORDS}")
    print()

if __name__ == "__main__":
    create_dataset()