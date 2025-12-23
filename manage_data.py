#!/usr/bin/env python3
"""
Quản lý và thống kê dữ liệu
"""

from pathlib import Path
import numpy as np
from collections import Counter

def analyze_data():
    """Phân tích dữ liệu hiện có"""

    processed_dir = Path("real_data/processed")

    if not processed_dir.exists():
        print("❌ Chưa có dữ liệu!")
        return

    files = list(processed_dir.glob("*.npy"))

    if len(files) == 0:
        print("❌ Chưa có file dữ liệu!")
        return

    print("\n" + "="*60)
    print("THỐNG KÊ DỮ LIỆU")
    print("="*60)

    # Thống kê theo người
    people = set()
    signs = []

    for filepath in files:
        parts = filepath.stem.split('_')

        # Lấy tên người (phần đầu trước "sign")
        person = parts[0]
        people.add(person)

        # Lấy sign_id
        try:
            sign_idx = parts.index('sign')
            sign_id = int(parts[sign_idx + 1])
            signs.append(sign_id)
        except:
            continue

    print(f"\n📊 TỔNG QUAN:")
    print(f"   Tổng số file: {len(files)}")
    print(f"   Số người tham gia: {len(people)}")
    print(f"   Danh sách người: {', '.join(sorted(people))}")

    # Thống kê theo ký hiệu
    sign_counts = Counter(signs)

    print(f"\n📊 PHÂN BỐ THEO KÝ HIỆU:")
    for sign_id in sorted(sign_counts.keys()):
        count = sign_counts[sign_id]
        bar = "█" * (count // 5)
        print(f"   Ký hiệu {sign_id:2d}: {count:3d} mẫu {bar}")

    # Thống kê theo người
    print(f"\n📊 PHÂN BỐ THEO NGƯỜI:")
    for person in sorted(people):
        person_files = [f for f in files if f.stem.startswith(person)]
        print(f"   {person:15s}: {len(person_files):3d} mẫu")

    # Kiểm tra cân bằng
    min_samples = min(sign_counts.values())
    max_samples = max(sign_counts.values())

    print(f"\n⚖️  CÂN BẰNG DỮ LIỆU:")
    if max_samples - min_samples > 10:
        print(f"   ⚠️  Không cân bằng! Chênh lệch: {max_samples - min_samples} mẫu")
        print(f"   💡 Nên bổ sung thêm mẫu cho các ký hiệu ít")
    else:
        print(f"   ✅ Cân bằng tốt! Chênh lệch: {max_samples - min_samples} mẫu")

    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    analyze_data()