#!/usr/bin/env python3
"""
Train mo hinh voi du lieu thuc tu camera
Luu model ra file .pkl
"""

import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from collections import Counter

def load_real_data():
    """Load du lieu thuc"""
    processed_dir = Path("real_data/processed")

    if not processed_dir.exists() or len(list(processed_dir.glob("*.npy"))) == 0:
        print("LOI: Chua co du lieu thuc!")
        print("Vui long chay: python3 capture_library_signs.py")
        return None, None, None

    X = []
    y = []
    filenames = []

    print("\nDang load du lieu...")
    for filepath in sorted(processed_dir.glob("*.npy")):
        # Parse filename: person_sign_000_00_timestamp.npy
        parts = filepath.stem.split('_')

        # Tim vi tri cua "sign"
        try:
            sign_idx = parts.index('sign')
            sign_id = int(parts[sign_idx + 1])
        except:
            print(f"CANH BAO: Bo qua file {filepath.name} (sai dinh dang)")
            continue

        # Load landmarks
        try:
            landmarks = np.load(filepath)
            X.append(landmarks)
            y.append(sign_id)
            filenames.append(filepath.name)
        except Exception as e:
            print(f"CANH BAO: Khong doc duoc {filepath.name}: {e}")
            continue

    if len(X) == 0:
        print("LOI: Khong co du lieu hop le!")
        return None, None, None

    X = np.array(X)
    y = np.array(y)

    print(f"\n>>> Da load {len(X)} mau tu du lieu thuc")
    print(f"    Shape: {X.shape}")
    print(f"    So ky hieu: {len(np.unique(y))}")

    # Thong ke phan bo
    sign_counts = Counter(y)
    print(f"\n>>> Phan bo du lieu:")
    for sign_id in sorted(sign_counts.keys()):
        count = sign_counts[sign_id]
        print(f"    Ky hieu {sign_id:2d}: {count:3d} mau")

    return X, y, filenames

def train_model(X, y):
    """Train Random Forest"""

    print(f"\n{'='*60}")
    print("TRAIN RANDOM FOREST")
    print(f"{'='*60}")

    # Flatten du lieu 3D -> 2D
    X_flat = X.reshape(X.shape[0], -1)

    # Chia train/test
    # Stratify de dam bao moi ky hieu deu co trong train va test
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_flat, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # Neu mot so ky hieu chi co 1 mau -> khong stratify
        print("CANH BAO: Mot so ky hieu co qua it mau, khong the stratify")
        X_train, X_test, y_train, y_test = train_test_split(
            X_flat, y, test_size=0.2, random_state=42
        )

    print(f"\n>>> Phan chia:")
    print(f"    Train: {len(X_train)} mau")
    print(f"    Test:  {len(X_test)} mau")

    # Train
    print(f"\n>>> Bat dau train...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    import time
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"\n>>> Hoan thanh train trong {train_time:.2f}s")

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"\n>>> Ket qua:")
    print(f"    Train accuracy: {train_acc*100:.1f}%")
    print(f"    Test accuracy:  {test_acc*100:.1f}%")

    # Classification report
    print(f"\n{'='*60}")
    print("BAO CAO CHI TIET")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred_test, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'So luong'})
    plt.title('MA TRAN NHAM LAN - DU LIEU THUC', fontsize=14, fontweight='bold')
    plt.xlabel('Du doan', fontsize=12)
    plt.ylabel('Thuc te', fontsize=12)
    plt.tight_layout()

    output_dir = Path('outputs_real')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'confusion_matrix_real.png', dpi=150, bbox_inches='tight')
    print(f"\n>>> Da luu: {output_dir / 'confusion_matrix_real.png'}")
    plt.close()

    return model, test_acc, train_acc, train_time

def save_model(model, accuracy, train_acc, train_time, X_shape, num_signs):
    """Luu model va metadata"""

    output_dir = Path('outputs_real')
    output_dir.mkdir(exist_ok=True)

    # Luu model
    model_path = output_dir / 'model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"\n>>> Da luu model: {model_path}")
    print(f"    Kich thuoc: {model_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Luu metadata
    metadata = {
        'test_accuracy': float(accuracy),
        'train_accuracy': float(train_acc),
        'train_time': float(train_time),
        'num_samples': int(X_shape[0]),
        'num_signs': int(num_signs),
        'input_shape': list(X_shape[1:]),
        'model_type': 'RandomForestClassifier',
        'timestamp': str(np.datetime64('now'))
    }

    metadata_path = output_dir / 'model_info.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f">>> Da luu metadata: {metadata_path}")

def main():
    print("\n" + "="*60)
    print("TRAIN MO HINH VOI DU LIEU THUC")
    print("="*60)

    # Load du lieu
    X, y, filenames = load_real_data()

    if X is None:
        return

    # Kiem tra du lieu co du khong
    unique_signs = np.unique(y)
    if len(unique_signs) < 2:
        print("\nLOI: Can it nhat 2 ky hieu khac nhau de train!")
        print(f"Hien tai chi co: {len(unique_signs)} ky hieu")
        return

    # Train
    model, test_acc, train_acc, train_time = train_model(X, y)

    # Luu model
    save_model(model, test_acc, train_acc, train_time, X.shape, len(unique_signs))

    print("\n" + "="*60)
    print("HOAN THANH!")
    print("="*60)
    print(f"\n>>> Do chinh xac: {test_acc*100:.1f}%")
    print(f">>> Ket qua luu trong: outputs_real/")
    print(f"\nTIEP THEO:")
    print(f"  1. Xem ket qua: ls -lh outputs_real/")
    print(f"  2. Test thoi gian thuc: python3 test_realtime.py")

    # Goi y cai thien
    if test_acc < 0.7:
        print(f"\nGOI Y CAI THIEN (Accuracy < 70%):")
        print(f"  - Thu them du lieu (hien tai: {X.shape[0]} mau)")
        print(f"  - Moi ky hieu nen co it nhat 20-30 mau")
        print(f"  - Thu tu nhieu nguoi khac nhau")
    elif test_acc < 0.85:
        print(f"\nGOI Y CAI THIEN (Accuracy < 85%):")
        print(f"  - Bo sung them 10-20 mau cho moi ky hieu")
        print(f"  - Thu trong dieu kien anh sang khac nhau")

if __name__ == "__main__":
    main()