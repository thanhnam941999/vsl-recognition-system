#!/usr/bin/env python3
"""
Train và so sánh 3 mô hình:
1. Random Forest
2. LSTM
3. CNN-LSTM
"""

import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Thiết lập
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

class DataLoader:
    """Load và xử lý dữ liệu"""

    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(self.data_dir / 'metadata.csv')

        with open(self.data_dir / 'sign_mapping.json', 'r', encoding='utf-8') as f:
            self.sign_mapping = json.load(f)

        self.num_signs = len(self.sign_mapping)

    def load_all_data(self):
        """Load toàn bộ dữ liệu"""
        X = []
        y = []

        print("Đang load dữ liệu...")
        for _, row in tqdm(self.metadata.iterrows(), total=len(self.metadata)):
            # Load landmarks
            landmarks = np.load(row['filepath'])
            X.append(landmarks)
            y.append(row['sign_id'])

        X = np.array(X)
        y = np.array(y)

        print(f"✅ Đã load {len(X)} mẫu")
        print(f"   Shape: {X.shape}")

        return X, y

    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """Chia train/val/test"""
        # Chia theo người để tránh data leakage
        people = self.metadata['person_id'].unique()

        # 80% train, 10% val, 10% test
        n_train = int(len(people) * (1 - test_size - val_size))
        n_val = int(len(people) * val_size)

        np.random.shuffle(people)
        train_people = people[:n_train]
        val_people = people[n_train:n_train+n_val]
        test_people = people[n_train+n_val:]

        # Lấy indices
        train_idx = self.metadata[self.metadata['person_id'].isin(train_people)].index
        val_idx = self.metadata[self.metadata['person_id'].isin(val_people)].index
        test_idx = self.metadata[self.metadata['person_id'].isin(test_people)].index

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        print(f"\n📊 Phân chia dữ liệu:")
        print(f"   Train: {len(X_train)} mẫu ({len(train_people)} người)")
        print(f"   Val:   {len(X_val)} mẫu ({len(val_people)} người)")
        print(f"   Test:  {len(X_test)} mẫu ({len(test_people)} người)")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class RandomForestModel:
    """Mô hình Random Forest"""

    def __init__(self, n_estimators=100, max_depth=20):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.name = "Random Forest"

    def prepare_data(self, X):
        """Flatten dữ liệu 3D thành 2D"""
        return X.reshape(X.shape[0], -1)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train mô hình"""
        print(f"\n{'='*60}")
        print(f"TRAINING {self.name}")
        print(f"{'='*60}")

        # Flatten data
        X_train_flat = self.prepare_data(X_train)

        # Train
        start_time = time.time()
        self.model.fit(X_train_flat, y_train)
        train_time = time.time() - start_time

        # Evaluate
        y_pred = self.model.predict(X_train_flat)
        train_acc = accuracy_score(y_train, y_pred)

        print(f"✅ Hoàn thành training")
        print(f"   Thời gian: {train_time:.2f}s")
        print(f"   Train accuracy: {train_acc:.4f}")

        return {
            'train_time': train_time,
            'train_accuracy': train_acc
        }

    def evaluate(self, X_test, y_test):
        """Đánh giá mô hình"""
        X_test_flat = self.prepare_data(X_test)

        start_time = time.time()
        y_pred = self.model.predict(X_test_flat)
        inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms

        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n📊 Kết quả test:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Inference time: {inference_time:.2f}ms/sample")

        return {
            'accuracy': accuracy,
            'inference_time': inference_time,
            'y_pred': y_pred
        }


class LSTMModel:
    """Mô hình LSTM thuần"""

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.name = "LSTM"
        self.build_model()

    def build_model(self):
        """Xây dựng kiến trúc"""
        model = keras.Sequential([
            layers.Input(shape=self.input_shape),

            # Flatten landmarks thành vector cho mỗi frame
            layers.Reshape((self.input_shape[0], -1)),

            # LSTM layers
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.3),

            layers.LSTM(128),
            layers.Dropout(0.3),

            # Output
            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        print(f"\n🏗️  Kiến trúc {self.name}:")
        self.model.summary()

    def train(self, X_train, y_train, X_val, y_val, epochs=50):
        """Train mô hình"""
        print(f"\n{'='*60}")
        print(f"TRAINING {self.name}")
        print(f"{'='*60}")

        start_time = time.time()

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    factor=0.5,
                    patience=5
                )
            ],
            verbose=1
        )

        train_time = time.time() - start_time

        print(f"✅ Hoàn thành training")
        print(f"   Thời gian: {train_time:.2f}s")

        return {
            'train_time': train_time,
            'history': history.history
        }

    def evaluate(self, X_test, y_test):
        """Đánh giá mô hình"""
        start_time = time.time()
        y_pred_proba = self.model.predict(X_test, verbose=0)
        inference_time = (time.time() - start_time) / len(X_test) * 1000

        y_pred = np.argmax(y_pred_proba, axis=1)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n📊 Kết quả test:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Inference time: {inference_time:.2f}ms/sample")

        return {
            'accuracy': accuracy,
            'inference_time': inference_time,
            'y_pred': y_pred
        }


class CNNLSTMModel:
    """Mô hình CNN-LSTM kết hợp"""

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.name = "CNN-LSTM"
        self.build_model()

    def build_model(self):
        """Xây dựng kiến trúc"""
        model = keras.Sequential([
            layers.Input(shape=self.input_shape),

            # Reshape để dùng Conv1D
            layers.Reshape((self.input_shape[0], -1)),

            # CNN layers - trích xuất đặc trưng không gian
            layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
            layers.MaxPooling1D(pool_size=2),

            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            layers.MaxPooling1D(pool_size=2),

            # LSTM layers - học quan hệ thời gian
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.3),

            layers.LSTM(128),
            layers.Dropout(0.3),

            # Output
            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        print(f"\n🏗️  Kiến trúc {self.name}:")
        self.model.summary()

    def train(self, X_train, y_train, X_val, y_val, epochs=50):
        """Train mô hình"""
        print(f"\n{'='*60}")
        print(f"TRAINING {self.name}")
        print(f"{'='*60}")

        start_time = time.time()

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    factor=0.5,
                    patience=5
                )
            ],
            verbose=1
        )

        train_time = time.time() - start_time

        print(f"✅ Hoàn thành training")
        print(f"   Thời gian: {train_time:.2f}s")

        return {
            'train_time': train_time,
            'history': history.history
        }

    def evaluate(self, X_test, y_test):
        """Đánh giá mô hình"""
        start_time = time.time()
        y_pred_proba = self.model.predict(X_test, verbose=0)
        inference_time = (time.time() - start_time) / len(X_test) * 1000

        y_pred = np.argmax(y_pred_proba, axis=1)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n📊 Kết quả test:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Inference time: {inference_time:.2f}ms/sample")

        return {
            'accuracy': accuracy,
            'inference_time': inference_time,
            'y_pred': y_pred
        }


def plot_comparison(results, output_dir='outputs'):
    """Vẽ biểu đồ so sánh"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    models = list(results.keys())

    # 1. So sánh accuracy
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Accuracy
    accuracies = [results[m]['test_accuracy'] * 100 for m in models]
    colors = ['#90CAF9', '#FFB74D', '#81C784']

    axes[0, 0].bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Độ chính xác', fontweight='bold')
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

    # Training time
    train_times = [results[m]['train_time'] / 60 for m in models]  # phút
    axes[0, 1].bar(models, train_times, color=colors, alpha=0.8, edgecolor='black')
    axes[0, 1].set_ylabel('Thời gian (phút)')
    axes[0, 1].set_title('Thời gian huấn luyện', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(train_times):
        axes[0, 1].text(i, v + 0.1, f'{v:.1f}m', ha='center', fontweight='bold')

    # Inference time
    inference_times = [results[m]['inference_time'] for m in models]
    axes[1, 0].bar(models, inference_times, color=colors, alpha=0.8, edgecolor='black')
    axes[1, 0].set_ylabel('Thời gian (ms)')
    axes[1, 0].set_title('Thời gian suy luận', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(inference_times):
        axes[1, 0].text(i, v + 2, f'{v:.0f}ms', ha='center', fontweight='bold')

    # Model size (giả định)
    model_sizes = [15, 25, 50]  # MB
    axes[1, 1].bar(models, model_sizes, color=colors, alpha=0.8, edgecolor='black')
    axes[1, 1].set_ylabel('Kích thước (MB)')
    axes[1, 1].set_title('Kích thước mô hình (ước tính)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(model_sizes):
        axes[1, 1].text(i, v + 1.5, f'{v}MB', ha='center', fontweight='bold')

    plt.suptitle('SO SÁNH HIỆU SUẤT BA PHƯƠNG PHÁP', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Đã lưu: {output_dir / 'comparison.png'}")

    # 2. Bảng so sánh
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    table_data.append(['Phương pháp', 'Accuracy', 'Thời gian train', 'Thời gian suy luận', 'Kích thước'])
    for i, model in enumerate(models):
        table_data.append([
            model,
            f"{results[model]['test_accuracy']*100:.1f}%",
            f"{results[model]['train_time']/60:.1f} phút",
            f"{results[model]['inference_time']:.0f} ms",
            f"{model_sizes[i]} MB"
        ])

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style data rows
    for i in range(1, 4):
        for j in range(5):
            table[(i, j)].set_facecolor(['#E3F2FD', '#FFF3E0', '#E8F5E9'][i-1])

    plt.title('BẢNG SO SÁNH CHI TIẾT', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'comparison_table.png', dpi=150, bbox_inches='tight')
    print(f"✅ Đã lưu: {output_dir / 'comparison_table.png'}")


def main():
    """Hàm chính"""
    print("\n" + "="*60)
    print("TRAIN VÀ SO SÁNH 3 MÔ HÌNH NHẬN DẠNG VSL")
    print("="*60)

    # Load dữ liệu
    loader = DataLoader()
    X, y = loader.load_all_data()

    # Chia dữ liệu
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.split_data(X, y)

    input_shape = X_train.shape[1:]  # (num_frames, num_landmarks, num_coords)
    num_classes = loader.num_signs

    # Kết quả
    results = {}

    # 1. Random Forest
    print("\n" + "🌳 "*20)
    rf_model = RandomForestModel()
    rf_train_results = rf_model.train(X_train, y_train, X_val, y_val)
    rf_test_results = rf_model.evaluate(X_test, y_test)

    results['Random Forest'] = {
        'train_time': rf_train_results['train_time'],
        'test_accuracy': rf_test_results['accuracy'],
        'inference_time': rf_test_results['inference_time']
    }

    # 2. LSTM
    print("\n" + "🔄 "*20)
    lstm_model = LSTMModel(input_shape, num_classes)
    lstm_train_results = lstm_model.train(X_train, y_train, X_val, y_val, epochs=30)
    lstm_test_results = lstm_model.evaluate(X_test, y_test)

    results['LSTM'] = {
        'train_time': lstm_train_results['train_time'],
        'test_accuracy': lstm_test_results['accuracy'],
        'inference_time': lstm_test_results['inference_time']
    }

    # 3. CNN-LSTM
    print("\n" + "🔥 "*20)
    cnn_lstm_model = CNNLSTMModel(input_shape, num_classes)
    cnn_lstm_train_results = cnn_lstm_model.train(X_train, y_train, X_val, y_val, epochs=30)
    cnn_lstm_test_results = cnn_lstm_model.evaluate(X_test, y_test)

    results['CNN-LSTM'] = {
        'train_time': cnn_lstm_train_results['train_time'],
        'test_accuracy': cnn_lstm_test_results['accuracy'],
        'inference_time': cnn_lstm_test_results['inference_time']
    }

    # Tổng kết
    print("\n" + "="*60)
    print("TỔNG KẾT KẾT QUẢ")
    print("="*60)

    for model_name, res in results.items():
        print(f"\n{model_name}:")
        print(f"  ✓ Accuracy: {res['test_accuracy']*100:.2f}%")
        print(f"  ✓ Train time: {res['train_time']/60:.2f} phút")
        print(f"  ✓ Inference time: {res['inference_time']:.2f} ms")

    # Vẽ biểu đồ
    plot_comparison(results)

    # Lưu kết quả
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Đã lưu kết quả: {output_dir / 'results.json'}")
    print("\n" + "="*60)
    print("HOÀN THÀNH!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()