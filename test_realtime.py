#!/usr/bin/env python3
"""
Test nhan dang ky hieu thoi gian thuc
Su dung model da train
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import time

class RealtimeRecognizer:
    def __init__(self):
        # Khoi tao MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Load model
        self.model = None
        self.load_model()

        # Buffer de thu thap frames
        self.frame_buffer = []
        self.buffer_size = 30

        # Mapping ky hieu (30 ky hieu)
        self.sign_names = {
            # So 0-9
            0: "So 0", 1: "So 1", 2: "So 2", 3: "So 3", 4: "So 4",
            5: "So 5", 6: "So 6", 7: "So 7", 8: "So 8", 9: "So 9",
            # Loai tai lieu
            10: "Sach", 11: "Bao", 12: "Tap chi", 13: "Giao trinh", 14: "Luan van",
            # Hanh dong
            15: "Tim", 16: "Tim kiem", 17: "Doc", 18: "Muon", 19: "Tra",
            # Thuoc tinh
            20: "Tac gia", 21: "Tieu de", 22: "Nam", 23: "Moi", 24: "Cu",
            # Chu de
            25: "Cong nghe", 26: "Khoa hoc", 27: "Van hoc", 28: "Lich su", 29: "Toan hoc"
        }

        # Lich su nhan dang
        self.prediction_history = []
        self.history_size = 5

    def load_model(self):
        """Load mo hinh da train"""
        model_path = Path("outputs_real/model.pkl")

        if model_path.exists():
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(">>> Da load model thanh cong!")
            print(f"    Mo hinh: {model_path}")
        else:
            print("CANH BAO: Chua co model!")
            print("Vui long chay: python3 train_with_real_data.py")
            print("\nCHU Y: Ban co the chay thu nghiem khong co model")
            print("       Nhung se khong co ket qua nhan dang")

    def process_landmarks(self, hand_landmarks):
        """Xu ly landmarks"""
        landmarks_array = []
        for landmark in hand_landmarks.landmark:
            landmarks_array.append([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks_array)

    def normalize_sequence(self, sequence):
        """Chuan hoa chuoi landmarks"""
        sequence = np.array(sequence)

        # Chuan hoa ve co tay
        wrist = sequence[:, 0:1, :]
        sequence = sequence - wrist

        # Scale
        max_dist = np.max(np.abs(sequence))
        if max_dist > 0:
            sequence = sequence / max_dist

        # Pad/truncate ve buffer_size
        if len(sequence) < self.buffer_size:
            padding = np.repeat(sequence[-1:], self.buffer_size - len(sequence), axis=0)
            sequence = np.concatenate([sequence, padding], axis=0)
        elif len(sequence) > self.buffer_size:
            indices = np.linspace(0, len(sequence) - 1, self.buffer_size, dtype=int)
            sequence = sequence[indices]

        return sequence

    def predict(self):
        """Du doan ky hieu"""
        if self.model is None or len(self.frame_buffer) < 15:
            return None, 0.0

        try:
            # Chuan hoa
            sequence = self.normalize_sequence(self.frame_buffer)

            # Flatten
            features = sequence.reshape(1, -1)

            # Predict
            prediction = self.model.predict(features)[0]
            proba = np.max(self.model.predict_proba(features))

            # Them vao lich su
            self.prediction_history.append((prediction, proba))
            if len(self.prediction_history) > self.history_size:
                self.prediction_history.pop(0)

            # Voting tu lich su
            if len(self.prediction_history) >= 3:
                recent_predictions = [p[0] for p in self.prediction_history[-3:]]
                # Kiem tra xem co 2/3 ket qua giong nhau khong
                most_common = max(set(recent_predictions), key=recent_predictions.count)
                if recent_predictions.count(most_common) >= 2:
                    avg_proba = np.mean([p[1] for p in self.prediction_history[-3:]])
                    return most_common, avg_proba

            return prediction, proba

        except Exception as e:
            print(f"Loi khi du doan: {e}")
            return None, 0.0

    def run(self):
        """Chay nhan dang thoi gian thuc"""

        print("\n" + "="*60)
        print("NHAN DANG KY HIEU THOI GIAN THUC")
        print("="*60)
        print("\nHUONG DAN:")
        print("  * Giu ky hieu trong 1-2 giay")
        print("  * Nhan 'c' de xoa buffer")
        print("  * Nhan 's' de xem danh sach ky hieu")
        print("  * Nhan 'q' de thoat")

        if self.model is None:
            print("\nCHU Y: Dang chay che do demo (khong co model)")
            input("\nNhan ENTER de tiep tuc...")
        else:
            print(f"\nSo ky hieu co the nhan dang: {len(set(self.sign_names.keys()))}")
            input("\nNhan ENTER de bat dau...")

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print("LOI: Khong mo duoc camera!")
            return

        print("\n>>> Camera da san sang!")

        last_prediction = None
        last_proba = 0.0
        stable_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Phat hien ban tay
            results = self.hands.process(frame_rgb)

            hand_detected = False
            if results.multi_hand_landmarks:
                hand_detected = True
                hand_landmarks = results.multi_hand_landmarks[0]

                # Ve landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
                )

                # Them vao buffer
                landmarks = self.process_landmarks(hand_landmarks)
                self.frame_buffer.append(landmarks)

                # Gioi han buffer
                if len(self.frame_buffer) > self.buffer_size:
                    self.frame_buffer.pop(0)

                # Predict moi 5 frames
                if len(self.frame_buffer) >= 20 and len(self.frame_buffer) % 5 == 0:
                    prediction, proba = self.predict()
                    if prediction is not None and proba > 0.5:
                        if prediction == last_prediction:
                            stable_count += 1
                        else:
                            stable_count = 0
                            last_prediction = prediction
                            last_proba = proba

                        # Cap nhat proba trung binh
                        if stable_count > 0:
                            last_proba = (last_proba + proba) / 2
            else:
                # Khong thay tay -> reset
                if len(self.frame_buffer) > 0:
                    self.frame_buffer = []
                    stable_count = 0

            # ===== HIEN THI THONG TIN =====

            # Header - Nen den
            cv2.rectangle(frame, (0, 0), (1280, 100), (0, 0, 0), -1)

            cv2.putText(frame, "NHAN DANG KY HIEU THOI GIAN THUC", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

            # Trang thai
            status_text = "Da phat hien ban tay" if hand_detected else "Chua phat hien ban tay"
            status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
            cv2.putText(frame, status_text, (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            # Buffer status
            buffer_percent = int((len(self.frame_buffer) / self.buffer_size) * 100)
            buffer_text = f"Buffer: {len(self.frame_buffer)}/{self.buffer_size} ({buffer_percent}%)"
            buffer_color = (0, 255, 0) if len(self.frame_buffer) >= 20 else (200, 200, 200)
            cv2.putText(frame, buffer_text, (900, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, buffer_color, 2)

            # ===== KET QUA NHAN DANG =====
            if last_prediction is not None and stable_count >= 1:
                sign_name = self.sign_names.get(last_prediction, f"Ky hieu {last_prediction}")

                # Mau sac theo do tin cay
                if last_proba >= 0.8:
                    box_color = (0, 150, 0)  # Xanh dam
                    border_color = (0, 255, 0)
                    confidence_color = (0, 255, 0)
                elif last_proba >= 0.6:
                    box_color = (0, 100, 100)  # Vang
                    border_color = (0, 255, 255)
                    confidence_color = (0, 255, 255)
                else:
                    box_color = (0, 50, 100)  # Cam
                    border_color = (0, 165, 255)
                    confidence_color = (0, 165, 255)

                # Background cho ket qua
                cv2.rectangle(frame, (50, 200), (650, 400), box_color, -1)
                cv2.rectangle(frame, (50, 200), (650, 400), border_color, 5)

                # Ten ky hieu
                cv2.putText(frame, sign_name, (80, 290),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

                # Do tin cay
                confidence_text = f"Do tin cay: {last_proba*100:.1f}%"
                cv2.putText(frame, confidence_text, (80, 360),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, confidence_color, 3)

                # ID ky hieu
                id_text = f"(ID: {last_prediction})"
                cv2.putText(frame, id_text, (500, 360),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            elif self.model is None:
                # Che do demo
                cv2.putText(frame, "CHE DO DEMO - Chua co model",
                           (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

            # ===== HUONG DAN =====
            cv2.putText(frame, "Phim tat: 'c' xoa buffer | 's' danh sach | 'q' thoat",
                       (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            # Hien thi cua so
            cv2.imshow('Nhan dang ky hieu VSL - Thu vien so', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n>>> Dang thoat...")
                break
            elif key == ord('c'):
                self.frame_buffer = []
                self.prediction_history = []
                last_prediction = None
                stable_count = 0
                print(">>> Da xoa buffer")
            elif key == ord('s'):
                self.show_sign_list()

        cap.release()
        cv2.destroyAllWindows()

        print("\n" + "="*60)
        print("DA THOAT")
        print("="*60)

    def show_sign_list(self):
        """Hien thi danh sach ky hieu"""
        print("\n" + "="*60)
        print("DANH SACH 30 KY HIEU")
        print("="*60)

        print("\nNHOM 1: SO (0-9)")
        for i in range(10):
            print(f"  {i}: {self.sign_names[i]}")

        print("\nNHOM 2: LOAI TAI LIEU (10-14)")
        for i in range(10, 15):
            print(f"  {i}: {self.sign_names[i]}")

        print("\nNHOM 3: HANH DONG (15-19)")
        for i in range(15, 20):
            print(f"  {i}: {self.sign_names[i]}")

        print("\nNHOM 4: THUOC TINH (20-24)")
        for i in range(20, 25):
            print(f"  {i}: {self.sign_names[i]}")

        print("\nNHOM 5: CHU DE (25-29)")
        for i in range(25, 30):
            print(f"  {i}: {self.sign_names[i]}")

        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    recognizer = RealtimeRecognizer()
    recognizer.run()