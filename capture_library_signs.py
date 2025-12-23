#!/usr/bin/env python3
"""
Thu thap 30 ky hieu cho thu vien so
Gom: 10 so (0-9) + 20 tu lien quan thu vien
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path
from datetime import datetime

class LibrarySignCapture:
    def __init__(self):
        # Khoi tao MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Danh sach 30 ky hieu
        self.signs = [
            # Nhom 1: So (0-9)
            {"id": 0, "name": "so_0", "desc": "Nam tay thanh qua bop"},
            {"id": 1, "name": "so_1", "desc": "Gio ngon tro len"},
            {"id": 2, "name": "so_2", "desc": "Gio ngon tro va ngon giua"},
            {"id": 3, "name": "so_3", "desc": "Gio 3 ngon"},
            {"id": 4, "name": "so_4", "desc": "Gio 4 ngon (tru ngon cai)"},
            {"id": 5, "name": "so_5", "desc": "Gio ca 5 ngon"},
            {"id": 6, "name": "so_6", "desc": "Cham ngon cai voi ngon ut"},
            {"id": 7, "name": "so_7", "desc": "Cham ngon cai voi ngon ap ut"},
            {"id": 8, "name": "so_8", "desc": "Cham ngon cai voi ngon giua"},
            {"id": 9, "name": "so_9", "desc": "Cham ngon cai voi ngon tro"},

            # Nhom 2: Loai tai lieu (10-14)
            {"id": 10, "name": "sach", "desc": "Hai tay mo ra nhu mo sach"},
            {"id": 11, "name": "bao", "desc": "Tay phai mo ra phang"},
            {"id": 12, "name": "tap_chi", "desc": "Hai tay xep chong len nhau"},
            {"id": 13, "name": "giao_trinh", "desc": "Tay mo sach, ngon tro len"},
            {"id": 14, "name": "luan_van", "desc": "Hai tay xep ngang, mo ra"},

            # Nhom 3: Hanh dong (15-19)
            {"id": 15, "name": "tim", "desc": "Tay phai xoay tron truoc mat"},
            {"id": 16, "name": "tim_kiem", "desc": "Ca 2 tay xoay tron"},
            {"id": 17, "name": "doc", "desc": "Tay trai giu, tay phai lat"},
            {"id": 18, "name": "muon", "desc": "Tay dua ra phia truoc"},
            {"id": 19, "name": "tra", "desc": "Tay rut ve phia sau"},

            # Nhom 4: Thuoc tinh (20-24)
            {"id": 20, "name": "tac_gia", "desc": "Ngon tro chi vao nguoi"},
            {"id": 21, "name": "tieu_de", "desc": "Hai tay gio len tren dau"},
            {"id": 22, "name": "nam", "desc": "Gio 5 ngon roi gap lai"},
            {"id": 23, "name": "moi", "desc": "Ngon tro gio len cao"},
            {"id": 24, "name": "cu", "desc": "Tay xoay lui ve phia sau"},

            # Nhom 5: Chu de (25-29)
            {"id": 25, "name": "cong_nghe", "desc": "Hai tay danh may tinh"},
            {"id": 26, "name": "khoa_hoc", "desc": "Tay phai ve hinh tron"},
            {"id": 27, "name": "van_hoc", "desc": "Tay cam but viet"},
            {"id": 28, "name": "lich_su", "desc": "Tay chi ve phia sau"},
            {"id": 29, "name": "toan_hoc", "desc": "Tay ve dau cong"},
        ]

        self.setup_directories()

    def setup_directories(self):
        """Tao cau truc thu muc"""
        Path("real_data/raw").mkdir(parents=True, exist_ok=True)
        Path("real_data/processed").mkdir(parents=True, exist_ok=True)

    def get_person_info(self):
        """Nhap thong tin nguoi tham gia"""
        print("\n" + "="*60)
        print("THONG TIN NGUOI THAM GIA")
        print("="*60)

        person_id = input("Nhap ten (VD: nam): ").strip() or f"person_{int(time.time())%1000}"

        print(f"\nXin chao {person_id}!")
        print(f"Ban se thu thap {len(self.signs)} ky hieu")
        print(f"Moi ky hieu thu 10 lan")
        print(f"Tong cong: {len(self.signs) * 10} mau")

        return person_id

    def capture_session(self, person_id):
        """Phien thu thap du lieu"""

        print(f"\n{'='*60}")
        print(f"BAT DAU THU THAP CHO: {person_id}")
        print(f"{'='*60}\n")

        # Mo camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print("LOI: Khong the mo camera!")
            return

        print("Da mo camera thanh cong")
        print("\nHUONG DAN:")
        print("  * Nhan SPACE de bat dau ghi (3 giay)")
        print("  * Nhan 'n' de chuyen sang ky hieu tiep theo")
        print("  * Nhan 'r' de ghi lai ky hieu hien tai")
        print("  * Nhan 'q' de thoat")
        print(f"\nSo ky hieu can thu: {len(self.signs)}")
        print(f"Moi ky hieu ghi 10 lan\n")

        current_sign_idx = 0
        current_count = 0
        total_per_sign = 10

        is_recording = False
        recorded_frames = []
        record_start_time = 0
        record_duration = 3

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Phat hien ban tay
            results = self.hands.process(frame_rgb)

            # Ve landmarks
            hand_detected = False
            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                        self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
                    )

            # Hien thi thong tin
            if current_sign_idx >= len(self.signs):
                break

            sign = self.signs[current_sign_idx]

            # ===== HEADER - Nen den =====
            cv2.rectangle(frame, (0, 0), (1280, 140), (0, 0, 0), -1)

            # Thong tin nguoi
            cv2.putText(frame, f"Nguoi: {person_id}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Ky hieu hien tai
            cv2.putText(frame, f"Ky hieu {sign['id']}: {sign['name']} (Lan {current_count + 1}/{total_per_sign})",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # Mo ta
            cv2.putText(frame, f"Mo ta: {sign['desc']}",
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            # ===== TIEN DO - Thanh tien trinh =====
            progress = (current_sign_idx * total_per_sign + current_count) / (len(self.signs) * total_per_sign)
            cv2.rectangle(frame, (10, 680), (1270, 710), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 680), (int(10 + 1260 * progress), 710), (0, 255, 0), -1)

            # Hien thi so phan tram
            percent_text = f"Tien do: {current_sign_idx}/{len(self.signs)} ky hieu - {int(progress * 100)}%"
            cv2.putText(frame, percent_text, (480, 702),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # ===== TRANG THAI BAN TAY =====
            status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
            status_text = "Da phat hien ban tay!" if hand_detected else "Chua phat hien ban tay"
            cv2.putText(frame, status_text, (10, 640),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            # ===== NEU DANG GHI =====
            if is_recording:
                elapsed = time.time() - record_start_time
                remaining = record_duration - elapsed

                if remaining > 0:
                    # Ve vong tron countdown
                    cv2.circle(frame, (640, 360), 120, (0, 0, 255), 10)
                    cv2.putText(frame, f"{int(remaining) + 1}", (590, 390),
                               cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 10)

                    # Chu "DANG GHI"
                    cv2.putText(frame, "DANG GHI...", (500, 470),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)

                    # Thu thap frame
                    if hand_detected:
                        hand_landmarks = results.multi_hand_landmarks[0]
                        landmarks_array = []
                        for landmark in hand_landmarks.landmark:
                            landmarks_array.append([landmark.x, landmark.y, landmark.z])
                        recorded_frames.append(landmarks_array)
                else:
                    # Het thoi gian
                    is_recording = False
                    if len(recorded_frames) > 10:
                        self.save_recording(recorded_frames, person_id, sign, current_count)
                        current_count += 1
                    else:
                        print("CANH BAO: Qua it frame, vui long thu lai!")

                    recorded_frames = []

                    # Neu du so lan
                    if current_count >= total_per_sign:
                        current_count = 0
                        current_sign_idx += 1
                        print(f"\n=== HOAN THANH KY HIEU: {sign['name']} ===\n")
            else:
                # ===== HUONG DAN =====
                cv2.putText(frame, "Nhan SPACE de bat dau ghi (3 giay)",
                           (340, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.putText(frame, "Nhan 'n' de bo qua | 'r' de ghi lai | 'q' de thoat",
                           (240, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

            # Hien thi cua so
            cv2.imshow('Thu thap ky hieu - Thu vien so VSL', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' ') and not is_recording and hand_detected:
                # Bat dau ghi
                is_recording = True
                record_start_time = time.time()
                recorded_frames = []
                print(f">>> Dang ghi: {sign['name']} (Lan {current_count + 1}/{total_per_sign})")

            elif key == ord(' ') and not is_recording and not hand_detected:
                print("CANH BAO: Chua phat hien ban tay! Dua tay vao khung hinh.")

            elif key == ord('n'):
                # Chuyen ky hieu
                current_count = 0
                current_sign_idx += 1
                if current_sign_idx >= len(self.signs):
                    break
                print(f"\n>>> Chuyen sang: {self.signs[current_sign_idx]['name']}\n")

            elif key == ord('r'):
                # Ghi lai
                if current_count > 0:
                    current_count -= 1
                print(f">>> Ghi lai: {sign['name']} (Lan {current_count + 1}/{total_per_sign})")

            elif key == ord('q'):
                print("\n>>> Dang thoat...")
                break

        cap.release()
        cv2.destroyAllWindows()

        print(f"\n{'='*60}")
        print(f"DA HOAN THANH THU THAP CHO: {person_id}")
        print(f"{'='*60}")

    def save_recording(self, frames, person_id, sign, count):
        """Luu du lieu da ghi"""

        # Chuyen thanh numpy array
        landmarks = np.array(frames)

        # Chuan hoa ve co tay (diem 0)
        wrist = landmarks[:, 0:1, :]
        landmarks = landmarks - wrist

        # Scale ve [-1, 1]
        max_dist = np.max(np.abs(landmarks))
        if max_dist > 0:
            landmarks = landmarks / max_dist

        # Pad hoac truncate ve 30 frames
        target_length = 30
        current_length = len(landmarks)

        if current_length < target_length:
            padding = np.repeat(landmarks[-1:], target_length - current_length, axis=0)
            landmarks = np.concatenate([landmarks, padding], axis=0)
        elif current_length > target_length:
            indices = np.linspace(0, current_length - 1, target_length, dtype=int)
            landmarks = landmarks[indices]

        # Tao ten file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{person_id}_sign_{sign['id']:03d}_{count:02d}_{timestamp}.npy"
        filepath = Path("real_data/processed") / filename

        # Luu file
        np.save(filepath, landmarks)

        print(f"    Da luu: {filename} ({len(frames)} -> {len(landmarks)} frames)")

    def run(self):
        """Chay chuong trinh"""

        print("\n" + "="*60)
        print("CONG CU THU THAP DU LIEU THU VIEN SO VSL")
        print("="*60)
        print("\nGom 30 ky hieu:")
        print("  - Nhom 1: So 0-9 (10 ky hieu)")
        print("  - Nhom 2: Loai tai lieu (5 ky hieu)")
        print("  - Nhom 3: Hanh dong (5 ky hieu)")
        print("  - Nhom 4: Thuoc tinh (5 ky hieu)")
        print("  - Nhom 5: Chu de (5 ky hieu)")

        # Nhap thong tin
        person_id = self.get_person_info()

        input("\nNhan ENTER de bat dau...")

        # Bat dau thu thap
        self.capture_session(person_id)

        print("\n" + "="*60)
        print("KET THUC THU THAP")
        print("="*60)
        print(f"\nDu lieu da luu trong: real_data/processed/")

        # Thong ke
        processed_files = list(Path("real_data/processed").glob(f"{person_id}_*.npy"))
        print(f"Tong so file vua thu: {len(processed_files)}")
        print(f"\nTIEP THEO:")
        print(f"  1. Kiem tra du lieu: python3 manage_data.py")
        print(f"  2. Train mo hinh: python3 train_with_real_data.py")
        print(f"  3. Test: python3 test_realtime.py")

if __name__ == "__main__":
    capture = LibrarySignCapture()
    capture.run()