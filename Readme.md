Thuyết trình thực tiễn thực hiện:
Cài python version 3.14 mới -> tensoflow không hỗ trợ thư viện nên cài 3.11
# Tạo venv với Python 3.11:  python3.11 -m venv venv -> kích hoạt: source venv/bin/activate -> sau đó kiểm tra đã dùng chưa : python --version
tạo file requirements.txt với : cat > requirements.txt << 'EOF' numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.0 tensorflow==2.15.0 opencv-python==4.8.1.78 mediapipe==0.10.8 matplotlib==3.8.2 seaborn==0.13.0 tqdm==4.66.1 Pillow==10.1.0 EOF -> sau đó : pip install --upgrade pip pip install -r requirements.txt -> có thể lỗi phiên bản ->
Lần lượt kiểm tra cài đặt: python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)" python -c "import numpy as np; print('NumPy:', np.__version__)" python -c "import sklearn; print('Scikit-learn:', sklearn.__version__)"




Sau khi kiểm tra
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"                                                          -> TensorFlow: 2.15.0
python -c "import numpy as np; print('NumPy:', np.__version__)"                                           
-> NumPy: 1.24.3
python -c "import sklearn; print('Scikit-learn:', sklearn.__version__)"
-> Scikit-learn: 1.3.0
- Lưu ý python nên dùng trên pycharm IDE
- Tạo dữ liệu mẫu


# Train mô hình python train_compare_models.py








Sau khi chạy dữ liệu ảo để test so sánh các phương pháp, chuyển sang với dữ liệu thật
Tạo file capture_real_data.py -> tạo file train thật train_with_real_data.py
Cách dùng:
Chạy file python3 capture_real_data.py



🎯 DANH SÁCH 30 KÝ HIỆU
Nhóm 1: Số (0-9) - 10 ký hiệu
số_0 - Nắm tay thành quả bóp
số_1 - Giơ ngón trỏ lên
số_2 - Giơ ngón trỏ và ngón giữa
số_3 - Giơ 3 ngón
số_4 - Giơ 4 ngón (trừ ngón cái)
số_5 - Giơ cả 5 ngón
số_6 - Chạm ngón cái với ngón út
số_7 - Chạm ngón cái với ngón áp út
số_8 - Chạm ngón cái với ngón giữa
số_9 - Chạm ngón cái với ngón trỏ
Nhóm 2: Loại tài liệu (10-14) - 5 ký hiệu
sach - Hai tay mở ra như mở sách
bao - Tay phải mở ra phẳng
tap_chi - Hai tay xếp chồng lên nhau
giao_trinh - Tay mở sách, ngón trỏ lên
luan_van - Hai tay xếp ngang, mở ra
Nhóm 3: Hành động (15-19) - 5 ký hiệu
tim - Tay phải xoay tròn trước mặt
tim_kiem - Cả 2 tay xoay tròn
doc - Tay trái giữ, tay phải lật
muon - Tay đưa ra phía trước
tra - Tay rút về phía sau
Nhóm 4: Thuộc tính (20-24) - 5 ký hiệu
tac_gia - Ngón trỏ chỉ vào người
tieu_de - Hai tay giơ lên trên đầu
nam - Giơ 5 ngón rồi gập lại
moi - Ngón trỏ giơ lên cao
cu - Tay xoay lùi về phía sau
Nhóm 5: Chủ đề (25-29) - 5 ký hiệu
cong_nghe - Hai tay đánh máy tính
khoa_hoc - Tay phải vẽ hình tròn
van_hoc - Tay cầm bút viết
lich_su - Tay chỉ về phía sau
toan_hoc - Tay vẽ dấu cộng

cách chạy:
chạy lệnh python3 capture_library_signs.py
### **2. Nhập tên** ``` Nhap ten (VD: nam): nam ``` ### **3. Thực hiện** - Mỗi ký hiệu thu **10 lần** - Tổng: **30 ký hiệu × 10 lần = 300 mẫu** - Thời gian: ~30-45 phút ### **4. Kết quả** ``` real_data/processed/ nam_sign_000_00_20231223_143022.npy nam_sign_000_01_20231223_143028.npy ... nam_sign_029_09_20231223_150512.npy

xem thống kê: python3 manage_data.py
train model: python3 train_with_real_data.py
test thời gian thực: python3 test_realtime.py


Sau khi huấn luyện xong mô hình thì test real time -> python3 test_realtime.py
kết quả 

