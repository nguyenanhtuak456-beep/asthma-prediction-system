import os
import pandas as pd
import numpy as np

print("=" * 70)
print("BƯỚC: XÂY DỰNG ĐẶC TRƯNG VÀ CHIA TẬP DỮ LIỆU (TRAIN / VAL / TEST)")
print("=" * 70)

# =====================================================================
# 1. ĐỌC DỮ LIỆU ĐÃ CÀO SẴN (DỮ LIỆU SẠCH)
# =====================================================================
BASE_DIR = "."
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MASTER_FILE = os.path.join(DATASET_DIR, "vietnam_air_quality_combined_full.csv")

if not os.path.exists(MASTER_FILE):
    raise FileNotFoundError(f"Không tìm thấy file tại {MASTER_FILE}")

print("\n1. Đang nạp dữ liệu sạch từ:", MASTER_FILE)
df = pd.read_csv(MASTER_FILE)
df['time'] = pd.to_datetime(df['time'])

# BẮT BUỘC: Sắp xếp theo thành phố và thời gian để tính Lag/Rolling chính xác
df = df.sort_values(by=['city', 'time']).reset_index(drop=True)
print("-> Kích thước dữ liệu:", df.shape)

# =====================================================================
# 2. TẠO BIẾN MỤC TIÊU (TARGET T+3H)
# =====================================================================
print("\n2. Đang tạo biến mục tiêu dự báo trước 3 giờ...")
df['target'] = df.groupby('city')['asthma_risk_level'].shift(-3)

# =====================================================================
# 3. FEATURE ENGINEERING: LAG & ROLLING
# =====================================================================
print("3. Đang tạo đặc trưng chuỗi thời gian (Lag 1, 3, 6 và Rolling 3)...")
feature_cols_base = ['pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 'us_aqi']
lag_cols = []

for col in feature_cols_base:
    # Đặc trưng trễ (Lag)
    for lag in [1, 3, 6]:
        new_col = f'{col}_lag{lag}'
        df[new_col] = df.groupby('city')[col].shift(lag)
        lag_cols.append(new_col)

    # Trung bình trượt (Rolling)
    rolling_col = f'{col}_rolling_mean3'
    df[rolling_col] = df.groupby('city')[col].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    lag_cols.append(rolling_col)

# Bỏ các dòng bị NaN do quá trình dịch chuyển (shift)
df = df.dropna(subset=['target'] + lag_cols).copy()
df['target'] = df['target'].astype(int)
df['city'] = df['city'].astype(str)
df = df.reset_index(drop=True)
print("-> Số dòng sau khi tạo đặc trưng:", len(df))

# =====================================================================
# 4. CHIA TẬP TRAIN / VAL / TEST THEO THỜI GIAN
# =====================================================================
print("\n4. Đang phân chia Train / Validation / Test...")

# Tập Train/Val lấy trước 2026, tập Test lấy từ 2026 trở đi
train_df = df[df['time'].dt.year < 2026].copy()
test_df = df[df['time'].dt.year >= 2026].copy()

# Xử lý dự phòng nếu dữ liệu cào về chưa có năm 2026 (chia 80/20 theo thời gian)
if test_df.empty:
    print("CẢNH BÁO: Không có dữ liệu năm 2026 -> Chia tự động 80% Train, 20% Test.")
    unique_times = np.sort(df['time'].unique())
    split_idx = int(len(unique_times) * 0.8)
    split_time = unique_times[split_idx]
    
    train_df = df[df['time'] < split_time].copy()
    test_df = df[df['time'] >= split_time].copy()

# Tách Validation từ phần Train (Lấy 15% cuối cùng của tập Train)
unique_train_times = np.sort(train_df['time'].unique())
val_idx = max(1, int(len(unique_train_times) * 0.85))
val_start = unique_train_times[val_idx]

fit_df = train_df[train_df['time'] < val_start].copy()
val_df = train_df[train_df['time'] >= val_start].copy()

print(f"-> Train: {fit_df.shape} | Từ {fit_df['time'].min()} đến {fit_df['time'].max()}")
print(f"-> Validation: {val_df.shape} | Từ {val_df['time'].min()} đến {val_df['time'].max()}")
print(f"-> Test: {test_df.shape} | Từ {test_df['time'].min()} đến {test_df['time'].max()}")

# =====================================================================
# 5. LƯU DỮ LIỆU ĐÃ XỬ LÝ (Để các mô hình sử dụng)
# =====================================================================
print("\n5. Đang xuất file dữ liệu .csv...")
os.makedirs(os.path.join(DATASET_DIR, "dataset_processed"), exist_ok=True)

# Ghi đè vào thư mục dataset để các script mô hình dễ đọc
train_path = os.path.join(DATASET_DIR, "train_processed.csv")
val_path = os.path.join(DATASET_DIR, "val_processed.csv")
test_path = os.path.join(DATASET_DIR, "test_processed.csv")

fit_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

print("✅ Đã lưu thành công:")
print(f"   + {train_path}")
print(f"   + {val_path}")
print(f"   + {test_path}")
