import os
import shutil
import pandas as pd
from catboost import CatBoostClassifier
from huggingface_hub import hf_hub_download, upload_file

# 1. Cấu hình thông tin Hugging Face Repo
HF_REPO_ID = "TuDUE/asthma-catboost-weights"
MODEL_FILENAME = "asthma_model.cbm"
LOCAL_MODEL_PATH = MODEL_FILENAME

print("=== BẮT ĐẦU QUÁ TRÌNH FINE-TUNE TỰ ĐỘNG ===")

# 2. Tải mô hình cũ từ Hugging Face về (nếu có)
try:
    print(f"Đang tải mô hình từ Hugging Face Hub: {HF_REPO_ID}...")
    downloaded_model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME,
        token=os.environ.get("HF_TOKEN"),
    )
    # Chỉ chép nội dung (copyfile), không chép quyền "chỉ đọc" của file trong cache
    if os.path.exists(LOCAL_MODEL_PATH):
        os.remove(LOCAL_MODEL_PATH)
    shutil.copyfile(downloaded_model_path, LOCAL_MODEL_PATH)
    print("✅ Tải mô hình cũ thành công!")
except Exception as e:
    print(f"⚠️ Không tìm thấy mô hình cũ trên Hugging Face ({e}), sẽ tiến hành train mới hoàn toàn.")

# 3. Đọc dữ liệu mới từ Bước 1
data_path = "dataset/live_processed.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {data_path}")

df = pd.read_csv(data_path)
print(f"Đã đọc dữ liệu thành công với {len(df):,} dòng.")

# 4. Chuẩn bị tập Features (X) và Label (y)
features = [col for col in df.columns if col not in ['time', 'city', 'region', 'asthma_risk_level']]
target = 'asthma_risk_level'

X = df[features].fillna(0)
y = df[target]


def new_model():
    return CatBoostClassifier(
        iterations=100,          # Số vòng lặp fine-tune thêm
        learning_rate=0.03,      # Learning rate nhỏ để không phá vỡ tri thức cũ
        depth=6,
        loss_function='MultiClass',
        random_seed=42,
    )


# 5. Fine-tune (hoặc train mới nếu chưa có model cũ / model cũ không tương thích)
model = new_model()
if os.path.exists(LOCAL_MODEL_PATH):
    print("🔄 Đang thực hiện Fine-tune với init_model...")
    try:
        model.fit(X, y, init_model=LOCAL_MODEL_PATH, verbose=10)
    except Exception as e:
        print(f"⚠️ Không fine-tune được từ model cũ ({e}), chuyển sang train mới hoàn toàn.")
        model = new_model()
        model.fit(X, y, verbose=10)
else:
    print("🚀 Đang huấn luyện mô hình mới từ đầu...")
    model.fit(X, y, verbose=10)

# 6. Lưu lại file trọng số mô hình mới (xóa file cũ trước để tránh lỗi quyền ghi)
if os.path.exists(LOCAL_MODEL_PATH):
    os.remove(LOCAL_MODEL_PATH)
model.save_model(LOCAL_MODEL_PATH)
print(f"💾 Đã lưu mô hình mới tại: {LOCAL_MODEL_PATH}")

# 7. Tự động đẩy file .cbm mới lên Hugging Face Hub
print("🚀 Đang đẩy mô hình lên Hugging Face Hub...")
upload_file(
    path_or_fileobj=LOCAL_MODEL_PATH,
    path_in_repo=MODEL_FILENAME,
    repo_id=HF_REPO_ID,
    repo_type="model",
    token=os.environ.get("HF_TOKEN"),
)
print("🎉 Hoàn tất Fine-tune và cập nhật mô hình lên Hugging Face thành công!")
