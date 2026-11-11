import os
import pandas as pd
from catboost import CatBoostClassifier
from huggingface_hub import hf_hub_download, upload_file

# 1. Cấu hình thông tin Hugging Face Repo của bạn
HF_REPO_ID = "Tên_User_HuggingFace_Của_Bạn/asthma-catboost-weights"  # Thay bằng ID repo của bạn trên HF
MODEL_FILENAME = "asthma_model.cbm"
LOCAL_MODEL_PATH = MODEL_FILENAME

print("=== BẮT ĐẦU QUÁ TRÌNH FINE-TUNE TỰ ĐỘNG ===")

# 2. Tải mô hình cũ từ Hugging Face về (nếu có)
try:
    print(f"Đang tải mô hình từ Hugging Face Hub: {HF_REPO_ID}...")
    downloaded_model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME,
        token=os.environ.get("HF_TOKEN")
    )
    # Copy về thư mục làm việc hiện tại
    if downloaded_model_path != LOCAL_MODEL_PATH:
        import shutil
        shutil.copy(downloaded_model_path, LOCAL_MODEL_PATH)
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
# (Điều chỉnh các cột đặc trưng phù hợp với bài toán của bạn)
features = [col for col in df.columns if col not in ['time', 'city', 'region', 'asthma_risk_level']]
target = 'asthma_risk_level'

X = df[features].fillna(0)
y = df[target]

# 5. Khởi tạo và Fine-tune CatBoost Model
model = CatBoostClassifier(
    iterations=100,          # Số vòng lặp fine-tune thêm
    learning_rate=0.03,      # Learning rate nhỏ để không phá vỡ tri thức cũ
    depth=6,
    loss_function='MultiClass',
    random_seed=42
)

if os.path.exists(LOCAL_MODEL_PATH):
    print("🔄 Đang thực hiện Fine-tune với init_model...")
    model.fit(X, y, init_model=LOCAL_MODEL_PATH, verbose=10)
else:
    print("🚀 Đang huấn luyện mô hình mới từ đầu...")
    model.fit(X, y, verbose=10)

# 6. Lưu lại file trọng số mô hình mới
model.save_model(LOCAL_MODEL_PATH)
print(f"💾 Đã lưu mô hình mới tại: {LOCAL_MODEL_PATH}")

# 7. Tự động đẩy file .cbm mới lên Hugging Face Hub
print("🚀 Đang đẩy mô hình lên Hugging Face Hub...")
upload_file(
    path_or_fileobj=LOCAL_MODEL_PATH,
    path_in_repo=MODEL_FILENAME,
    repo_id=HF_REPO_ID,
    repo_type="model",
    token=os.environ.get("HF_TOKEN")
)
print("🎉 Hoàn tất Fine-tune và cập nhật mô hình lên Hugging Face thành công!")
