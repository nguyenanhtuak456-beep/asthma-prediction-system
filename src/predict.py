import os
import pandas as pd
from catboost import CatBoostClassifier
from huggingface_hub import hf_hub_download

HF_REPO_ID = "TuDUE/asthma-catboost-weights"
MODEL_FILENAME = "asthma_model_v2.cbm"
DATA_PATH = "dataset/live_processed.csv"
OUT_DIR = "predictions"
OUT_PATH = os.path.join(OUT_DIR, "latest_predictions.csv")
LABELS = {0: "Thấp", 1: "Trung bình", 2: "Cao", 3: "Rất cao"}

print("=== DỰ ĐOÁN NGUY CƠ HEN SUYỄN ===")

# 1. Tải model mới nhất từ Hugging Face
model_path = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename=MODEL_FILENAME,
    token=os.environ.get("HF_TOKEN"),
)
model = CatBoostClassifier()
model.load_model(model_path)

# 2. Đọc dữ liệu và lấy dòng mới nhất (đã có số liệu thật) của mỗi tỉnh
df = pd.read_csv(DATA_PATH)
df["time"] = pd.to_datetime(df["time"])
now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None)
latest = (
    df[(df["time"] <= now_utc)]
    .dropna(subset=["pm2_5"])
    .sort_values("time")
    .groupby("city")
    .tail(1)
    .copy()
)

# 3. Dự đoán, dùng đúng danh sách cột mà model đã học
X = latest[model.feature_names_].fillna(0)
latest["predicted_class"] = model.predict(X).flatten().astype(int)
latest["predicted_level"] = latest["predicted_class"].map(LABELS)

# 4. Lưu file kết quả nhỏ (khoảng 63 dòng)
latest["forecast_for"] = latest["time"] + pd.Timedelta(hours=1)   # mức nguy cơ này là DỰ BÁO cho giờ kế tiếp
out_cols = ["city", "region", "time", "forecast_for", "pm2_5", "us_aqi", "predicted_class", "predicted_level"]
os.makedirs(OUT_DIR, exist_ok=True)
latest[out_cols].sort_values("city").to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print(f"Đã lưu {len(latest)} dự đoán tại: {OUT_PATH}")
