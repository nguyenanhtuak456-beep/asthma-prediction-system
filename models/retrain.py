import os
import sys
import json
import shutil
import pandas as pd
from catboost import CatBoostClassifier
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download

# ===================== CẤU HÌNH =====================
HF_REPO_ID = "TuDUE/asthma-catboost-weights"
MODEL_FILENAME = "asthma_model.cbm"
STATE_FILENAME = "last_trained.json"      # file lưu mốc "lần học cuối" trên Hugging Face
LOCAL_MODEL_PATH = MODEL_FILENAME
DATA_PATH = "dataset/live_processed.csv"

FULL_ITERATIONS = 100          # số cây khi học lần đầu / học lại từ đầu
INCREMENTAL_ITERATIONS = 30    # số cây thêm vào mỗi lần học dữ liệu mới
MIN_NEW_ROWS = 500             # ít hơn số dòng mới này thì bỏ qua (63 tỉnh x 8 giờ ~ 504)

HF_TOKEN = os.environ.get("HF_TOKEN")
print("=== BẮT ĐẦU QUÁ TRÌNH FINE-TUNE TỰ ĐỘNG (v3) ===")

# ===================== 1. TẢI MODEL CŨ =====================
have_model = False
try:
    print(f"Đang tải mô hình từ Hugging Face Hub: {HF_REPO_ID}...")
    downloaded_model_path = hf_hub_download(
        repo_id=HF_REPO_ID, filename=MODEL_FILENAME, token=HF_TOKEN
    )
    # copyfile: chỉ chép nội dung, không chép quyền "chỉ đọc" của file trong cache
    if os.path.exists(LOCAL_MODEL_PATH):
        os.remove(LOCAL_MODEL_PATH)
    shutil.copyfile(downloaded_model_path, LOCAL_MODEL_PATH)
    os.chmod(LOCAL_MODEL_PATH, 0o644)
    have_model = True
    print("✅ Tải mô hình cũ thành công!")
except Exception as e:
    print(f"⚠️ Không tìm thấy mô hình cũ trên Hugging Face ({e}), sẽ train mới hoàn toàn.")

# ===================== 2. ĐỌC MỐC "LẦN HỌC CUỐI" =====================
last_trained = None
if have_model:
    try:
        state_path = hf_hub_download(
            repo_id=HF_REPO_ID, filename=STATE_FILENAME, token=HF_TOKEN
        )
        with open(state_path, encoding="utf-8") as f:
            last_trained = pd.Timestamp(json.load(f)["last_trained_time"])
        print(f"📌 Lần học cuối đến thời điểm: {last_trained} (UTC)")
    except Exception as e:
        print(f"ℹ️ Chưa có mốc thời gian ({e}), lần này sẽ học toàn bộ dữ liệu đang có.")

# ===================== 3. ĐỌC DỮ LIỆU VÀ LỌC PHẦN MỚI =====================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df["time"] = pd.to_datetime(df["time"])
print(f"Đã đọc dữ liệu thành công với {len(df):,} dòng.")

# Bỏ các giờ tương lai (API trả cả số liệu dự báo của hôm nay, chưa phải số liệu thật)
now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None)
df = df[df["time"] <= now_utc]

if last_trained is not None:
    if df["time"].min() > last_trained:
        print("⚠️ Dữ liệu cào về bắt đầu SAU mốc lần học cuối: có khoảng trống chưa học được. "
              "Hãy tăng số ngày của START_DATE trong fetch_and_process.py.")
    df_new = df[df["time"] > last_trained]
    print(f"Số dòng MỚI chưa học: {len(df_new):,}")
    if len(df_new) < MIN_NEW_ROWS:
        print(f"⏭️ Ít hơn {MIN_NEW_ROWS} dòng mới, bỏ qua lần học này (model giữ nguyên).")
        sys.exit(0)
    iterations = INCREMENTAL_ITERATIONS
else:
    df_new = df
    iterations = FULL_ITERATIONS

if df_new.empty:
    print("Không có dữ liệu để học.")
    sys.exit(0)

# ===================== 4. CHUẨN BỊ X, y =====================
features = [c for c in df.columns if c not in ["time", "city", "region", "asthma_risk_level"]]
target = "asthma_risk_level"


def new_model(n_iter):
    return CatBoostClassifier(
        iterations=n_iter,
        learning_rate=0.03,
        depth=6,
        loss_function="MultiClass",
        random_seed=42,
    )


# ===================== 5. HỌC =====================
df_used = df_new
model = new_model(iterations)
if have_model:
    print(f"🔄 Fine-tune thêm {iterations} cây trên {len(df_new):,} dòng mới...")
    try:
        model.fit(df_new[features].fillna(0), df_new[target], init_model=LOCAL_MODEL_PATH, verbose=10)
    except Exception as e:
        print(f"⚠️ Không fine-tune được từ model cũ ({e}), chuyển sang train mới trên toàn bộ dữ liệu.")
        df_used = df
        model = new_model(FULL_ITERATIONS)
        model.fit(df[features].fillna(0), df[target], verbose=10)
else:
    print(f"🚀 Train mới từ đầu trên {len(df_new):,} dòng...")
    model.fit(df_new[features].fillna(0), df_new[target], verbose=10)

# ===================== 6. LƯU MODEL (xóa file cũ trước để tránh lỗi quyền ghi) =====================
if os.path.exists(LOCAL_MODEL_PATH):
    os.remove(LOCAL_MODEL_PATH)
model.save_model(LOCAL_MODEL_PATH)
print(f"💾 Đã lưu mô hình mới tại: {LOCAL_MODEL_PATH}")

# ===================== 7. ĐẨY MODEL + MỐC THỜI GIAN LÊN HF (1 commit duy nhất) =====================
new_cutoff = df_used["time"].max()
state = {
    "last_trained_time": str(new_cutoff),
    "rows_used": int(len(df_used)),
    "updated_at_utc": str(now_utc),
}
print(f"🚀 Đang đẩy mô hình + mốc thời gian ({new_cutoff}) lên Hugging Face Hub...")
HfApi(token=HF_TOKEN).create_commit(
    repo_id=HF_REPO_ID,
    repo_type="model",
    operations=[
        CommitOperationAdd(path_in_repo=MODEL_FILENAME, path_or_fileobj=LOCAL_MODEL_PATH),
        CommitOperationAdd(
            path_in_repo=STATE_FILENAME,
            path_or_fileobj=json.dumps(state, ensure_ascii=False, indent=2).encode("utf-8"),
        ),
    ],
    commit_message=f"Fine-tune đến {new_cutoff} ({len(df_used)} dòng)",
)
print("🎉 Hoàn tất Fine-tune và cập nhật mô hình + mốc thời gian lên Hugging Face thành công!")
