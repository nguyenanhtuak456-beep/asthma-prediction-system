import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from catboost import CatBoostClassifier
from huggingface_hub import hf_hub_download
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, cohen_kappa_score, mean_absolute_error,
)

# ===================== CẤU HÌNH =====================
HF_REPO_ID = "TuDUE/asthma-catboost-weights"
MODEL_FILENAME = "asthma_model_v2.cbm"
STATE_FILENAME = "last_trained_v2.json"
DATA_PATH = "dataset/live_processed.csv"
OUT_DIR = "predictions"
TARGET = "target_next_hour"
MIN_EVAL_ROWS = 200          # ít hơn số dòng chưa-thấy này thì chưa đủ tin cậy để chấm điểm
LABELS = [0, 1, 2, 3]


def compute_metrics(y_true, y_pred):
    present = sorted(set(y_true) | set(y_pred))
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=LABELS)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    overall = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=present, average="weighted", zero_division=0),
        "mae_level": mean_absolute_error(y_true, y_pred),
        "within_1_level": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "qwk": float(qwk) if not np.isnan(qwk) else np.nan,
    }
    per_class = pd.DataFrame({
        "class": LABELS,
        "precision": p,
        "recall": np.where(s > 0, r, np.nan),   # lớp không xuất hiện thì recall = NaN
        "f1": np.where(s > 0, f, np.nan),
        "support": s,
    })
    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=LABELS),
        index=[f"true_{c}" for c in LABELS],
        columns=[f"pred_{c}" for c in LABELS],
    )
    return overall, per_class, cm


def main():
    token = os.environ.get("HF_TOKEN")
    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME, token=token)
        state_path = hf_hub_download(repo_id=HF_REPO_ID, filename=STATE_FILENAME, token=token)
    except Exception as e:
        print(f"Chưa có model/mốc phiên bản mới trên Hugging Face ({e}) -> bỏ qua đánh giá.")
        sys.exit(0)

    model = CatBoostClassifier()
    model.load_model(model_path)
    with open(state_path, encoding="utf-8") as f:
        last_trained = pd.Timestamp(json.load(f)["last_trained_time"])

    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"])
    now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None)

    # Chỉ chấm trên dòng MÔ HÌNH CHƯA TỪNG HỌC, và giờ sau của dòng đó đã có số liệu thật
    ev = df[
        (df["time"] > last_trained)
        & (df["time"] + pd.Timedelta(hours=1) <= now_utc)
    ].dropna(subset=[TARGET]).copy()
    print(f"Mốc học cuối: {last_trained} | số dòng chưa học để chấm điểm: {len(ev):,}")
    if len(ev) < MIN_EVAL_ROWS:
        print(f"Ít hơn {MIN_EVAL_ROWS} dòng -> chưa đủ để đánh giá, bỏ qua lần này.")
        sys.exit(0)

    X = ev[model.feature_names_].fillna(0)
    y_true = ev[TARGET].astype(int).to_numpy()
    y_model = model.predict(X).flatten().astype(int)
    y_base = ev["asthma_risk_level"].astype(int).to_numpy()   # baseline: giờ sau = giờ này

    rows, pcs = [], []
    for name, y_pred in (("catboost", y_model), ("baseline_giu_nguyen", y_base)):
        overall, per_class, cm = compute_metrics(y_true, y_pred)
        rows.append({
            "timestamp_utc": str(now_utc.floor("s")),
            "model": name,
            "n_rows": len(ev),
            "eval_from": str(ev["time"].min()),
            "eval_to": str(ev["time"].max()),
            **{k: round(v, 4) if pd.notna(v) else v for k, v in overall.items()},
        })
        per_class.insert(0, "model", name)
        pcs.append(per_class)
        cm.to_csv(os.path.join(OUT_DIR, f"confusion_matrix_{name}.csv"))

        print("\n" + "=" * 60)
        print(f"MÔ HÌNH: {name}")
        print("=" * 60)
        for k, v in overall.items():
            print(f"{k:16s}: {v:.4f}" if pd.notna(v) else f"{k:16s}: n/a")
        print("\nPrecision / Recall / F1 từng lớp:")
        print(per_class.round(4).to_string(index=False))
        print("\nConfusion matrix (hàng = thực tế, cột = dự đoán):")
        print(cm.to_string())

    # metrics.csv: ghi nối thêm mỗi lần chạy; metrics_per_class.csv: bản mới nhất
    mpath = os.path.join(OUT_DIR, "metrics.csv")
    new = pd.DataFrame(rows)
    if os.path.exists(mpath):
        new = pd.concat([pd.read_csv(mpath), new], ignore_index=True).tail(2000)
    new.to_csv(mpath, index=False)
    pd.concat(pcs).round(4).to_csv(os.path.join(OUT_DIR, "metrics_per_class.csv"), index=False)
    print(f"\nĐã ghi {mpath}")


if __name__ == "__main__":
    main()
