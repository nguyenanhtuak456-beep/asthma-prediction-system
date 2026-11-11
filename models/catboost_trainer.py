import os
import json
import shutil
from datetime import datetime

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, confusion_matrix
)

# ======================= CẤU HÌNH =======================
EXCLUDE_COLS = ['time', 'region', 'asthma_risk_level', 'target', 'city']
ITERATIONS = 800
LR_SCRATCH = 0.05
LR_FINETUNE = 0.01
DEPTH = 6
EARLY_STOP = 40
SEED = 42
FORECAST_HORIZON = 3


def find_base_dir():
    """Tự nhận diện nơi Google Drive được mount."""
    for root in ("/content/gdrive", "/content/drive"):
        base = os.path.join(root, "MyDrive", "AI")
        if os.path.isdir(base):
            return base
    raise FileNotFoundError(
        "Không tìm thấy Google Drive. Hãy mount Drive trước: "
        "from google.colab import drive; drive.mount('/content/drive')"
    )


def main():
    print("=" * 70)
    print("HUẤN LUYỆN VÀ FINE-TUNING MÔ HÌNH CATBOOST DỰ BÁO HEN SUYỄN")
    print("=" * 70)

    BASE_DIR = find_base_dir()
    DATASET_DIR = os.path.join(BASE_DIR, "dataset", "dataset_processed")
    OUT_DIR = os.path.join(BASE_DIR, "Script", "export")  # MyDrive/AI/Script/export
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"-> Thư mục gốc: {BASE_DIR}")

    # ------------------- 1. NẠP DỮ LIỆU -------------------
    print("\n1. Đang nạp bộ dữ liệu từ thư mục dataset_processed...")
    try:
        train_df = pd.read_csv(os.path.join(DATASET_DIR, "train_processed.csv"))
        val_df = pd.read_csv(os.path.join(DATASET_DIR, "val_processed.csv"))
        test_df = pd.read_csv(os.path.join(DATASET_DIR, "test_processed.csv"))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu trong {DATASET_DIR}: {e}"
        )

    # Chỉ lấy cột số, loại cột định danh/nhãn
    feature_cols = [
        c for c in train_df.columns
        if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    # Đảm bảo val/test có đủ cột đặc trưng như train
    for name, df in (("val", val_df), ("test", test_df)):
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Tập {name} thiếu các cột: {missing}")

    X_train, y_train = train_df[feature_cols], train_df['target'].astype(int)
    X_val, y_val = val_df[feature_cols], val_df['target'].astype(int)
    X_test, y_test = test_df[feature_cols], test_df['target'].astype(int)

    print(f"-> Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
    print(f"-> Số lượng đặc trưng số đưa vào mô hình: {len(feature_cols)}")
    n_nan = int(X_train.isna().sum().sum())
    print(f"-> Giá trị thiếu (NaN) trong train: {n_nan} (CatBoost tự xử lý)")
    print("-> Phân bố nhãn train:\n", y_train.value_counts().sort_index().to_string())

    # ------------------- 2. THIẾT LẬP MÔ HÌNH -------------------
    print("\n2. Kiểm tra và thiết lập cơ chế Fine-tuning CatBoost...")
    BEST_CBM_PATH = os.path.join(OUT_DIR, "asthma_catboost_old.cbm")
    META_PATH = os.path.join(OUT_DIR, "best_meta.json")
    has_old_model = os.path.exists(BEST_CBM_PATH)

    # Đọc số đặc trưng/F1 của model cũ để kiểm tra tương thích và so sánh
    old_meta = {}
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            old_meta = json.load(f)
    if has_old_model and old_meta.get("feature_cols") not in (None, feature_cols):
        print("-> Bộ đặc trưng đã thay đổi so với model cũ, không thể fine-tune.")
        has_old_model = False

    def build_model(finetune):
        return CatBoostClassifier(
            iterations=ITERATIONS,
            learning_rate=LR_FINETUNE if finetune else LR_SCRATCH,
            depth=DEPTH,
            auto_class_weights='Balanced',
            early_stopping_rounds=EARLY_STOP,
            random_seed=SEED,
            verbose=100,
        )

    def train_scratch():
        m = build_model(finetune=False)
        m.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        return m

    if has_old_model:
        print(f"-> Phát hiện trọng số cũ tại: {BEST_CBM_PATH}")
        print("-> Đang thực hiện Fine-tuning (kế thừa trọng số cũ)...")
        try:
            model = build_model(finetune=True)
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                use_best_model=True,
                init_model=BEST_CBM_PATH,
            )
        except Exception as e:
            print(f"-> Fine-tune thất bại ({e}). Chuyển sang train từ đầu...")
            model = train_scratch()
    else:
        print("-> Không có mô hình cũ phù hợp, huấn luyện mới hoàn toàn (Train from scratch)...")
        model = train_scratch()

    print("\n-> Hoàn tất quá trình huấn luyện!")
    print("-> Best iteration:", model.get_best_iteration())

    # ------------------- 3. ĐÁNH GIÁ -------------------
    print("\n3. Đang đánh giá mô hình trên tập Test chuẩn...")
    y_pred = model.predict(X_test).ravel().astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_m = f1_score(y_test, y_pred, average='macro', zero_division=0)

    labels = sorted(set(y_train) | set(y_val) | set(y_test) | set(y_pred))
    report = classification_report(
        y_test, y_pred, labels=labels, digits=4, zero_division=0
    )

    print("\n" + "=" * 60)
    print("KẾT QUẢ DỰ BÁO NGUY CƠ HEN SUYỄN - CATBOOST (FINE-TUNED)")
    print("=" * 60)
    print(f"Accuracy:            {acc:.4f}")
    print(f"F1-score (Weighted): {f1_w:.4f}")
    print(f"F1-score (Macro):    {f1_m:.4f}")
    print("\nClassification Report:\n")
    print(report)

    # Feature importance
    importance = (
        pd.DataFrame({
            'feature': feature_cols,
            'importance': model.get_feature_importance(),
        })
        .sort_values('importance', ascending=False)
        .reset_index(drop=True)
    )
    print("Top 10 đặc trưng quan trọng nhất:")
    print(importance.head(10).to_string(index=False))

    # ------------------- 4. LƯU KẾT QUẢ -------------------
    print("\n4. Đang xuất báo cáo và lưu mô hình...")
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    REPORT_PATH = os.path.join(OUT_DIR, f"report_catboost_ft_{ts}.txt")
    CM_PATH = os.path.join(OUT_DIR, f"cm_catboost_ft_{ts}.png")
    FI_PATH = os.path.join(OUT_DIR, f"feature_importance_{ts}.csv")
    CBM_TS_PATH = os.path.join(OUT_DIR, f"asthma_catboost_ft_{ts}.cbm")
    PKL_PATH = os.path.join(OUT_DIR, f"asthma_catboost_ft_{ts}.pkl")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=== BÁO CÁO MÔ HÌNH CATBOOST (FINE-TUNED) ===\n")
        f.write(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Fine-tune: {has_old_model}\n")
        f.write(f"Best iteration: {model.get_best_iteration()}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1-score (weighted): {f1_w:.4f}\n")
        f.write(f"F1-score (macro): {f1_m:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    importance.to_csv(FI_PATH, index=False)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - CatBoost Fine-Tuned\n{ts}')
    plt.xlabel(f'Predicted Risk Level (t+{FORECAST_HORIZON}h)')
    plt.ylabel(f'True Risk Level (t+{FORECAST_HORIZON}h)')
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=300, bbox_inches='tight')
    plt.close()

    # Luôn lưu bản có timestamp
    model.save_model(CBM_TS_PATH)
    joblib.dump(
        {
            'model': model,
            'feature_cols': feature_cols,
            'target_name': 'target',
            'forecast_horizon': FORECAST_HORIZON,
            'metrics': {'accuracy': acc, 'f1_weighted': f1_w, 'f1_macro': f1_m},
        },
        PKL_PATH,
    )

    # Chỉ cập nhật model "chính" (dùng cho lần fine-tune sau) nếu tốt hơn
    old_f1 = old_meta.get("f1_macro", -1.0)
    if (not os.path.exists(BEST_CBM_PATH)) or f1_m >= old_f1:
        if os.path.exists(BEST_CBM_PATH):
            shutil.copy(BEST_CBM_PATH, BEST_CBM_PATH + ".bak")
        model.save_model(BEST_CBM_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {'f1_macro': f1_m, 'accuracy': acc,
                 'feature_cols': feature_cols, 'timestamp': ts},
                f, ensure_ascii=False, indent=2,
            )
        best_msg = f"Đã CẬP NHẬT model chính (F1 macro {old_f1:.4f} -> {f1_m:.4f})"
    else:
        best_msg = (f"Model mới ({f1_m:.4f}) KHÔNG tốt hơn model cũ ({old_f1:.4f}), "
                    "giữ nguyên model chính")

    print("\n" + "=" * 70)
    print("HOÀN TẤT TOÀN BỘ QUY TRÌNH!")
    print("=" * 70)
    print(f"-> {best_msg}")
    print(f"-> Thư mục kết quả:        {OUT_DIR}")
    print(f"-> File native (.cbm):     {CBM_TS_PATH}")
    print(f"-> File gói Joblib (.pkl): {PKL_PATH}")
    print(f"-> Báo cáo Text:           {REPORT_PATH}")
    print(f"-> Confusion Matrix:       {CM_PATH}")
    print(f"-> Feature importance:     {FI_PATH}")


if __name__ == "__main__":
    main()
