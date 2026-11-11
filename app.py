import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
from huggingface_hub import hf_hub_download

st.title("Hệ thống Dự báo sớm Nguy cơ Hen suyễn (t+3h) 🌬️")

# 1. Tải mô hình mới nhất trực tiếp từ Hugging Face Hub về web
@st.cache_resource
def load_latest_model():
    model_path = hf_hub_download(repo_id="username_cua_ban/asthma-catboost-weights", filename="asthma_model.cbm")
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model

model = load_latest_model()

# 2. Giao diện cho người dùng chọn khu vực
city = st.selectbox("Chọn khu vực của bạn:", ["Đà Nẵng", "Hà Nội", "TP. Hồ Chí Minh", "Gia Lai (Pleiku)"])

if st.button("Dự báo nguy cơ 3 giờ tới"):
    # Đọc dữ liệu thời tiết live mới nhất từ GitHub Raw URL
    csv_url = "https://raw.githubusercontent.com/username_cua_ban/ten_repo/main/live_processed.csv"
    live_df = pd.read_csv(csv_url)
    
    city_data = live_df[live_df['city'] == city]
    
    if not city_data.empty:
        # Lọc các cột đặc trưng và dự báo
        exclude_cols = ['time', 'region', 'city']
        features = city_data[[col for col in city_data.columns if col not in exclude_cols]]
        
        prediction = int(model.predict(features)[0])
        
        # Hiển thị kết quả trực quan
        st.subheader(f"Kết quả dự báo cho: {city}")
        if prediction == 0:
            st.success("Mức độ: Lớp 0 - Thấp (An toàn).")
        elif prediction == 1:
            st.info("Mức độ: Lớp 1 - Trung bình (Theo dõi thêm).")
        elif prediction == 2:
            st.warning("Mức độ: Lớp 2 - Cao (Cần chú ý, hạn chế ra ngoài khi không cần thiết).")
        else:
            st.error("Mức độ: Lớp 3 - Rất cao (Nguy hiểm! Cần mang theo thuốc cắt cơn).")
    else:
        st.error("Hiện chưa có dữ liệu thời tiết thời gian thực cho khu vực này.")
