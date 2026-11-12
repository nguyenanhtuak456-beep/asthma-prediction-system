import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dự báo nguy cơ hen suyễn", page_icon="🌬️")
st.title("Hệ thống Dự báo Nguy cơ Hen suyễn 🌬️")

PRED_URL = (
    "https://raw.githubusercontent.com/"
    "nguyenanhtuak456-beep/asthma-prediction-system/main/predictions/latest_predictions.csv"
)

@st.cache_data(ttl=600)
def load_predictions():
    return pd.read_csv(PRED_URL)

try:
    preds = load_predictions()
except Exception as e:
    st.error(f"Chưa đọc được file dự đoán: {e}")
    st.stop()

city = st.selectbox("Chọn khu vực của bạn:", sorted(preds["city"].unique()))
row = preds[preds["city"] == city].iloc[0]

st.subheader(f"Kết quả cho: {city}")
st.caption(f"Số liệu lúc {row['time']} (UTC) · PM2.5 = {row['pm2_5']:.1f} · US AQI = {row['us_aqi']:.0f}")

level = int(row["predicted_class"])
if level == 0:
    st.success("Mức độ: Thấp (An toàn).")
elif level == 1:
    st.info("Mức độ: Trung bình (Theo dõi thêm).")
elif level == 2:
    st.warning("Mức độ: Cao (Hạn chế ra ngoài khi không cần thiết).")
else:
    st.error("Mức độ: Rất cao (Nguy hiểm! Cần mang theo thuốc cắt cơn).")

with st.expander("Xem bảng dự đoán toàn bộ tỉnh thành"):
    st.dataframe(preds, use_container_width=True)
