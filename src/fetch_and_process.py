import os
import time
import requests
import pandas as pd
from datetime import datetime

# 1. Cấu hình thư mục lưu trữ cho GitHub Actions
DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

# 2. Danh sách toàn bộ 63 tỉnh thành phố trực thuộc trung ương
CITIES_63 = {
    # --- MIỀN BẮC ---
    "hanoi": {"name": "Hà Nội", "region": "Mien_Bac", "lat": 21.0285, "lon": 105.8542},
    "haiphong": {"name": "Hải Phòng", "region": "Mien_Bac", "lat": 20.8449, "lon": 106.6881},
    "quangninh": {"name": "Quảng Ninh", "region": "Mien_Bac", "lat": 21.0064, "lon": 107.2925},
    "bacninh": {"name": "Bắc Ninh", "region": "Mien_Bac", "lat": 21.1861, "lon": 106.0763},
    "haiduong": {"name": "Hải Dương", "region": "Mien_Bac", "lat": 20.9389, "lon": 106.3315},
    "hungyen": {"name": "Hưng Yên", "region": "Mien_Bac", "lat": 20.6463, "lon": 106.0511},
    "hanam": {"name": "Hà Nam", "region": "Mien_Bac", "lat": 20.5846, "lon": 105.9124},
    "namdinh": {"name": "Nam Định", "region": "Mien_Bac", "lat": 20.4354, "lon": 106.1793},
    "thaibinh": {"name": "Thái Bình", "region": "Mien_Bac", "lat": 20.4463, "lon": 106.3366},
    "ninhbinh": {"name": "Ninh Bình", "region": "Mien_Bac", "lat": 20.2506, "lon": 105.9744},
    "vinhphuc": {"name": "Vĩnh Phúc", "region": "Mien_Bac", "lat": 21.3089, "lon": 105.6044},
    "thainguyen": {"name": "Thái Nguyên", "region": "Mien_Bac", "lat": 21.5942, "lon": 105.8481},
    "phutho": {"name": "Phú Thọ", "region": "Mien_Bac", "lat": 21.3257, "lon": 105.2307},
    "bacgiang": {"name": "Bắc Giang", "region": "Mien_Bac", "lat": 21.2731, "lon": 106.1947},
    "tuyenquang": {"name": "Tuyên Quang", "region": "Mien_Bac", "lat": 21.8229, "lon": 105.2144},
    "laocai": {"name": "Lào Cai", "region": "Mien_Bac", "lat": 22.4856, "lon": 103.9704},
    "yenbai": {"name": "Yên Bái", "region": "Mien_Bac", "lat": 21.7167, "lon": 104.8667},
    "dienbien": {"name": "Điện Biên", "region": "Mien_Bac", "lat": 21.3833, "lon": 103.0167},
    "laichau": {"name": "Lai Châu", "region": "Mien_Bac", "lat": 22.3967, "lon": 103.4731},
    "sonla": {"name": "Sơn La", "region": "Mien_Bac", "lat": 21.3283, "lon": 103.8981},
    "hoabinh": {"name": "Hòa Bình", "region": "Mien_Bac", "lat": 20.8146, "lon": 105.3377},
    "hagiang": {"name": "Hà Giang", "region": "Mien_Bac", "lat": 22.8233, "lon": 104.9839},
    "backan": {"name": "Bắc Kạn", "region": "Mien_Bac", "lat": 22.1471, "lon": 105.8348},
    "caobang": {"name": "Cao Bằng", "region": "Mien_Bac", "lat": 22.6667, "lon": 106.2500},
    "langson": {"name": "Lạng Sơn", "region": "Mien_Bac", "lat": 21.8523, "lon": 106.7581},

    # --- MIỀN TRUNG & TÂY NGUYÊN ---
    "thanhhoa": {"name": "Thanh Hóa", "region": "Mien_Trung", "lat": 19.8067, "lon": 105.7851},
    "nghean": {"name": "Nghệ An", "region": "Mien_Trung", "lat": 18.6796, "lon": 105.6813},
    "hatinh": {"name": "Hà Tĩnh", "region": "Mien_Trung", "lat": 18.3424, "lon": 105.9056},
    "quangbinh": {"name": "Quảng Bình", "region": "Mien_Trung", "lat": 17.4700, "lon": 106.6000},
    "quangtri": {"name": "Quảng Trị", "region": "Mien_Trung", "lat": 16.8167, "lon": 107.1000},
    "hue": {"name": "Thừa Thiên Huế", "region": "Mien_Trung", "lat": 16.4637, "lon": 107.5909},
    "danang": {"name": "Đà Nẵng", "region": "Mien_Trung", "lat": 16.0544, "lon": 108.2022},
    "quangnam": {"name": "Quảng Nam", "region": "Mien_Trung", "lat": 15.5760, "lon": 108.4714},
    "quangngai": {"name": "Quảng Ngãi", "region": "Mien_Trung", "lat": 15.1214, "lon": 108.7923},
    "binhdinh": {"name": "Bình Định", "region": "Mien_Trung", "lat": 13.7820, "lon": 109.2197},
    "phuyen": {"name": "Phú Yên", "region": "Mien_Trung", "lat": 13.0883, "lon": 109.2929},
    "khanhhoa": {"name": "Khánh Hòa", "region": "Mien_Trung", "lat": 12.2388, "lon": 109.1967},
    "ninhthuan": {"name": "Ninh Thuận", "region": "Mien_Trung", "lat": 11.5654, "lon": 108.9882},
    "binhthuan": {"name": "Bình Thuận", "region": "Mien_Trung", "lat": 10.9298, "lon": 108.1021},
    "kontum": {"name": "Kon Tum", "region": "Mien_Trung", "lat": 14.3545, "lon": 108.0079},
    "gialai": {"name": "Gia Lai", "region": "Mien_Trung", "lat": 13.9785, "lon": 108.0056},
    "daklak": {"name": "Đắk Lắk", "region": "Mien_Trung", "lat": 12.6667, "lon": 108.0383},
    "daknong": {"name": "Đắk Nông", "region": "Mien_Trung", "lat": 11.9904, "lon": 107.6836},
    "lamdong": {"name": "Lâm Đồng", "region": "Mien_Trung", "lat": 11.9404, "lon": 108.4383},

    # --- MIỀN NAM ---
    "hcm": {"name": "TP. Hồ Chí Minh", "region": "Mien_Nam", "lat": 10.8231, "lon": 106.6297},
    "binhduong": {"name": "Bình Dương", "region": "Mien_Nam", "lat": 11.0092, "lon": 106.6573},
    "dongnai": {"name": "Đồng Nai", "region": "Mien_Nam", "lat": 10.9574, "lon": 106.8427},
    "bariavungtau": {"name": "Bà Rịa - Vũng Tàu", "region": "Mien_Nam", "lat": 10.4114, "lon": 107.1362},
    "binhphuoc": {"name": "Bình Phước", "region": "Mien_Nam", "lat": 11.7544, "lon": 106.9080},
    "tayninh": {"name": "Tây Ninh", "region": "Mien_Nam", "lat": 11.3100, "lon": 106.1000},
    "longan": {"name": "Long An", "region": "Mien_Nam", "lat": 10.5333, "lon": 106.4167},
    "tiengiang": {"name": "Tiền Giang", "region": "Mien_Nam", "lat": 10.4500, "lon": 106.3500},
    "bentre": {"name": "Bến Tre", "region": "Mien_Nam", "lat": 10.2434, "lon": 106.3756},
    "travinh": {"name": "Trà Vinh", "region": "Mien_Nam", "lat": 9.9347, "lon": 106.3453},
    "vinhlong": {"name": "Vĩnh Long", "region": "Mien_Nam", "lat": 10.2542, "lon": 105.9722},
    "dongthap": {"name": "Đồng Tháp", "region": "Mien_Nam", "lat": 10.4628, "lon": 105.6802},
    "angiang": {"name": "An Giang", "region": "Mien_Nam", "lat": 10.5250, "lon": 105.1250},
    "kiengiang": {"name": "Kiên Giang", "region": "Mien_Nam", "lat": 10.0121, "lon": 105.0809},
    "cantho": {"name": "Cần Thơ", "region": "Mien_Nam", "lat": 10.0452, "lon": 105.7469},
    "haugiang": {"name": "Hậu Giang", "region": "Mien_Nam", "lat": 9.7833, "lon": 105.4667},
    "soctrang": {"name": "Sóc Trăng", "region": "Mien_Nam", "lat": 9.5954, "lon": 105.9744},
    "baclieu": {"name": "Bạc Liêu", "region": "Mien_Nam", "lat": 9.2940, "lon": 105.7230},
    "camau": {"name": "Cà Mau", "region": "Mien_Nam", "lat": 9.1769, "lon": 105.1524}
}

START_DATE = "2024-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# 3. Hàm gán nhãn nguy cơ hen suyễn chuẩn hóa y tế
def calculate_asthma_risk(row):
    us_aqi = row.get('us_aqi')
    pm2_5 = row.get('pm2_5')
    if pd.notnull(us_aqi):
        if us_aqi <= 50: return 0
        elif us_aqi <= 100: return 1
        elif us_aqi <= 150: return 2
        else: return 3
    if pd.notnull(pm2_5):
        if pm2_5 <= 12.0: return 0
        elif pm2_5 <= 35.4: return 1
        elif pm2_5 <= 55.4: return 2
        else: return 3
    return 0

# 4. Tiến hành cào dữ liệu toàn quốc 63 tỉnh thành với cơ chế chống nghẽn API (429)
all_dfs = []
print(f"=== BẮT ĐẦU CÀO DỮ LIỆU 63 TỈNH THÀNH (TỪ {START_DATE} ĐẾN {END_DATE}) ===")

for key, city in CITIES_63.items():
    print(f"---> Đang cào: {city['name']} ({city['region']})...")
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={city['lat']}&longitude={city['lon']}&"
        f"hourly=pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,european_aqi,us_aqi&"
        f"start_date={START_DATE}&end_date={END_DATE}"
    )

    success = False
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=20)
            if response.status_code == 200 and "hourly" in response.json():
                df = pd.DataFrame(response.json()["hourly"])
                df['city'] = city['name']
                df['region'] = city['region']
                df['asthma_risk_level'] = df.apply(calculate_asthma_risk, axis=1)
                
                all_dfs.append(df)
                success = True
                break
            elif response.status_code == 429:
                print(f"⚠️ Bị nghẽn API tại {city['name']}, chờ 10 giây thử lại...")
                time.sleep(10)
            else:
                break
        except Exception as e:
            time.sleep(4)
            
    # Nghỉ nhẹ giữa các request để bảo vệ IP không bị chặn
    time.sleep(2.5)

# 5. Xử lý tính toán đặc trưng thời gian (Lag, Rolling, Diff) trên quy mô lớn
if all_dfs:
    print("\n⏳ Đang gộp dữ liệu và tính toán các đặc trưng thời gian (Lag, Rolling, Diff)...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    if 'time' in combined_df.columns:
        combined_df['time'] = pd.to_datetime(combined_df['time'])
        combined_df = combined_df.sort_values(['city', 'time'])

    # Các cột cần tính toán đặc trưng chuỗi thời gian
    features_to_process = ['pm2_5', 'us_aqi', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide']
    for col in features_to_process:
        if col in combined_df.columns:
            # Lag features
            combined_df[f'{col}_lag1'] = combined_df.groupby('city')[col].shift(1)
            combined_df[f'{col}_lag24'] = combined_df.groupby('city')[col].shift(24)
            
            # Rolling features
            combined_df[f'{col}_rolling_mean_3'] = combined_df.groupby('city')[col].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
            combined_df[f'{col}_rolling_mean_24'] = combined_df.groupby('city')[col].rolling(window=24, min_periods=1).mean().reset_index(0, drop=True)
            
            # Diff features
            combined_df[f'{col}_diff_1'] = combined_df.groupby('city')[col].diff(1)

    # 6. Lưu ra file live_processed.csv theo đúng yêu cầu cấu trúc dự án
    output_file_path = os.path.join(DATASET_DIR, "live_processed.csv")
    combined_df.to_csv(output_file_path, index=False, encoding="utf-8-sig")
    
    print("\n" + "="*70)
    print("=== HOÀN TẤT XỬ LÝ TOÀN BỘ 63 TỈNH THÀNH ===")
    print("="*70)
    print(f"Tổng số bản ghi thu được: {len(combined_df):,} dòng.")
    print(f"Đã xuất file kết quả thành công tại: {output_file_path}")
