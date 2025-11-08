import pandas as pd
import numpy as np
# (เราไม่ต้องการ cross_val_score หรือ KFold ที่นี่แล้ว)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib # <-- สำคัญมากสำหรับ Save

print("1. กำลังโหลด Dataset.csv (ที่แก้ไขแล้ว)...")
df = pd.read_csv("Dataset.csv")

# --- 2. การทำ ENCODING ---
print("2. กำลังทำ Label Encoding...")
cat_cols = ['vehicle_type', 'fuel_type']
encoders = {} 
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
print("   - Encoders ที่สร้าง:", list(encoders.keys()))

# --- 3. Feature Engineering (The Robust Fix) ---
print("3. กำลังทำ Feature Engineering...")
df['relative_load'] = df['load_weight_kg'] / df['vehicle_capacity_kg']

# --- 4. (✅ เกราะที่ 1) บีบอัดข้อมูล (Log Transform) ---
print("4. กำลังบีบอัดข้อมูล (Log Transform)...")
df['distance_km'] = np.log1p(df['distance_km'])
df['load_weight_kg'] = np.log1p(df['load_weight_kg'])

# --- 5. กำหนด Features (X) และ Target (y) ---
feature_cols = [
    'distance_km',       # <-- นี่คือค่า Log
    'load_weight_kg',    # <-- นี่คือค่า Log
    'vehicle_type',      
    'fuel_type',         
    'relative_load'
]
target_col = 'carbon_kg'

X = df[feature_cols] # <-- ใช้ข้อมูลทั้งหมด (500 แถว)
y = df[target_col]   # <-- ใช้ข้อมูลทั้งหมด (500 แถว)

# --- 6. (The Robust Fix) สร้างโมเดล ---
print("5. กำลังสร้างโมเดล XGBoost (แบบทนทาน)...")
model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    
    # --- (✅ เกราะที่ 2) เปลี่ยนเป้าหมาย ---
    objective='reg:absoluteerror',  
    
    random_state=42 
)

# --- 7. (ใหม่) เทรนด้วย "ข้อมูลทั้งหมด" ---
# (เราไม่แบ่ง Test แล้ว เพราะเรา "เชื่อมั่น" ใน MAE 4.854 แล้ว)
# (เราจะเทรนโมเดล "เวอร์ชันดีที่สุด" ด้วยข้อมูล 100%)
print("6. กำลังเทรนโมเดล (ด้วยข้อมูล 100%)...")
model.fit(X, y) # <-- เทรนด้วย X, y (ทั้งหมด)
print("   - เทรนโมเดล (Final) เสร็จสิ้น!")

# --- 8. บันทึก (Final) ---
print("7. กำลังบันทึกโมเดล (Final) และ Encoders...")
joblib.dump(model, "xgboost_carbon_model.pkl")
joblib.dump(encoders, "label_encoders.pkl")

print("   - บันทึก 'xgboost_carbon_model.pkl' (Final) สำเร็จ")
print("   - บันทึก 'label_encoders.pkl' สำเร็จ")