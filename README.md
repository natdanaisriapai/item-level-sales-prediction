# Item-Level Sales Prediction Project
## โปรเจคการทำนายยอดขายรายสินค้า 3 เดือนข้างหน้าสำหรับแต่ละสาขา

### 📋 วัตถุประสงค์
- ทำนายยอดขายรายสินค้าในอีก 3 เดือนข้างหน้า
- แยกการทำนายตามสาขา (store locations)
- วิเคราะห์ปัจจัยที่มีผลต่อยอดขาย
- สร้างโมเดลที่แม่นยำสำหรับการวางแผนธุรกิจ

### 🗂️ โครงสร้างโปรเจค
```
item-level sales prediction/
├── data/
│   ├── train.csv          # ข้อมูลสำหรับเทรนโมเดล
│   └── test.csv           # ข้อมูลสำหรับทดสอบ
├── 01_data_loading.py     # ขั้นตอนที่ 1: โหลดและสำรวจข้อมูล
├── 02_data_preprocessing.py # ขั้นตอนที่ 2: เตรียมข้อมูล
├── 03_data_visualization.py # ขั้นตอนที่ 3: แสดงผลและวิเคราะห์
├── 04_model_development.py  # ขั้นตอนที่ 4: สร้างและเทรนโมเดล
├── 05_prediction.py       # ขั้นตอนที่ 5: ทำนายยอดขาย
├── 06_results_visualization.py # ขั้นตอนที่ 6: แสดงผลและรายงาน
├── run_all_steps.py       # สคริปต์หลักสำหรับรันทุกขั้นตอน
├── requirements.txt       # ไลบรารีที่จำเป็น
└── README.md             # คู่มือการใช้งาน
```

### 🚀 วิธีใช้งาน

#### 1. ติดตั้งไลบรารี
```bash
pip install -r requirements.txt
```

#### 2. รันทุกขั้นตอนพร้อมกัน
```bash
python run_all_steps.py
```

#### 3. รันแต่ละขั้นตอนแยกกัน
```bash
# ขั้นตอนที่ 1: โหลดและสำรวจข้อมูล
python 01_data_loading.py

# ขั้นตอนที่ 2: เตรียมข้อมูล
python 02_data_preprocessing.py

# ขั้นตอนที่ 3: แสดงผลและวิเคราะห์
python 03_data_visualization.py

# ขั้นตอนที่ 4: สร้างและเทรนโมเดล
python 04_model_development.py

# ขั้นตอนที่ 5: ทำนายยอดขาย
python 05_prediction.py

# ขั้นตอนที่ 6: แสดงผลและรายงาน
python 06_results_visualization.py
```

### 📊 ขั้นตอนการทำงาน

#### Step 1: Data Loading และ Exploration
- โหลดข้อมูลจากไฟล์ CSV
- สำรวจโครงสร้างข้อมูล
- ตรวจสอบ missing values
- วิเคราะห์สถิติพื้นฐาน

#### Step 2: Data Preprocessing
- แปลงข้อมูลวันที่
- สร้าง features ใหม่ (ปี, เดือน, วัน, วันในสัปดาห์)
- สร้าง lag features สำหรับ time series
- จัดการ missing values

#### Step 3: Data Visualization และ Analysis
- แสดงกราฟการกระจายของยอดขาย
- วิเคราะห์ยอดขายตามเดือนและวันในสัปดาห์
- แสดง top stores และ items
- วิเคราะห์แนวโน้มยอดขาย

#### Step 4: Model Development
- เตรียม features สำหรับ machine learning
- เทรนโมเดลหลายตัว (Random Forest, Gradient Boosting, Linear Regression)
- ประเมินประสิทธิภาพโมเดล
- เลือกโมเดลที่ดีที่สุด

#### Step 5: Prediction และ Forecasting
- สร้างข้อมูลสำหรับทำนาย 3 เดือนข้างหน้า
- ใช้โมเดลที่เทรนแล้วทำนายยอดขาย
- บันทึกผลการทำนาย

#### Step 6: Results Visualization และ Reporting
- แสดงผลการทำนายในรูปแบบกราฟ
- สร้างรายงานสรุป
- วิเคราะห์ผลลัพธ์ระดับสาขาและสินค้า

### 📁 ไฟล์ผลลัพธ์

หลังจากรันเสร็จแล้ว จะได้ไฟล์ดังนี้:

- `processed_train_data.csv` - ข้อมูลที่ผ่านการเตรียมแล้ว
- `processed_test_data.csv` - ข้อมูลทดสอบที่ผ่านการเตรียมแล้ว
- `data_visualization.png` - กราฟการวิเคราะห์ข้อมูล
- `best_sales_model.pkl` - โมเดลที่เทรนแล้ว
- `label_encoders.pkl` - encoders สำหรับ categorical variables
- `feature_importance.csv` - ความสำคัญของ features
- `sales_predictions_3months.csv` - ผลการทำนาย 3 เดือนข้างหน้า
- `prediction_visualization.png` - กราฟผลการทำนาย
- `prediction_summary_stats.csv` - สถิติสรุปผลการทำนาย

### 🔧 การปรับแต่ง

#### ปรับแต่งโมเดล
แก้ไขไฟล์ `04_model_development.py`:
```python
# เปลี่ยน hyperparameters
models = {
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
    'Linear Regression': LinearRegression()
}
```

#### ปรับแต่งการทำนาย
แก้ไขไฟล์ `05_prediction.py`:
```python
# เปลี่ยนระยะเวลาการทำนาย (วัน)
future_dates = generate_future_dates(last_date, periods=180)  # 6 เดือน
```

### 📈 การประเมินผล

#### Metrics ที่ใช้
- **RMSE (Root Mean Square Error)** - วัดความคลาดเคลื่อนเฉลี่ย
- **MAE (Mean Absolute Error)** - วัดความคลาดเคลื่อนสัมบูรณ์
- **R² Score** - วัดความแม่นยำของโมเดล

#### การแปลผล
- RMSE ต่ำ = โมเดลแม่นยำ
- R² สูง (ใกล้ 1) = โมเดลอธิบายข้อมูลได้ดี
- MAE ต่ำ = ความคลาดเคลื่อนน้อย

### 🛠️ การแก้ไขปัญหา

#### ปัญหาที่พบบ่อย

1. **ไฟล์ข้อมูลไม่พบ**
   ```
   FileNotFoundError: [Errno 2] No such file or directory: 'data/train.csv'
   ```
   **วิธีแก้**: ตรวจสอบว่ามีไฟล์ `train.csv` และ `test.csv` ในโฟลเดอร์ `data/`

2. **ไลบรารีไม่พบ**
   ```
   ModuleNotFoundError: No module named 'sklearn'
   ```
   **วิธีแก้**: รัน `pip install -r requirements.txt`

3. **Memory Error**
   ```
   MemoryError
   ```
   **วิธีแก้**: ลดขนาดข้อมูลหรือใช้ subset ของข้อมูล

### 📞 การติดต่อ

หากมีปัญหาหรือคำถาม สามารถติดต่อได้ที่:
- Email: [your-email@example.com]
- GitHub: [your-github-username]

### 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**หมายเหตุ**: โปรเจคนี้ถูกออกแบบให้ใช้งานง่ายและเข้าใจได้ โดยแต่ละขั้นตอนจะมีการแสดงผลและบันทึกไฟล์ผลลัพธ์เพื่อให้สามารถติดตามความคืบหน้าได้
