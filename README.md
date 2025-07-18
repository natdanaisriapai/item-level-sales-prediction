# test1.ipynb - Item-Level Sales Prediction Notebook
## โน้ตบุ๊คสำหรับการทำนายยอดขายรายสินค้าและการวิเคราะห์ผล

### 📋 วัตถุประสงค์
- ทดลองใช้โมเดลหลายตัวในการทำนายยอดขาย (LightGBM, Random Forest, Gradient Boosting)
- สร้าง Ensemble Model จากโมเดลที่ดีที่สุด
- วิเคราะห์และเปรียบเทียบประสิทธิภาพของแต่ละโมเดล
- แสดงผลการทำนายในรูปแบบกราฟที่เข้าใจง่าย

### 🔄 ขั้นตอนการทำงาน
1. **Data Loading & Preprocessing**
   - โหลดข้อมูลจาก train.csv และ test.csv
   - แปลงข้อมูลวันที่เป็น datetime
   - สร้าง features ใหม่ (ปี, เดือน, วัน, วันในสัปดาห์)
   - สร้าง lag features (7, 14, 30 วัน)
   - สร้าง rolling mean features

2. **Model Training**
   - แบ่งข้อมูลเป็น train, validation, test (45,000 records for validation and test)
   - เทรนโมเดล 3 ตัว:
     * LightGBM
     * Random Forest
     * Gradient Boosting
   - สร้าง Ensemble Model โดยใช้ weighted average ตาม R² score

3. **Prediction Generation**
   - ทำนายยอดขาย 3 เดือนข้างหน้า
   - คำนวณ lag features แบบ rolling
   - ใช้ทั้ง individual models และ ensemble model

4. **Visualization**
   - แสดงกราฟเปรียบเทียบ Actual vs Predicted Sales
   - แยกแสดงผลการทำนายของแต่ละโมเดล
   - วิเคราะห์ top 3 store-item combinations

### 📊 การแสดงผล
- **กราฟหลัก**: แสดงยอดขายรวมทุกสาขาและสินค้า
  * เส้นสีน้ำเงิน: ยอดขายจริง
  * เส้นประสีแดง: Ensemble Prediction
  * เส้นจุดสีเขียว: LightGBM Prediction
  * เส้นจุดสีส้ม: Random Forest Prediction
  * เส้นจุดสีม่วง: Gradient Boosting Prediction

- **กราฟรายสินค้า**: แสดงยอดขายแยกตาม store-item combinations ที่มียอดขายสูงสุด 3 อันดับแรก

### 📈 Model Weights
- แสดงน้ำหนักของแต่ละโมเดลใน Ensemble
- คำนวณจาก R² score ของแต่ละโมเดล
- แสดงเป็นเปอร์เซ็นต์

### 🔧 การใช้งาน
1. ตรวจสอบว่ามีไฟล์ข้อมูลครบ:
   - train.csv
   - test.csv
   - best_model_LightGBM.joblib
   - feature_scaler.joblib

2. ติดตั้ง dependencies:
   ```bash
   pip install -r requirement.txt
   ```

3. รัน notebook ทั้งหมดเพื่อดูผลการวิเคราะห์

### 📝 หมายเหตุ
- การทำนายใช้ข้อมูล lag 30 วันล่าสุด
- ถ้าไม่มีข้อมูล lag จะใช้ค่าเฉลี่ยของ store-item combination นั้นๆ แทน
- สามารถปรับ hyperparameters ของแต่ละโมเดลได้ตามต้องการ
- กราฟสามารถ zoom และ pan เพื่อดูรายละเอียดได้

### ⚠️ ข้อควรระวัง
- ต้องมี RAM เพียงพอสำหรับการโหลดและประมวลผลข้อมูล
- การรันอาจใช้เวลานานขึ้นอยู่กับขนาดของข้อมูลและสเปคเครื่อง
- ควรตรวจสอบ lag features ว่ามีข้อมูลเพียงพอก่อนการทำนาย
