# NavBay

ทำนายคะแนนร้านอาหาร 1–5 ดาวจากข้อความภาษาไทยด้วย **Multinomial Naive Bayes ที่เขียนสูตรเอง**
ใช้ PyThaiNLP ตัดคำ และใช้ scikit-learn คำนวณคะแนนประเมินผล

## วิธีรัน

Python 3.13 เป็นเวอร์ชันที่ใช้ทดสอบ เปิด terminal ในโฟลเดอร์นี้:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe run.py
```

ถ้าเครื่องสร้าง venv แล้วไม่มี pip ให้ใช้ `python -m pip --python .venv/Scripts/python.exe install -r requirements-lock.txt`
ครั้งแรกต้องใช้อินเทอร์เน็ตดาวน์โหลดข้อมูล หลังจากนั้นอ่าน parquet ที่เก็บใน `data/`
Windows ใช้ฟอนต์ Tahoma ที่ติดตั้งในเครื่อง ส่วนระบบอื่นให้ระบุฟอนต์ไทยที่มีอยู่:

```bash
python run.py --font /path/to/NotoSansThai-Regular.ttf
```

## โครงสร้าง

```text
run.py                  โหลด → ตัดคำ → ฝึก → ทำนาย → ประเมิน → สร้างกราฟ
src/data.py             โหลด official train/test และตรวจข้อมูล
src/preprocessing.py    ตัดคำ newmm และกรอง stop words
src/model.py            สูตร Naive Bayes และการนับคำเอง
src/plots.py            Heatmap และ Word Cloud
tests/test_model.py     ตรวจสูตรกับค่าคำนวณมือและกรณีขอบ
outputs/                กราฟ ตาราง และสรุปผลจากการรันจริง
data/                   ข้อมูลดาวน์โหลด (ไม่ commit)
```

## Dataset และการเตรียมข้อมูล

ใช้ [iamwarint/wongnai-restaurant-review](https://huggingface.co/datasets/iamwarint/wongnai-restaurant-review)
ตามชุด `data/train-00000-of-00001.parquet` และ `data/test-00000-of-00001.parquet` ที่เจ้าของแบ่งไว้
ใช้ `review_body` เป็น input และ `stars` เป็น label โดยไม่แบ่งชุดใหม่
License ของ Dataset: LGPL-3.0

ปรับ Unicode เป็น NFC, lowercase ภาษาอังกฤษ, ลบ URL, ตัดคำด้วย PyThaiNLP `newmm`,
กรอง stop words และ token ที่ไม่มีตัวอักษร เก็บคำปฏิเสธ เช่น “ไม่” “ไม่มี” ไว้ ไม่ทำ stemming ภาษาอังกฤษกับข้อความไทย
สร้างคลังคำและนับคำจาก train เท่านั้น ใช้ test วัดผลหลังฝึกเสร็จ
ตรวจและรายงานข้อความซ้ำใน `metrics.json` โดยไม่ได้ลบหรือย้ายรีวิวระหว่างชุด

## สูตรที่ใช้

กำหนด c เป็นคลาสดาว, V เป็นคลังคำฝึก, N_c เป็นจำนวนรีวิวฝึกในคลาส c และ N เป็นจำนวนรีวิวฝึกทั้งหมด

```text
Prior:       P(c) = N_c / N
Likelihood:  P(w|c) = (count(w,c) + 1) / (sum_v count(v,c) + |V|)
Score:       log P(c) + sum_w count(w,review) * log P(w|c)
Prediction:  คลาสที่มี Score สูงสุด
```

`count(w,c)` คือจำนวนครั้งที่คำปรากฏทั้งหมดในคลาส ไม่ใช่จำนวนรีวิวที่มีคำ
Laplace smoothing ใช้ alpha=1; รวม log เพื่อป้องกัน underflow
คำใหม่ที่ไม่มีในคลังคำฝึกจะข้ามไป ถ้าไม่มีคำที่รู้จักเลยจะทายดาวที่มี Prior สูงสุด

## ผลลัพธ์

- [สรุปผลและข้อจำกัด](outputs/summary.md)
- `metrics.json`: Accuracy, Macro-F1, ผลรายดาว และจำนวนข้อมูล
- `word_probability_heatmap.png` / `word_probabilities.csv`: **P(คำ | ดาว)** หลัง smoothing
- `wordcloud_all.png` และ `wordcloud_1.png`–`wordcloud_5.png`: ความถี่คำจาก train รวม 6 ภาพ
- `confusion_matrix.png` / `.csv`: จำนวนรีวิวจริงเทียบดาวที่ทำนายบน test
- `examples.csv`: ตัวอย่างข้อความแบบย่อที่ทายถูก/ผิดอย่างละไม่เกิน 3 รายการ
- `predictions.csv`: ผล test ทุกรายการ เก็บในเครื่อง ไม่ commit

ผลทดลองได้ Accuracy **25.2%** และ Macro-F1 **0.1900** โดยทาย 1–2 ดาวได้ไม่ดี
รายละเอียดและกราฟอยู่ในสรุปผลด้านบน

## ทดสอบสูตร

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

ตรวจสูตรเทียบกับค่าคำนวณมือ รวมถึงคำที่ไม่เคยพบ รีวิวว่าง ข้อความยาว และการเก็บคำว่า “ไม่”
