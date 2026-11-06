# Naive Bayes วิเคราะห์รีวิว Wongnai

ใช้ข้อความรีวิวทำนาย 1–5 ดาว โดยเขียนสูตร Multinomial Naive Bayes เอง
และใช้ PyThaiNLP ตัดคำภาษาไทย

## ไฟล์ที่ใช้

- `main.py` — โหลดข้อมูล ตัดคำ ฝึกโมเดล ประเมินผล และสร้างกราฟ ตามลำดับ
- `naive_bayes.py` — นับคำ คำนวณ Prior, Likelihood และทำนายดาว
- `requirements.txt` — ไลบรารีที่ต้องติดตั้ง
- `outputs/` — กราฟและสรุปผล

## วิธีรัน

ทดสอบด้วย Python 3.13 เปิด terminal ในโฟลเดอร์นี้:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

ถ้ามี `.venv` และติดตั้งไลบรารีแล้ว ใช้แค่คำสั่งสุดท้าย
หาก venv ไม่มี pip ใช้ `python -m pip --python .venv/Scripts/python.exe install -r requirements.txt`

ครั้งแรกดาวน์โหลดข้อมูลจาก Hugging Face แล้วเก็บใน `data/` สำหรับรันครั้งต่อไป
ใช้ฟอนต์ Tahoma บน Windows หากใช้ระบบอื่นให้แก้ `FONT_PATH` ใน `main.py` เป็นฟอนต์ไทยในเครื่อง

## ข้อมูลและสูตร

Dataset: [iamwarint/wongnai-restaurant-review](https://huggingface.co/datasets/iamwarint/wongnai-restaurant-review)
ใช้ `review_body` ทำนาย `stars` โดยใช้ train/test ที่แบ่งไว้แล้ว (19,458 / 500 รีวิว)
License ของ Dataset: LGPL-3.0

ตัดคำด้วย `newmm` กรอง stop words แต่เก็บคำปฏิเสธ เช่น “ไม่” ไว้
สร้างคลังคำจาก train เท่านั้น ใช้จำนวนคำเป็นข้อมูลเข้าโมเดล

```text
P(c)   = จำนวนรีวิวฝึกในดาว c / จำนวนรีวิวฝึกทั้งหมด
P(w|c) = (จำนวนครั้งที่คำ w ปรากฏในดาว c + 1) / (จำนวนคำทั้งหมดในดาว c + ขนาดคลังคำ)
Score  = log P(c) + ผลรวมของ [จำนวนคำในรีวิว × log P(w|c)]
```

ทายดาวที่ได้ Score สูงสุด ใช้ log เพื่อไม่ให้ผลคูณเล็กจนเป็นศูนย์
คำที่ไม่เคยพบใน train จะข้ามไป ส่วน scikit-learn ใช้เฉพาะวัดผล

## ผลลัพธ์

- [สรุปผล](outputs/summary.md): Accuracy, Macro-F1 และผลรายดาว
- `word_probability_heatmap.png`: ความน่าจะเป็นของคำแยกตามดาว
- `wordcloud_all.png` และ `wordcloud_1.png`–`wordcloud_5.png`: Word Cloud 6 ภาพ
- `confusion_matrix.png`: ดาวจริงเทียบกับดาวที่ทำนาย
- `examples.csv`: ตัวอย่างทายถูก/ผิด; `predictions.csv`: ผลทุกรีวิว (เก็บเฉพาะในเครื่อง)

ผลที่รันได้: Accuracy **25.2%**, Macro-F1 **0.1900** เทียบกับ baseline 20%
โมเดลยังทาย 1–2 ดาวได้ไม่ดี ดูรายละเอียดในสรุปผล
