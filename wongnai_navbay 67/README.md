# Naive Bayes วิเคราะห์รีวิว Wongnai (1–5 ดาว)

จำแนกคะแนนรีวิวร้านอาหารด้วย Multinomial Naive Bayes
Dataset: [Highgroundbkk/wongnai_reviews](https://huggingface.co/datasets/Highgroundbkk/wongnai_reviews)

## ขั้นตอน
1. โหลด dataset
2. ตัดคำ (pythainlp) + Stop words removing
3. สร้างตาราง TF
4. Naive Bayes + Laplace smoothing (alpha=1)
5. Heat map P(คำ | ดาว)
6. Word cloud แยก 1–5 ดาว
7. Performance test (Accuracy)

## วิธีรัน
```
pip install -r requirements.txt
python wongnai_nb_short.py
```
ผลลัพธ์จะถูกบันทึกในโฟลเดอร์ `outputs/`
