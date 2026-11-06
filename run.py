"""Run from any directory: python run.py [--font /path/to/ThaiFont.ttf]."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.data import load_splits
from src.model import NaiveBayes
from src.plots import make_plots, thai_font
from src.preprocessing import tokenize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", help="Path to a Thai-capable TTF/OTF font")
    args = parser.parse_args()
    font = thai_font(args.font)
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    print("Loading official train/test splits...", flush=True)
    frames, audit = load_splits(ROOT / "data")
    documents = {}
    for split, frame in frames.items():
        print(f"Tokenizing {split}: {len(frame):,} reviews...", flush=True)
        documents[split] = [tokenize(text) for text in frame.review_body]
        audit[split]["empty_after_tokenization"] = sum(not doc for doc in documents[split])
    model = NaiveBayes(alpha=1).fit(documents["train"], frames["train"].stars.to_numpy())
    if list(model.classes) != [1, 2, 3, 4, 5]:
        raise ValueError("Expected all five classes in training")
    actual = frames["test"].stars.to_numpy()
    predicted = model.predict(documents["test"])
    matrix = confusion_matrix(actual, predicted, labels=model.classes)
    majority = int(frames["train"].stars.mode().iloc[0])
    metrics = {
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(actual, predicted, labels=model.classes, average="macro", zero_division=0),
        "majority_baseline_accuracy": float(np.mean(actual == majority)),
        "majority_class": majority,
        "vocabulary_size": len(model.words),
        "alpha": model.alpha,
        "classification_report": classification_report(actual, predicted, labels=model.classes,
                                                        output_dict=True, zero_division=0),
        "data_audit": audit,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(matrix, index=model.classes, columns=model.classes).to_csv(output / "confusion_matrix.csv")
    predictions = frames["test"].copy()
    predictions["predicted_stars"] = predicted
    predictions["correct"] = actual == predicted
    predictions.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    # Short excerpts for a compact reviewable report; full predictions stay local.
    examples = pd.concat([predictions[predictions.correct].head(3), predictions[~predictions.correct].head(3)]).copy()
    examples["review_body"] = examples.review_body.str.replace(r"\s+", " ", regex=True).str.slice(0, 180)
    examples.to_csv(output / "examples.csv", index=False, encoding="utf-8-sig")
    print("Rendering heatmaps and six word clouds...", flush=True)
    make_plots(model, matrix, output, font)
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    source, target = np.unravel_index(off_diagonal.argmax(), matrix.shape)
    class_rows = "\n".join(
        f"| {star} | {audit['train']['stars'][star]} | {audit['test']['stars'][star]} |"
        for star in model.classes
    )
    report = f"""# ผลการทดลอง

ใช้ชุด train {len(frames['train']):,} รีวิว และ test {len(actual):,} รีวิวตามที่ Dataset แบ่งไว้
หลังตัดคำได้คลังคำ {len(model.words):,} คำ ฝึกด้วย Multinomial Naive Bayes และ Laplace smoothing = 1

| ดาว | Train | Test |
| --- | ---: | ---: |
{class_rows}

พบข้อความซ้ำตรงกันระหว่าง train/test {audit['exact_text_overlap_unique']} ข้อความ

## ผลทำนาย

- Accuracy: **{metrics['accuracy']:.2%}** (ถูก {int((actual == predicted).sum())} จาก {len(actual)} รีวิว)
- Macro-F1: **{metrics['macro_f1']:.4f}**
- ถ้าทาย {majority} ดาวทุกรีวิว จะได้ Accuracy {metrics['majority_baseline_accuracy']:.2%}

คู่ที่ทายผิดมากที่สุดคือ {model.classes[source]} ดาวเป็น {model.classes[target]} ดาว จำนวน {matrix[source, target]} รีวิว
ส่วน 1 และ 2 ดาวมี Recall เพียง {metrics['classification_report']['1']['recall']:.2%} และ {metrics['classification_report']['2']['recall']:.2%}
ดูตัวอย่างทายถูกและผิดได้ใน [examples.csv](examples.csv)

ผลยังไม่ดี โดยเฉพาะดาวน้อย ซึ่งมีตัวอย่างฝึกน้อยกว่าดาวอื่นมาก
อีกข้อจำกัดคือโมเดลนับคำแยกกัน จึงไม่เข้าใจวลีอย่าง “ไม่อร่อย” แม้จะเก็บคำว่า “ไม่” ไว้แล้ว

## กราฟ

![ความน่าจะเป็นของคำในแต่ละดาว](word_probability_heatmap.png)

Heatmap แสดง P(คำ | ดาว) จากชุดฝึก เลือก 8 คำที่มีค่าสูงสุดของแต่ละดาวแล้วรวมโดยไม่ซ้ำ
คำที่พบบ่อยอาจพบในหลายดาว จึงไม่จำเป็นต้องเป็นคำที่ใช้แยกดาวได้ดี

![ผลทำนายเทียบกับดาวจริง](confusion_matrix.png)

Word Cloud มีภาพรวม `wordcloud_all.png` และแยกดาว `wordcloud_1.png` ถึง `wordcloud_5.png`
ขนาดคำมาจากความถี่ในชุดฝึก แต่ละภาพปรับขนาดแยกกัน
"""
    (output / "summary.md").write_text(report, encoding="utf-8")
    print(f"Accuracy={metrics['accuracy']:.4f}; Macro-F1={metrics['macro_f1']:.4f}")
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
