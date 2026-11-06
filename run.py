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
    novel = ~frames["test"].review_body.isin(set(frames["train"].review_body)).to_numpy()
    metrics["test_without_exact_train_overlap"] = {
        "rows": int(novel.sum()),
        "accuracy": accuracy_score(actual[novel], predicted[novel]) if novel.any() else None,
    }
    total_tokens = sum(len(doc) for doc in documents["test"])
    unknown = sum(word not in model.vocabulary for doc in documents["test"] for word in doc)
    metrics["test_unknown_token_fraction"] = unknown / total_tokens if total_tokens else 0
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
    top_words = []
    for label, row in zip(model.classes, model.log_likelihood):
        top_words.append(f"- {label} ดาว: " + ", ".join(model.words[i] for i in np.argsort(-row)[:8]))
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    source, target = np.unravel_index(off_diagonal.argmax(), matrix.shape)
    report = f"""# ผลการทดลอง Wongnai Naive Bayes

- Train: {len(frames['train']):,} รีวิว / Test: {len(actual):,} รีวิว (ชุดแบ่งเดิม)
- จำนวนแต่ละดาวใน train: {audit['train']['stars']}
- จำนวนแต่ละดาวใน test: {audit['test']['stars']}
- Vocabulary: {len(model.words):,} คำ; Laplace alpha = 1
- Test Accuracy: {metrics['accuracy']:.2%}
- Macro-F1: {metrics['macro_f1']:.4f}
- Baseline ทายดาว {majority} ทุกรีวิว: {metrics['majority_baseline_accuracy']:.2%}
- ข้อความซ้ำตรงกันระหว่าง train/test: {audit['exact_text_overlap_unique']} ข้อความไม่ซ้ำกัน
- ผลบน test ที่ไม่ซ้ำกับ train: {metrics['test_without_exact_train_overlap']}
- สัดส่วน token ใน test ที่ไม่มีในคลังคำ: {metrics['test_unknown_token_fraction']:.2%}

## คำที่มี P(word | stars) สูงสุดจากชุดฝึก

{chr(10).join(top_words)}

คำเหล่านี้พบบ่อยในคลาสนั้น ไม่จำเป็นต้องเป็นคำที่แยกคลาสได้ดีที่สุด
Heatmap ใช้ความน่าจะเป็นจริงและสเกลสีเดียวกันทุกดาว ไม่มีการ normalize ใหม่เฉพาะคำที่เลือก
Word Cloud ใช้จำนวนคำดิบจากชุดฝึก ขนาดคำเปรียบเทียบภายในภาพเป็นหลัก

## ข้อผิดพลาดและข้อจำกัด

คู่ที่ทายผิดมากที่สุด: จริง {model.classes[source]} ดาว → ทาย {model.classes[target]} ดาว ({matrix[source, target]} รีวิว)
ดูรายคลาสใน metrics.json และตัวอย่างถูก/ผิดใน examples.csv
โมเดลนับคำโดยไม่รู้ลำดับคำ จึงยังตีความคำปฏิเสธ ประโยคประชด และรีวิวที่ชมปนติได้จำกัด
เก็บคำปฏิเสธไว้แต่ไม่ได้แปลว่าโมเดลเข้าใจวลี เช่น “ไม่อร่อย” ทั้งวลี
ข้อมูลแต่ละดาวมีจำนวนไม่เท่ากัน จึงควรพิจารณา Macro-F1 และ baseline ร่วมกับ Accuracy
ผลครั้งนี้ทายถูก {int((actual == predicted).sum())}/{len(actual)} รีวิว เทียบ baseline {int((actual == majority).sum())}/{len(actual)} รีวิว
Recall ของ 1 ดาว = {metrics['classification_report']['1']['recall']:.2%} และ 2 ดาว = {metrics['classification_report']['2']['recall']:.2%}
ผลยังต่ำสำหรับการนำไปใช้งานจริง โดยเฉพาะคลาสดาวน้อย; ความไม่สมดุลของ train และข้อจำกัดของ Bag of Words เป็นประเด็นที่ควรศึกษาต่อ
คงข้อมูลและชุดแบ่งต้นฉบับ รวมถึงข้อความซ้ำ เพื่อไม่เปลี่ยนโจทย์; ผลแยกข้างต้นตัดเฉพาะ exact overlap ตอนประเมิน
ไม่ได้ปรับพารามิเตอร์จากผล test และไม่ได้อ้างว่าเป็นโมเดลที่ดีที่สุด
"""
    (output / "summary.md").write_text(report, encoding="utf-8")
    print(f"Accuracy={metrics['accuracy']:.4f}; Macro-F1={metrics['macro_f1']:.4f}")
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
