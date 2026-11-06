"""รันงานทั้งหมดด้วย python main.py"""
import os
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# เก็บ cache ของไลบรารีไว้ในโปรเจกต์
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("PYTHAINLP_DATA", str(ROOT / ".cache/pythainlp"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from wordcloud import WordCloud
from naive_bayes import NaiveBayes

FONT_PATH = "C:/Windows/Fonts/tahoma.ttf"  # เปลี่ยนเป็นฟอนต์ไทยในเครื่องหากไม่ได้ใช้ Windows
STOPWORDS = set(thai_stopwords()) - {"ไม่", "ไม่ได้", "ไม่มี", "ไม่ใช่", "มิ", "ห้าม", "อย่า", "ดี", "มาก", "น้อย"}


def tokenize(text):
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = re.sub(r"https?://\S+", " ", text)
    return [word.strip() for word in word_tokenize(text, engine="newmm", keep_whitespace=False)
            if word.strip() not in STOPWORDS and any(char.isalpha() for char in word)]


def main():
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    font_path = FONT_PATH
    if not Path(font_path).is_file():
        raise FileNotFoundError("แก้ FONT_PATH ใน main.py ให้เป็นที่อยู่ฟอนต์ไทยในเครื่อง")

    # 1. โหลดชุด train/test เดิม และเก็บไว้เพื่อไม่ต้องดาวน์โหลดซ้ำ
    frames = {}
    for split in ["train", "test"]:
        path = data / f"{split}.parquet"
        if not path.exists():
            url = f"hf://datasets/iamwarint/wongnai-restaurant-review/data/{split}-00000-of-00001.parquet"
            pd.read_parquet(url).to_parquet(path, index=False)
        frames[split] = pd.read_parquet(path)
    train, test = frames["train"], frames["test"]
    print(f"Train: {len(train):,}, Test: {len(test):,}", flush=True)
    print(pd.DataFrame({"train": train.stars.value_counts(), "test": test.stars.value_counts()}).sort_index())

    # 2. ตัดคำและฝึก Naive Bayes (สร้างคลังคำจาก train เท่านั้น)
    print("Tokenizing reviews...", flush=True)
    train_tokens = [tokenize(text) for text in train.review_body]
    test_tokens = [tokenize(text) for text in test.review_body]
    model = NaiveBayes(alpha=1).fit(train_tokens, train.stars.to_numpy())

    # 3. ทำนายและวัดผลบน test
    actual = test.stars.to_numpy()
    predicted = model.predict(test_tokens)
    accuracy = accuracy_score(actual, predicted)
    macro_f1 = f1_score(actual, predicted, average="macro", zero_division=0)
    matrix = confusion_matrix(actual, predicted, labels=model.classes)
    majority = train.stars.mode().iloc[0]
    baseline = np.mean(actual == majority)
    report = classification_report(actual, predicted, labels=model.classes, zero_division=0)
    print(f"Accuracy: {accuracy:.2%}, Macro-F1: {macro_f1:.4f}")
    print(report)
    predictions = test.copy()
    predictions["predicted_stars"] = predicted
    predictions["correct"] = actual == predicted
    predictions.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    examples = pd.concat([predictions[predictions.correct].head(3), predictions[~predictions.correct].head(3)]).copy()
    examples["review_body"] = examples.review_body.str.replace(r"\s+", " ", regex=True).str.slice(0, 180)
    examples.to_csv(output / "examples.csv", index=False, encoding="utf-8-sig")

    # 4. Heatmap ของ P(คำ | ดาว), Word Cloud 6 ภาพ และ Confusion Matrix
    print("Drawing charts...", flush=True)
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    probabilities = np.exp(model.log_likelihood)
    # Union of each class's top 8 words; selection never examines test data.
    selected = []
    for row in probabilities:
        for index in np.argsort(-row, kind="stable")[:8]:
            if index not in selected:
                selected.append(int(index))
    table = pd.DataFrame(probabilities[:, selected], index=model.classes,
                         columns=[model.words[i] for i in selected])
    table.index.name = "stars"
    fig, ax = plt.subplots(figsize=(max(12, len(selected) * .48), 4.8))
    plot = ax.imshow(table.values, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(len(selected)), table.columns, rotation=60, ha="right")
    ax.set_yticks(range(5), model.classes)
    ax.set(ylabel="Stars", title="P(word | stars), Laplace smoothing (training data)")
    fig.colorbar(plot, ax=ax, label="Probability (shared scale)")
    fig.tight_layout()
    fig.savefig(output / "word_probability_heatmap.png", dpi=180)
    plt.close(fig)

    for label, counts in [("all", model.counts.sum(axis=0))] + list(zip(model.classes, model.counts)):
        frequencies = {word: int(count) for word, count in zip(model.words, counts) if count > 0}
        cloud = WordCloud(font_path=font_path, width=1400, height=850,
                          background_color="white", max_words=100, random_state=42,
                          collocations=False).generate_from_frequencies(frequencies)
        cloud.to_file(str(output / f"wordcloud_{label}.png"))

    fig, ax = plt.subplots(figsize=(6, 5))
    plot = ax.imshow(matrix, cmap="Blues")
    for (row, col), count in np.ndenumerate(matrix):
        ax.text(col, row, str(count), ha="center", va="center",
                color="white" if count > matrix.max() / 2 else "black")
    ax.set_xticks(range(5), model.classes)
    ax.set_yticks(range(5), model.classes)
    ax.set(xlabel="Predicted stars", ylabel="Actual stars", title="Test confusion matrix (counts)")
    fig.colorbar(plot, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    # 5. บันทึกผลสรุป
    summary = f"""# ผลการทดลอง Naive Bayes

Train {len(train):,} รีวิว / Test {len(test):,} รีวิว ตามชุดแบ่งเดิมของ Dataset
คลังคำ {len(model.words):,} คำ ใช้ Laplace smoothing = 1

- Accuracy: **{accuracy:.2%}**
- Macro-F1: **{macro_f1:.4f}**
- Baseline ทาย {majority} ดาวทุกรีวิว: {baseline:.2%}

```text
{report}```

ข้อมูลฝึกส่วนใหญ่เป็น 4–5 ดาว ขณะที่ test มีดาวละ 100 รีวิว
ผลของ 1–2 ดาวยังต่ำ และโมเดลนับคำแยกกันจึงไม่เข้าใจวลี เช่น “ไม่อร่อย”
ตัวอย่างทายถูกและผิดอยู่ใน [examples.csv](examples.csv)

![ความน่าจะเป็นของคำ](word_probability_heatmap.png)

Heatmap ใช้ P(คำ | ดาว) เลือก 8 คำที่มีค่าสูงสุดของแต่ละดาวแล้วรวมโดยไม่ซ้ำ
Word Cloud ใช้ความถี่คำจาก train มีภาพรวมและแยก 1–5 ดาว ขนาดคำปรับแยกในแต่ละภาพ

![ผลทำนายเทียบกับดาวจริง](confusion_matrix.png)
"""
    (output / "summary.md").write_text(summary, encoding="utf-8")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
