# Naive Bayes วิเคราะห์รีวิว Wongnai (1–5 ดาว)
# ติดตั้งก่อน: pip install datasets pandas pythainlp wordcloud scikit-learn seaborn matplotlib

import os, re, urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from datasets import load_dataset
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud

os.makedirs("outputs", exist_ok=True)

# ---------- 1) โหลด Dataset ----------
ds = load_dataset("Highgroundbkk/wongnai_reviews")
df = pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)
df = df.rename(columns={"star_rating": "rating"})
df = df[df["rating"] >= 0]
if df["rating"].min() == 0:          # label 0–4 → 1–5 ดาว
    df["rating"] = df["rating"] + 1
print(df["rating"].value_counts().sort_index())

# ---------- 2) Preprocessing: ตัดคำ + Stop words removing ----------
# (ภาษาไทยไม่มีการผันคำ จึงไม่ต้องทำ stemming)
stopwords = set(thai_stopwords()) | {
    "ร้าน", "ร้านอาหาร", "อาหาร", "กิน", "ทาน", "สั่ง", "เมนู", "จาน", "บาท",
    "คน", "ผม", "ลอง", "รสชาติ", "ราคา", "นะคะ", "ค่ะ", "ครับ",
}

def preprocess(text):
    text = re.sub(r"[^ก-๙a-z\s]", " ", text.lower())   # เก็บแค่อักษรไทย/อังกฤษ
    text = re.sub(r"(.)\1{2,}", r"\1", text)             # อร่อยยยย → อร่อย
    tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
    return [t for t in tokens if t not in stopwords and len(t) > 1]

print("กำลังตัดคำ... (ใช้เวลาสักครู่)")
df["tokens"] = df["review_body"].apply(preprocess)

# ---------- 3) แบ่ง Train/Test + สร้าง TF ----------
X_train, X_test, y_train, y_test = train_test_split(
    df["tokens"], df["rating"], test_size=0.2, random_state=42, stratify=df["rating"])

vectorizer = CountVectorizer(analyzer=lambda x: x, min_df=3)
X_train_tf = vectorizer.fit_transform(X_train)
X_test_tf = vectorizer.transform(X_test)
vocab = vectorizer.get_feature_names_out()

# ---------- 4) Naive Bayes (alpha=1 คือ Laplace smoothing +1) ----------
model = MultinomialNB(alpha=1.0)
model.fit(X_train_tf, y_train)

# ฟอนต์ไทยสำหรับกราฟ
font_path = "Sarabun-Regular.ttf"
if not os.path.exists(font_path):
    urllib.request.urlretrieve(
        "https://github.com/google/fonts/raw/main/ofl/sarabun/Sarabun-Regular.ttf", font_path)
fm.fontManager.addfont(font_path)
plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()

# ---------- 5) Heat map ของ P(คำ | ดาว) ----------
prob = pd.DataFrame(np.exp(model.feature_log_prob_), index=model.classes_, columns=vocab)
words = []
for r in model.classes_:                       # คำที่เด่นที่สุด 5 คำของแต่ละดาว
    ratio = prob.loc[r] / prob.drop(index=r).mean()
    words += list(ratio[prob.loc[r] > 0.0005].nlargest(5).index)

plt.figure(figsize=(16, 5))
sns.heatmap(prob[words] * 1000, annot=True, fmt=".2f", cmap="YlOrRd")
plt.title("P(คำ | ดาว) × 1000")
plt.savefig("outputs/heatmap.png", dpi=200, bbox_inches="tight")
plt.show()

# ---------- 6) Word cloud แยก 1–5 ดาว ----------
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
for ax, r in zip(axes, range(1, 6)):
    freq = dict(zip(vocab, np.asarray(X_train_tf[(y_train == r).values].sum(axis=0)).ravel()))
    wc = WordCloud(font_path=font_path, width=500, height=400, background_color="white",
                   prefer_horizontal=1.0, colormap="Dark2").generate_from_frequencies(freq)
    ax.imshow(wc)
    ax.axis("off")
    ax.set_title(f"{r} ดาว", fontsize=24)
plt.savefig("outputs/wordcloud.png", dpi=200, bbox_inches="tight")
plt.show()

# ---------- 7) Performance test ----------
y_pred = model.predict(X_test_tf)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy = {acc:.4f}")
print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=range(1, 6), yticklabels=range(1, 6))
plt.xlabel("ทำนาย")
plt.ylabel("จริง")
plt.title(f"Confusion Matrix (Accuracy = {acc:.2%})")
plt.savefig("outputs/confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.show()
