"""Training-only word visualizations and held-out confusion matrix."""
from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from wordcloud import WordCloud


def thai_font(explicit=None):
    candidates = [explicit, os.environ.get("THAI_FONT"), "C:/Windows/Fonts/tahoma.ttf"]
    candidates += [f.fname for f in font_manager.fontManager.ttflist
                   if any(name in f.name.lower() for name in ("noto sans thai", "sarabun", "tlwg"))]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise FileNotFoundError("Supply a Thai font with --font PATH or THAI_FONT")


def make_plots(model, matrix, output, font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    probabilities = np.exp(model.log_likelihood)
    # Union of each class's top 8 words; selection never examines test data.
    selected = list(dict.fromkeys(
        int(index) for row in probabilities for index in np.argsort(-row, kind="stable")[:8]
    ))
    table = pd.DataFrame(probabilities[:, selected], index=model.classes,
                         columns=[model.words[i] for i in selected])
    table.index.name = "stars"
    table.to_csv(output / "word_probabilities.csv", encoding="utf-8-sig")
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
