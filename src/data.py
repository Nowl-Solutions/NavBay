"""Download the supplied official splits once, then reuse local parquet files."""
import pandas as pd

DATASET = "iamwarint/wongnai-restaurant-review"
SPLITS = {name: f"data/{name}-00000-of-00001.parquet" for name in ("train", "test")}


def load_splits(directory):
    directory.mkdir(parents=True, exist_ok=True)
    frames, audit = {}, {}
    for name, remote in SPLITS.items():
        local = directory / f"{name}.parquet"
        if not local.exists():
            frame = pd.read_parquet(f"hf://datasets/{DATASET}/{remote}")
            frame.to_parquet(local, index=False)
        frame = pd.read_parquet(local)
        if not {"review_body", "stars"}.issubset(frame.columns):
            raise ValueError(f"{name}: expected review_body and stars")
        valid = frame.review_body.notna() & frame.stars.isin([1, 2, 3, 4, 5])
        valid &= frame.review_body.fillna("").astype(str).str.strip().ne("")
        if not valid.all():
            raise ValueError(f"{name}: {(~valid).sum()} invalid rows; inspect before training")
        frames[name] = frame.reset_index(drop=True)
        audit[name] = {
            "rows": len(frame),
            "stars": frame.stars.value_counts().sort_index().to_dict(),
            "duplicate_text_rows": int(frame.review_body.duplicated().sum()),
        }
    overlap = set(frames["train"].review_body) & set(frames["test"].review_body)
    audit["exact_text_overlap_unique"] = len(overlap)
    return frames, audit
