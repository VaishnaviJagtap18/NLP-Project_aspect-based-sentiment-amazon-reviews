"""
Product Review Sentiment Analysis with Aspect Extraction (Amazon reviews)

Pipeline
  1. Load an Amazon reviews CSV (column names auto-detected)
  2. Create sentiment labels from star ratings (1-2 neg, 3 neutral, 4-5 pos)
  3. Train + evaluate a sentiment model (TF-IDF + Logistic Regression, or a
     pretrained transformer)
  4. Split every review into clauses, tag each clause with aspects
     (price, quality, delivery, ...) using a keyword lexicon
  5. Predict sentiment for each (review, aspect) clause
  6. Save CSVs + model for the Streamlit dashboard (app.py)

Usage
  python pipeline.py --data Reviews.csv                 # your Amazon CSV
  python pipeline.py --data *.csv                       # several CSVs are combined + de-duplicated
  python pipeline.py --data Reviews.csv --backend hf --max-reviews 5000   # transformer sentiment
  python pipeline.py --demo                             # synthetic data, smoke test
"""
import argparse
import json
import random
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

LABELS = ["negative", "neutral", "positive"]

# --------------------------------------------------------------------------
# Aspect lexicon: edit freely. Keywords match as word prefixes (batter -> battery).
# --------------------------------------------------------------------------
ASPECTS = {
    "price": ["price", "cost", "cheap", "expensive", "value", "worth", "money",
              "afford", "overpriced", "pricey", "bargain", "deal", "discount"],
    "quality": ["quality", "build", "durab", "material", "sturdy", "flimsy", "broke",
                "broken", "stopped working", "defect", "cheaply made", "well made",
                "construction", "last long", "wear"],
    "performance": ["perform", "speed", "fast", "slow", "battery", "sound", "audio",
                    "screen", "display", "camera", "work", "lag", "powerful", "noise",
                    "taste", "flavor", "flavour", "smell"],
    "delivery": ["deliver", "shipping", "shipped", "arrive", "package", "packaging",
                 "packed", "box", "late", "courier", "dispatch", "on time", "transit"],
    "service": ["service", "support", "refund", "return", "replace", "warranty",
                "seller", "customer care", "customer service", "exchange"],
    "design": ["design", "look", "color", "colour", "size", "fit", "style", "weight",
               "comfort", "appearance", "compact", "light weight", "lightweight"],
}
_ASPECT_RE = {
    a: re.compile(r"\b(?:" + "|".join(re.escape(k) for k in kws) + r")\w*", re.I)
    for a, kws in ASPECTS.items()
}
_SPLIT_RE = re.compile(
    r"(?<=[.!?;])\s+|\s+(?:but|however|although|though|yet|whereas)\s+|\n+", re.I
)

# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
def light_clean(text: str) -> str:
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize(text: str) -> str:
    """Model input: lowercase, keep letters/digits/apostrophes. Keeps 'not', 'no'."""
    text = light_clean(text).lower()
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def stars_to_label(r: float) -> str:
    return "negative" if r <= 2 else "neutral" if r == 3 else "positive"


def split_clauses(text: str):
    parts = [p.strip() for p in _SPLIT_RE.split(light_clean(text)) if p]
    return [p for p in parts if len(p.split()) >= 2]


def find_aspects(clause: str):
    return [a for a, rx in _ASPECT_RE.items() if rx.search(clause)]


# --------------------------------------------------------------------------
# Sentiment back-ends (same interface: predict(texts) -> labels, confidences)
# --------------------------------------------------------------------------
class SKSentiment:
    def __init__(self, max_features=100_000):
        self.pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                      max_features=max_features, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0)),
        ])

    def fit(self, texts, labels):
        self.pipe.fit([normalize(t) for t in texts], labels)
        return self

    def predict(self, texts):
        X = [normalize(t) for t in texts]
        proba = self.pipe.predict_proba(X)
        idx = proba.argmax(1)
        classes = self.pipe.classes_
        return [classes[i] for i in idx], proba.max(1)


class HFSentiment:
    """Pretrained 3-class transformer (no training needed)."""
    def __init__(self, name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        from transformers import pipeline as hf_pipeline
        self.name = name
        self.pipe = hf_pipeline("sentiment-analysis", model=name, top_k=None,
                                truncation=True, max_length=128, device=-1)

    def predict(self, texts, batch_size=32):
        out = self.pipe([light_clean(t)[:1000] for t in texts], batch_size=batch_size)
        labels, confs = [], []
        for res in out:
            best = max(res, key=lambda d: d["score"])
            labels.append(best["label"].lower())
            confs.append(best["score"])
        return labels, np.array(confs)


def load_backend(outdir: Path):
    meta = json.loads((outdir / "meta.json").read_text())
    if meta["backend"] == "hf":
        return HFSentiment(meta.get("hf_model", "cardiffnlp/twitter-roberta-base-sentiment-latest"))
    return joblib.load(outdir / "model.joblib")


# --------------------------------------------------------------------------
# Data loading (auto-detect Amazon column names)
# --------------------------------------------------------------------------
TEXT_COLS = ["reviews.text", "reviewText", "Text", "review_body", "text", "review",
             "Review", "content", "body"]
TITLE_COLS = ["reviews.title", "summary", "Summary", "review_headline", "title", "Title"]
RATING_COLS = ["reviews.rating", "overall", "Score", "star_rating", "rating", "Rating",
               "stars", "reviewRating"]
PRODUCT_COLS = ["name", "product_title", "asin", "ProductId", "product_id", "parent_asin",
                "Product", "product", "product_name"]
DATE_COLS = ["reviews.date", "reviewTime", "unixReviewTime", "Time", "review_date",
             "date", "timestamp"]


def _pick(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None


def _read_table(p: Path) -> pd.DataFrame:
    """Read a CSV/JSONL. For CSVs only the columns we need are loaded (saves RAM on 250MB files)."""
    if p.suffix.lower() in (".json", ".jsonl"):
        return pd.read_json(p, lines=True)
    head = pd.read_csv(p, nrows=0, encoding_errors="replace")
    wanted = [_pick(head, c) for c in (TEXT_COLS, TITLE_COLS, RATING_COLS, PRODUCT_COLS, DATE_COLS)]
    usecols = [c for c in wanted if c]
    return pd.read_csv(p, usecols=usecols or None, low_memory=False,
                       on_bad_lines="skip", encoding_errors="replace")


def _clean_product(s: pd.Series) -> pd.Series:
    """Datafiniti names look like 'Fire Tablet, 7 Display,,,\\r\\nFire Tablet, 7...' -> keep the first."""
    s = s.fillna("Unknown product").astype(str)
    s = s.str.split(r",,,|[\r\n]+", regex=True).str[0].str.strip()
    return s.replace("", "Unknown product")


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    raw = _read_table(p)
    tcol, rcol = _pick(raw, TEXT_COLS), _pick(raw, RATING_COLS)
    if tcol is None or rcol is None:
        raise ValueError(
            f"{p.name}: could not find review-text / rating columns. Columns: {list(raw.columns)}")
    pcol, dcol, ttl = _pick(raw, PRODUCT_COLS), _pick(raw, DATE_COLS), _pick(raw, TITLE_COLS)
    print(f"{p.name}: {len(raw):,} rows -> text: {tcol}, rating: {rcol}, "
          f"product: {pcol}, date: {dcol}, title: {ttl}")

    text = raw[tcol]
    if ttl and ttl != tcol:  # title often carries strong sentiment
        text = (raw[ttl].fillna("").astype(str) + ". " + text.astype(str)).where(text.notna())
    df = pd.DataFrame({
        "text": text,
        "rating": pd.to_numeric(raw[rcol], errors="coerce"),
        "product": _clean_product(raw[pcol]) if pcol else "All products",
    })
    if dcol:
        d = raw[dcol]
        if pd.api.types.is_numeric_dtype(d):
            unit = "ms" if d.dropna().median() > 1e11 else "s"
            df["date"] = pd.to_datetime(d, unit=unit, errors="coerce")
        else:
            df["date"] = pd.to_datetime(d, errors="coerce", utc=True).dt.tz_localize(None)
    return df


def prepare(df: pd.DataFrame, max_reviews: int, seed: int = 42) -> pd.DataFrame:
    df = df.dropna(subset=["text", "rating"]).copy()
    df["text"] = df["text"].map(light_clean)
    df = df[df["text"].str.split().str.len() >= 3]
    df = df.drop_duplicates(subset=["text", "product"])
    df = df[df["rating"].between(1, 5)]
    df["sentiment_true"] = df["rating"].round().astype(int).map(stars_to_label)
    if len(df) > max_reviews:
        df = df.sample(max_reviews, random_state=seed)
    df = df.reset_index(drop=True)
    df.insert(0, "review_id", df.index)
    return df


# --------------------------------------------------------------------------
# Demo data so the whole thing runs without downloading anything
# --------------------------------------------------------------------------
def make_demo(n=4000, seed=0) -> pd.DataFrame:
    rnd = random.Random(seed)
    products = ["Demo Wireless Headphones", "Demo Phone Charger", "Demo Smart Watch"]
    pos = {
        "price": ["great price for what you get", "totally worth the money", "very cheap and good value"],
        "quality": ["build quality is excellent", "feels sturdy and durable", "well made material"],
        "performance": ["sound is amazing", "battery lasts all day", "works perfectly and fast"],
        "delivery": ["delivery was quick", "arrived on time", "packaging was neat"],
        "service": ["customer service was helpful", "easy refund process", "seller replied fast"],
        "design": ["looks stylish", "very comfortable fit", "love the color"],
    }
    neg = {
        "price": ["way too expensive", "not worth the money", "overpriced for this"],
        "quality": ["feels flimsy and cheaply made", "it broke after a week", "poor build quality"],
        "performance": ["battery drains fast", "sound is terrible", "it is slow and laggy"],
        "delivery": ["delivery was very late", "box arrived damaged", "shipping took forever"],
        "service": ["customer service never replied", "refund was refused", "seller was rude"],
        "design": ["looks ugly", "too heavy and uncomfortable", "the size is wrong"],
    }
    rows = []
    for i in range(n):
        k = rnd.randint(1, 3)
        asps = rnd.sample(list(ASPECTS), k)
        signs = [rnd.random() < 0.6 for _ in asps]
        parts = [rnd.choice((pos if s else neg)[a]) for a, s in zip(asps, signs)]
        frac = sum(signs) / k
        rating = 5 if frac == 1 and rnd.random() < .6 else 4 if frac == 1 else \
            3 if frac == .5 or (0 < frac < 1) else 1 if rnd.random() < .5 else 2
        text = ". ".join(parts[:1]) + "".join(f" but {p}" if (signs[0] != signs[j + 1]) else f". {p}"
                                              for j, p in enumerate(parts[1:])) + "."
        rows.append({"text": text.capitalize(), "rating": rating,
                     "product": rnd.choice(products),
                     "date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=rnd.randint(0, 600))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Main steps
# --------------------------------------------------------------------------
def evaluate(model, test_df, outdir: Path, limit=3000):
    t = test_df.sample(min(limit, len(test_df)), random_state=0)
    pred, _ = model.predict(t["text"].tolist())
    metrics = {
        "n_test": int(len(t)),
        "accuracy": float(accuracy_score(t["sentiment_true"], pred)),
        "macro_f1": float(f1_score(t["sentiment_true"], pred, average="macro")),
        "report": classification_report(t["sentiment_true"], pred, labels=LABELS,
                                        output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(t["sentiment_true"], pred, labels=LABELS).tolist(),
        "labels": LABELS,
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Test accuracy {metrics['accuracy']:.3f} | macro-F1 {metrics['macro_f1']:.3f}")
    print(classification_report(t["sentiment_true"], pred, labels=LABELS, zero_division=0))
    return metrics


def extract_aspect_clauses(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        for clause in split_clauses(r.text):
            for asp in find_aspects(clause):
                rows.append((r.review_id, r.product, asp, clause))
    return pd.DataFrame(rows, columns=["review_id", "product", "aspect", "sentence"])


def run(args):
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        raw = make_demo()
    else:
        raw = pd.concat([load_data(f) for f in args.data], ignore_index=True)
        print(f"Combined {len(raw):,} rows from {len(args.data)} file(s)")
    df = prepare(raw, args.max_reviews)
    print(f"{len(df)} reviews after cleaning. Class balance:\n"
          f"{df['sentiment_true'].value_counts(normalize=True).round(3)}")

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["sentiment_true"],
                                         random_state=42)
    if args.backend == "hf":
        model = HFSentiment(args.hf_model)
        meta = {"backend": "hf", "hf_model": args.hf_model}
    else:
        model = SKSentiment().fit(train_df["text"], train_df["sentiment_true"])
        joblib.dump(model, outdir / "model.joblib")
        meta = {"backend": "sk"}
    (outdir / "meta.json").write_text(json.dumps(meta))

    evaluate(model, test_df, outdir)

    # Review-level sentiment for every review
    df["sentiment_pred"], df["confidence"] = model.predict(df["text"].tolist())

    # Aspect-level sentiment
    asp = extract_aspect_clauses(df)
    if asp.empty:
        raise SystemExit("No aspects found - check the aspect lexicon / review text.")
    asp["sentiment"], asp["confidence"] = model.predict(asp["sentence"].tolist())
    if "date" in df.columns:
        asp = asp.merge(df[["review_id", "date"]], on="review_id", how="left")

    df.to_csv(outdir / "reviews_scored.csv", index=False)
    asp.to_csv(outdir / "aspect_sentiment.csv", index=False)
    print(f"\nSaved to {outdir}/ : reviews_scored.csv ({len(df)} rows), "
          f"aspect_sentiment.csv ({len(asp)} rows), metrics.json")
    print(asp.groupby(["aspect", "sentiment"]).size().unstack(fill_value=0))
    print("\nNext: streamlit run app.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+",
                    help="One or more Amazon review CSV/JSONL files; wildcards like *.csv work")
    ap.add_argument("--demo", action="store_true", help="Use synthetic demo data")
    ap.add_argument("--backend", choices=["sk", "hf"], default="sk")
    ap.add_argument("--hf-model", dest="hf_model",
                    default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    ap.add_argument("--max-reviews", dest="max_reviews", type=int, default=100000)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    if not a.demo and not a.data:
        ap.error("Provide --data FILE [FILE ...] (wildcards ok) or use --demo")
    if a.data:  # expand wildcards ourselves so it also works in Windows PowerShell/cmd
        import glob
        a.data = [f for pat in a.data for f in (sorted(glob.glob(pat)) or [pat])]
        missing = [f for f in a.data if not Path(f).exists()]
        if missing:
            ap.error(f"File(s) not found: {missing}. Run this from the folder that holds the CSVs.")
        print("Input files:", *a.data, sep="\n  ")
    # Re-import as a normal module so the saved model pickles as `pipeline.SKSentiment`
    # (not `__main__.SKSentiment`), which lets app.py load it.
    import pipeline as _pipeline
    _pipeline.run(a)
