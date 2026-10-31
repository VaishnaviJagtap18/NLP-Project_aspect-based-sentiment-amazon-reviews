# Product Review Sentiment Analysis with Aspect Extraction

Classify Amazon product reviews as **positive, negative or neutral**, work out **which aspect** each opinion is about (price, quality, delivery, ...), and explore the results in an **interactive aspect-wise sentiment dashboard**.

> A review like *"Great sound but the delivery was late and the price is too high"* is not simply "mixed". This project breaks it down: **performance → positive, delivery → negative, price → negative**.

**🔴 Live demo: [Open the dashboard](https://nlp-project-aspectextraction.streamlit.app/)**

---

## Table of contents
1. [Features](#features)
2. [How it works](#how-it-works)
3. [Dataset](#dataset)
4. [Project structure](#project-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Methodology](#methodology)
8. [Output files](#output-files)
9. [Dashboard](#dashboard)
10. [Results](#results)
11. [Limitations](#limitations)
12. [Future work](#future-work)
13. [Troubleshooting](#troubleshooting)
14. [Acknowledgements](#acknowledgements)

---

## Features
- **Review-level sentiment classification** (positive / neutral / negative) from star ratings.
- **Two interchangeable sentiment back-ends**
  - TF-IDF (1-2 grams) + Logistic Regression: fast, runs on any laptop.
  - Pretrained transformer (`cardiffnlp/twitter-roberta-base-sentiment-latest`): no training, optional.
- **Aspect extraction** with an editable keyword lexicon: price, quality, performance, delivery, service, design.
- **Aspect-level sentiment**: every review is split into clauses and each aspect clause is scored separately.
- **Multi-file input**: point it at several CSVs; they are combined and de-duplicated.
- **Automatic column detection** for common Amazon dataset formats (Datafiniti, Fine Food Reviews, Amazon 2018/2023).
- **Streamlit dashboard** with product filter, aspect charts, example sentences, trends, word clouds, a live "try your own review" box and model evaluation.
- **Demo mode** (`--demo`) that runs the whole pipeline on synthetic data, with no download needed.

## How it works

```
 Amazon review CSV(s)
        |
        v
 [1] Load + clean            auto-detect columns, merge title + text, drop duplicates
        |
        v
 [2] Create labels           1-2 stars = negative, 3 = neutral, 4-5 = positive
        |
        v
 [3] Train / evaluate        80/20 stratified split -> accuracy, macro-F1, confusion matrix
        |
        v
 [4] Aspect extraction       split review into clauses -> tag clauses with aspects (keyword lexicon)
        |
        v
 [5] Aspect sentiment        predict sentiment for every (review, aspect) clause
        |
        v
 [6] Save results            CSVs + model + metrics  ->  Streamlit dashboard
```

## Dataset

This project uses the Kaggle dataset **Consumer Reviews of Amazon Products** (Datafiniti):
<https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products>

- A sample of consumer reviews for Amazon products such as the Kindle and Fire TV Stick, from Datafiniti's product database (September 2017 to October 2018).
- The download contains three CSV files. They overlap, so the pipeline combines them and removes duplicate reviews.
- Columns used: `reviews.text`, `reviews.title`, `reviews.rating`, `name`, `reviews.date`.

The data is **not included in this repository** (files are large and licensed by their owners). Download it from Kaggle and place the CSVs in the project folder.

Other datasets also work because columns are auto-detected, for example Kaggle's *Amazon Fine Food Reviews* (`Reviews.csv`) or Amazon Reviews 2018/2023 category files in CSV/JSONL format.

## Project structure

```
.
├── pipeline.py        # data loading, sentiment model, aspect extraction, saves results
├── app.py             # Streamlit dashboard
├── requirements.txt   # Python dependencies
├── README.md
├── .gitignore
└── outputs/           # created by pipeline.py (not committed)
    ├── reviews_scored.csv
    ├── aspect_sentiment.csv
    ├── metrics.json
    ├── meta.json
    └── model.joblib
```

## Installation

**Requirements:** Python 3.10 or newer is recommended.

```bash
# 1. clone the repository
git clone https://github.com/VaishnaviJagtap18/<repo-name>.git
cd <repo-name>

# 2. create and activate a virtual environment
python -m venv nlpvenv
nlpvenv\Scripts\activate          # Windows
# source nlpvenv/bin/activate     # macOS / Linux

# 3. install dependencies
pip install -r requirements.txt
```

To use the transformer back-end (optional), also run `pip install transformers torch`.

## Usage

### 1. Run the pipeline

Place the downloaded CSV files in the project folder, then:

```bash
# all CSVs in the folder (combined + de-duplicated)
python pipeline.py --data *.csv

# a single file
python pipeline.py --data Reviews.csv

# pretrained transformer for sentiment (slower; use a smaller sample)
python pipeline.py --data *.csv --backend hf --max-reviews 5000

# quick smoke test with synthetic data (no download)
python pipeline.py --demo
```

| Option | Default | Description |
|---|---|---|
| `--data` | none | One or more CSV/JSONL files. Wildcards such as `*.csv` work on all platforms. |
| `--demo` | off | Use synthetic demo data instead of a real dataset. |
| `--backend` | `sk` | `sk` = TF-IDF + Logistic Regression, `hf` = pretrained transformer. |
| `--hf-model` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Hugging Face model for the `hf` back-end. |
| `--max-reviews` | `100000` | Randomly sample at most this many reviews (lower it if you run out of memory). |
| `--out` | `outputs` | Folder for results. |

### 2. Launch the dashboard

```bash
streamlit run app.py
```

Open <http://localhost:8501> if the browser does not open automatically. Run the pipeline first, because the dashboard reads the files in `outputs/`.

## Methodology

### Data preparation
- The review title is prepended to the review text, since titles often carry strong sentiment.
- HTML tags and URLs are removed, whitespace is normalised and text is lowercased for the model. Negations such as "not" and "no" are deliberately **kept**.
- Reviews with fewer than 3 words, missing ratings, or duplicate (text, product) pairs are dropped.
- Product names are cleaned (Datafiniti names repeat the title several times).

### Sentiment labels
Star ratings are converted to labels:

| Stars | Label |
|---|---|
| 1-2 | negative |
| 3 | neutral |
| 4-5 | positive |

### Sentiment model
- **TF-IDF + Logistic Regression**: unigrams and bigrams, `min_df=2`, up to 100,000 features, sublinear TF, `class_weight="balanced"` to handle the dominance of positive reviews.
- **Transformer (optional)**: pretrained 3-class RoBERTa model used as-is.
- The split is 80% train / 20% test (stratified, fixed seed). Evaluation uses accuracy, **macro-F1** (more honest than accuracy on imbalanced data) and a confusion matrix.

### Aspect extraction
Each review is split into clauses at sentence boundaries and at contrast words (*but, however, although, though, yet, whereas*). A clause is tagged with an aspect when it contains one of that aspect's keywords (matched as word prefixes, so `batter` matches *battery* and *batteries*).

| Aspect | Example keywords |
|---|---|
| price | price, cost, cheap, expensive, value, worth, money |
| quality | quality, build, durable, material, sturdy, broke, defect |
| performance | speed, fast, slow, battery, sound, screen, camera, lag |
| delivery | delivery, shipping, arrived, packaging, late, courier |
| service | service, support, refund, return, replace, warranty, seller |
| design | design, look, color, size, fit, style, weight, comfort |

The full lexicon is the `ASPECTS` dictionary at the top of `pipeline.py`. Edit it to add aspects or keywords for your own product category.

### Aspect-level sentiment
The sentiment model is applied to **each aspect clause** (not the whole review), giving one row per (review, aspect). Results are aggregated per product into positive / neutral / negative percentages and a **net sentiment score** (positive % minus negative %).

## Output files

All files are written to `outputs/`.

| File | Contents |
|---|---|
| `reviews_scored.csv` | One row per review: `review_id`, `text`, `rating`, `product`, `date`, `sentiment_true` (from stars), `sentiment_pred`, `confidence` |
| `aspect_sentiment.csv` | One row per (review, aspect): `review_id`, `product`, `aspect`, `sentence`, `sentiment`, `confidence`, `date` |
| `metrics.json` | Accuracy, macro-F1, per-class report and confusion matrix on the test set |
| `model.joblib`, `meta.json` | Saved model and the back-end used (loaded by the dashboard) |

## Dashboard

- **Filters**: product, minimum confidence, aspects shown.
- **KPIs**: number of reviews, average rating, share of positive and negative reviews.
- **Sentiment by aspect**: stacked bars (positive / neutral / negative) and a net-sentiment ranking.
- **Example sentences**: the most confident positive / neutral / negative sentences for any aspect.
- **Trend**: average sentiment per aspect over time.
- **Word clouds** for positive and negative reviews.
- **Try your own review**: type a review and see its overall sentiment and the aspect-wise breakdown.
- **Model evaluation**: accuracy, macro-F1 and confusion matrix.

<!-- Add screenshots after you run the dashboard:
![Aspect-wise sentiment](docs/dashboard.png)
-->


## Limitations
- **Class imbalance**: most reviews are 4-5 stars, so negative and neutral examples are scarce and neutral is the hardest class.
- **Noisy labels**: star ratings are only a proxy for sentiment (for example, a 3-star review can be positive about one thing and negative about another).
- **Keyword-based aspects**: the lexicon misses implicit aspects ("it died in a week" is about quality) and can mis-tag words with several meanings.
- **Clause-level sentiment**: a model trained on whole reviews is applied to short clauses. A clause like "price is too high and delivery was late" gets the same label for both aspects.
- **Sarcasm and negation** are only partly handled.
- Only English reviews are supported.

## Future work
- Learn aspects automatically (spaCy noun phrases, KeyBERT, or zero-shot classification with `facebook/bart-large-mnli`).
- Fine-tune a transformer such as DistilBERT on the labelled reviews.
- Train on labelled aspect data (for example SemEval ABSA) and measure aspect precision and recall.
- Handle implicit aspects and split clauses more precisely.
- Support Hinglish and other Indian-language reviews.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | The virtual environment is not active. Activate it and run `pip install -r requirements.txt`. |
| `Could not find review-text / rating columns` | The file uses unfamiliar column names. Add them to `TEXT_COLS` / `RATING_COLS` in `pipeline.py`. |
| `File(s) not found` | Run the command from the folder that contains the CSV files. |
| Out-of-memory error | Use `--max-reviews 30000` (or lower). |
| Dashboard says "No results found" | Run `python pipeline.py --data *.csv` first. |
| Dashboard cannot load the model | Re-run `pipeline.py` with the same environment you use for `streamlit run`. |
| PowerShell blocks activation | Run `Set-ExecutionPolicy -Scope Process Bypass`, then activate again. |

## Acknowledgements
- Dataset: *Consumer Reviews of Amazon Products*, Datafiniti, via Kaggle.
- Pretrained model: [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).
- Libraries: scikit-learn, pandas, Streamlit, Plotly, Hugging Face Transformers.

## Authors
Vaishnavi Jagtap · [GitHub](https://github.com/VaishnaviJagtap18)
Komal Ghadage . [Github](#link)