"""Aspect-wise sentiment dashboard.  Run:  streamlit run app.py"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from pipeline import ASPECTS, find_aspects, load_backend, split_clauses  # noqa: F401

st.set_page_config(page_title="Aspect-wise Review Sentiment", layout="wide")
OUT = Path("outputs")
COLORS = {"positive": "#2e9e5b", "neutral": "#9aa0a6", "negative": "#d64545"}
ORDER = ["positive", "neutral", "negative"]


@st.cache_data
def load_tables():
    reviews = pd.read_csv(OUT / "reviews_scored.csv", parse_dates=["date"], low_memory=False) \
        if "date" in pd.read_csv(OUT / "reviews_scored.csv", nrows=1).columns \
        else pd.read_csv(OUT / "reviews_scored.csv", low_memory=False)
    aspects = pd.read_csv(OUT / "aspect_sentiment.csv", low_memory=False)
    if "date" in aspects.columns:
        aspects["date"] = pd.to_datetime(aspects["date"], errors="coerce")
    metrics = json.loads((OUT / "metrics.json").read_text())
    return reviews, aspects, metrics


@st.cache_resource
def get_model():
    return load_backend(OUT)


if not (OUT / "aspect_sentiment.csv").exists():
    st.error("No results found. Run `python pipeline.py --data <your_csv>` (or `--demo`) first.")
    st.stop()

reviews, aspects, metrics = load_tables()

# ---------------- Sidebar ----------------
st.sidebar.header("Filters")
top = reviews["product"].value_counts()
product = st.sidebar.selectbox(
    "Product", ["All products"] + top.index.tolist(),
    format_func=lambda p: p if p == "All products" else f"{p[:60]}  ({top[p]})")
min_conf = st.sidebar.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
sel_aspects = st.sidebar.multiselect("Aspects", list(ASPECTS), default=list(ASPECTS))

rv = reviews if product == "All products" else reviews[reviews["product"] == product]
av = aspects if product == "All products" else aspects[aspects["product"] == product]
av = av[(av["confidence"] >= min_conf) & (av["aspect"].isin(sel_aspects))]

st.title("Aspect-wise Sentiment Dashboard")
st.caption(product if product != "All products" else "All products in the dataset")

# ---------------- KPIs ----------------
share = rv["sentiment_pred"].value_counts(normalize=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews", f"{len(rv):,}")
c2.metric("Avg. rating", f"{rv['rating'].mean():.2f} ★")
c3.metric("Positive", f"{share.get('positive', 0):.0%}")
c4.metric("Negative", f"{share.get('negative', 0):.0%}")

# ---------------- Aspect chart ----------------
st.subheader("Sentiment by aspect")
if av.empty:
    st.info("No aspect mentions for this selection.")
else:
    counts = av.groupby(["aspect", "sentiment"]).size().unstack(fill_value=0)
    for s in ORDER:
        if s not in counts:
            counts[s] = 0
    counts = counts[ORDER]
    counts["total"] = counts.sum(axis=1)
    pct = counts[ORDER].div(counts["total"], axis=0) * 100
    pct["net"] = pct["positive"] - pct["negative"]

    left, right = st.columns([3, 2])
    long = pct[ORDER].reset_index().melt("aspect", var_name="sentiment", value_name="percent")
    fig = px.bar(long, x="aspect", y="percent", color="sentiment", color_discrete_map=COLORS,
                 category_orders={"sentiment": ORDER}, text=long["percent"].round(0).astype(int).astype(str) + "%")
    fig.update_layout(barmode="stack", yaxis_title="% of mentions", xaxis_title=None,
                      legend_title=None, height=420)
    left.plotly_chart(fig, width="stretch")

    net = pct["net"].sort_values().reset_index()
    fig2 = px.bar(net, x="net", y="aspect", orientation="h", color="net",
                  color_continuous_scale=["#d64545", "#e8e8e8", "#2e9e5b"], range_color=[-100, 100])
    fig2.update_layout(title="Net sentiment (pos% − neg%)", coloraxis_showscale=False,
                       xaxis_title=None, yaxis_title=None, height=420)
    right.plotly_chart(fig2, width="stretch")

    summary = counts.join(pct["net"].round(1).rename("net score"))
    st.dataframe(summary.rename(columns={"total": "mentions"}), width="stretch")

    # ---------------- Examples ----------------
    st.subheader("Example sentences")
    e1, e2 = st.columns(2)
    asp_pick = e1.selectbox("Aspect", sel_aspects)
    sent_pick = e2.selectbox("Sentiment", ORDER, index=2)
    ex = av[(av["aspect"] == asp_pick) & (av["sentiment"] == sent_pick)] \
        .sort_values("confidence", ascending=False).head(10)
    if ex.empty:
        st.write("No examples.")
    else:
        st.dataframe(ex[["sentence", "confidence"]].reset_index(drop=True), width="stretch")

    # ---------------- Trend ----------------
    if "date" in av.columns and av["date"].notna().any():
        st.subheader("Net sentiment over time")
        t = av.dropna(subset=["date"]).copy()
        t["month"] = t["date"].dt.to_period("M").dt.to_timestamp()
        t["score"] = t["sentiment"].map({"positive": 1, "neutral": 0, "negative": -1})
        tr = t.groupby(["month", "aspect"])["score"].mean().reset_index()
        st.plotly_chart(px.line(tr, x="month", y="score", color="aspect", markers=True)
                        .update_layout(height=380, yaxis_title="avg sentiment (-1..1)"),
                        width="stretch")

# ---------------- Word clouds ----------------
with st.expander("Word clouds (positive vs negative reviews)"):
    try:
        import matplotlib.pyplot as plt
        from wordcloud import STOPWORDS, WordCloud
        cols = st.columns(2)
        for col, s in zip(cols, ["positive", "negative"]):
            txt = " ".join(rv.loc[rv["sentiment_pred"] == s, "text"].head(3000))
            if txt.strip():
                wc = WordCloud(width=600, height=350, background_color="white",
                               stopwords=STOPWORDS).generate(txt)
                fig_, ax = plt.subplots()
                ax.imshow(wc)
                ax.axis("off")
                ax.set_title(s.title())
                col.pyplot(fig_)
    except ImportError:
        st.write("Install `wordcloud` and `matplotlib` to see word clouds.")

# ---------------- Live predictor ----------------
st.subheader("Try your own review")
txt = st.text_area("Type a review", "Great sound but the delivery was late and the price is too high.")
if st.button("Analyze") and txt.strip():
    model = get_model()
    overall, oc = model.predict([txt])
    st.write(f"**Overall:** {overall[0]} ({oc[0]:.0%} confidence)")
    rows = [(a, c) for c in split_clauses(txt) for a in find_aspects(c)]
    if rows:
        labs, confs = model.predict([c for _, c in rows])
        st.table(pd.DataFrame({"aspect": [a for a, _ in rows], "clause": [c for _, c in rows],
                               "sentiment": labs, "confidence": [f"{x:.0%}" for x in confs]}))
    else:
        st.write("No known aspect keywords found in this text.")

# ---------------- Model quality ----------------
with st.expander("Model evaluation (review-level sentiment, held-out test set)"):
    m1, m2 = st.columns(2)
    m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    m2.metric("Macro-F1", f"{metrics['macro_f1']:.3f}")
    cm = pd.DataFrame(metrics["confusion_matrix"], index=[f"true {l}" for l in metrics["labels"]],
                      columns=[f"pred {l}" for l in metrics["labels"]])
    st.dataframe(cm)
