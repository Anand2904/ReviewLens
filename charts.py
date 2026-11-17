"""All chart-building code (Plotly). Every function takes the analysed table and returns a
figure, or None when the data cannot support that chart (e.g. no dates in the file).

Colour rules:
  * sentiment / urgency use FIXED meaning colours (green / grey / red, grey / amber / red)
  * each category keeps the SAME colour in every chart
  * one axis per chart, thin bars, gaps between stacked segments, numbers shown on bars
"""
import pandas as pd
import plotly.graph_objects as go

INK, GRID = "#0F1111", "#E7E7E7"

SENT_ORDER = ["positive", "neutral", "negative"]
SENT_COLOR = {"positive": "#008300", "neutral": "#8A8F98", "negative": "#E34948"}
URG_ORDER = ["low", "medium", "high"]
URG_COLOR = {"low": "#8A8F98", "medium": "#EDA100", "high": "#E34948"}
CAT_COLOR = {"food_quality": "#2A78D6", "delivery": "#EB6834", "price": "#1BAF7A",
             "staff": "#E87BA4", "packaging": "#6250D6", "app": "#EDA100", "other": "#8A8F98"}


def _style(fig, height=320, legend=False):
    fig.update_layout(
        height=height, margin=dict(l=10, r=30, t=30 if legend else 10, b=10),
        paper_bgcolor="white", plot_bgcolor="white", font=dict(color=INK, size=13),
        showlegend=legend, legend=dict(orientation="h", y=1.15, x=0, traceorder="normal"), bargap=0.45,
        hoverlabel=dict(bgcolor="white", font_color=INK))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=False)
    return fig


def _hbar(labels, values, colors, unit="reviews", text=None):
    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation="h", marker_color=colors,
        text=text or values, textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x} " + unit + "<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _style(fig, height=max(180, 44 * len(labels) + 50))


def sentiment_bar(df):
    c = df["sentiment"].value_counts().reindex(SENT_ORDER, fill_value=0)
    pct = [f"{v}  ({v / max(len(df), 1):.0%})" for v in c.values]
    return _hbar([s.capitalize() for s in c.index], c.values, [SENT_COLOR[s] for s in c.index], text=pct)


def urgency_bar(df):
    c = df["urgency"].value_counts().reindex(URG_ORDER, fill_value=0)
    return _hbar([s.capitalize() for s in c.index], c.values, [URG_COLOR[s] for s in c.index])


def category_bar(df):
    c = df["category"].value_counts()
    if c.empty:
        return None
    return _hbar(list(c.index), c.values, [CAT_COLOR.get(k, "#2A78D6") for k in c.index])


def stacked_by(df, col, top=10):
    """Sentiment split for each value of `col` (category, platform, outlet...)."""
    if df[col].nunique() < 1:
        return None
    ct = pd.crosstab(df[col].astype(str), df["sentiment"]).reindex(columns=SENT_ORDER, fill_value=0)
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).head(top).index]
    fig = go.Figure()
    for s in SENT_ORDER:
        fig.add_bar(y=list(ct.index), x=ct[s], name=s.capitalize(), orientation="h",
                    marker_color=SENT_COLOR[s], marker_line=dict(color="white", width=2),
                    hovertemplate="%{y} · " + s + ": %{x}<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_yaxes(autorange="reversed")
    return _style(fig, height=max(200, 44 * len(ct) + 80), legend=True)


def daily_trend(df):
    d = df[df["date"].astype(str) != ""]
    if d["date"].nunique() < 2:
        return None
    pv = pd.crosstab(d["date"], d["sentiment"]).reindex(columns=SENT_ORDER, fill_value=0).sort_index()
    fig = go.Figure()
    for s in SENT_ORDER:
        fig.add_bar(x=list(pv.index), y=pv[s], name=s.capitalize(), marker_color=SENT_COLOR[s],
                    marker_line=dict(color="white", width=2),
                    hovertemplate="%{x} · " + s + ": %{y}<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_xaxes(type="category", showgrid=False)
    fig.update_yaxes(gridcolor=GRID)
    return _style(fig, height=320, legend=True)


def hotspot_bar(df, seg, min_reviews=3, top=10):
    """Which outlet / city / dish has the highest % of negative reviews."""
    if seg not in df.columns:
        return None
    g = df.groupby(seg).agg(n=("sentiment", "size"), neg=("sentiment", lambda s: int((s == "negative").sum())))
    g = g[g["n"] >= min_reviews]
    if g.empty:
        return None
    g["pct"] = (g["neg"] / g["n"] * 100).round().astype(int)
    g = g.sort_values(["pct", "n"], ascending=False).head(top)
    fig = go.Figure(go.Bar(
        y=[str(i) for i in g.index], x=g["pct"], orientation="h", marker_color=SENT_COLOR["negative"],
        text=[f"{p}% of {n}" for p, n in zip(g["pct"], g["n"])], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x}% negative<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, 115], ticksuffix="%")
    return _style(fig, height=max(180, 44 * len(g) + 50))


def language_bar(df):
    c = df["language"].astype(str).value_counts()
    if len(c) < 2:
        return None
    return _hbar(list(c.index), c.values, ["#232F3E"] * len(c))


def rating_vs_sentiment(df):
    r = pd.to_numeric(df["rating"], errors="coerce").round()
    d = df[r.notna()].assign(_r=r[r.notna()].astype(int))
    if d["_r"].nunique() < 2:
        return None
    pv = pd.crosstab(d["_r"], d["sentiment"]).reindex(columns=SENT_ORDER, fill_value=0).sort_index()
    fig = go.Figure()
    for s in SENT_ORDER:
        fig.add_bar(x=[f"{i}★" for i in pv.index], y=pv[s], name=s.capitalize(), marker_color=SENT_COLOR[s],
                    marker_line=dict(color="white", width=2),
                    hovertemplate="%{x} · " + s + ": %{y}<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_yaxes(gridcolor=GRID)
    return _style(fig, height=320, legend=True)


# ------------------------------------------------ Data Insights mode charts ----
def avg_rating_by(g, overall):
    """Average rating per group, worst first. Red = clearly below overall, green = clearly above."""
    g = g.head(15)
    # only gaps of 0.2 stars or more are coloured; smaller gaps are noise, so they stay grey
    colors = [SENT_COLOR["negative"] if v < overall - 0.2 else SENT_COLOR["positive"] if v > overall + 0.2
              else SENT_COLOR["neutral"] for v in g["avg"]]
    fig = go.Figure(go.Bar(
        y=[str(i) for i in g.index], x=g["avg"], orientation="h", marker_color=colors,
        text=[f"{a:.2f}  (n={n})" for a, n in zip(g["avg"], g["n"])], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x} avg rating<extra></extra>"))
    fig.add_vline(x=overall, line_dash="dot", line_color=INK, annotation_text=f"overall {overall}",
                  annotation_position="top")
    fig.update_xaxes(range=[0, 5.8])
    fig.update_yaxes(autorange="reversed")
    _style(fig, height=max(200, 44 * len(g) + 80))
    fig.update_layout(margin=dict(l=10, r=30, t=40, b=10))
    return fig


def count_by(g):
    g = g.sort_values("n", ascending=False).head(15)
    return _hbar([str(i) for i in g.index], g["n"].values, ["#232F3E"] * len(g))


def rating_hist(dist):
    fig = go.Figure(go.Bar(x=[f"{i}★" for i in dist.index], y=dist.values, marker_color="#232F3E",
                           text=dist.values, textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}: %{y} ratings<extra></extra>"))
    fig.update_yaxes(gridcolor=GRID)
    return _style(fig, height=300)


def metric_bars(g, overall):
    fig = go.Figure(go.Bar(x=list(g.index), y=g["avg"], marker_color="#2A78D6",
                           text=[f"{a:.2f}" for a in g["avg"]], textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}: %{y} avg rating<extra></extra>"))
    fig.add_hline(y=overall, line_dash="dot", line_color=INK)
    fig.update_yaxes(range=[0, 5.8], gridcolor=GRID)
    return _style(fig, height=300)


def monthly_line(g):
    fig = go.Figure(go.Scatter(x=list(g.index), y=g["avg"], mode="lines+markers",
                               line=dict(color="#2A78D6", width=2), marker=dict(size=8),
                               hovertemplate="%{x}: %{y} avg rating<extra></extra>"))
    fig.update_xaxes(type="category")
    fig.update_yaxes(gridcolor=GRID, title="Average rating")
    return _style(fig, height=300)
