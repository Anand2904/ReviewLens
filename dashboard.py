"""The results dashboard: KPI tiles, auto takeaways, chart tabs, HIGH-urgency filter,
weekly report and the full table. app.py just calls render_dashboard()."""
import pandas as pd
import streamlit as st

import charts
import insights
from report import compute_stats, generate_insights, render_report
from schemas import ANALYSIS_COLUMNS, STANDARD_COLUMNS

ORDERED = ["review_id", "date", "platform", "rating", "sentiment", "category", "urgency",
           "review_text", "summary", "suggested_reply", "language"]
assert set(ORDERED) == set(STANDARD_COLUMNS + ANALYSIS_COLUMNS)

COLUMN_CONFIG = {
    "review_id": st.column_config.TextColumn("ID", width="small"),
    "date": st.column_config.TextColumn(width="small"),
    "platform": st.column_config.TextColumn(width="small"),
    "rating": st.column_config.NumberColumn(width="small"),
    "review_text": st.column_config.TextColumn("Review", width="large"),
    "summary": st.column_config.TextColumn(width="large"),
    "suggested_reply": st.column_config.TextColumn("Suggested reply", width="large"),
}


def _plot(fig, key):
    if fig is not None:
        st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})


def _csv(df):
    # utf-8-sig adds a small marker so Excel shows Tamil/Hindi letters correctly
    return df.to_csv(index=False).encode("utf-8-sig")


def takeaways(df: pd.DataFrame) -> list:
    """Plain-English findings worked out with exact pandas maths (no AI guessing)."""
    out, n = [], len(df)
    neg = df[df["sentiment"] == "negative"]
    out.append(f"**{len(neg)} of {n} reviews ({len(neg) / n:.0%}) are negative.**")
    if len(neg):
        top = neg["category"].value_counts()
        out.append(f"Most complaints are about **{top.index[0].replace('_', ' ')}** "
                   f"({top.iloc[0]} reviews, {top.iloc[0] / len(neg):.0%} of all complaints).")
    high = int((df["urgency"] == "high").sum())
    if high:
        out.append(f"**{high} reviews are HIGH urgency** and need a reply today. See the 🚨 Urgent tab.")
    if df["platform"].nunique() > 1:
        g = df.groupby("platform")["sentiment"].agg(n="size", neg=lambda s: (s == "negative").mean())
        g = g[g["n"] >= 3]
        if len(g) > 1:
            w, b = g["neg"].idxmax(), g["neg"].idxmin()
            out.append(f"**{w}** has the highest share of negative reviews ({g.loc[w, 'neg']:.0%}); "
                       f"**{b}** the lowest ({g.loc[b, 'neg']:.0%}).")
    r = pd.to_numeric(df["rating"], errors="coerce")
    if r.notna().any():
        mismatch = int((((r >= 4) & (df["sentiment"] == "negative")) |
                        ((r <= 2) & (df["sentiment"] == "positive"))).sum())
        if mismatch:
            out.append(f"{mismatch} reviews have a star rating that contradicts the written text. "
                       "Worth a manual look.")
    return out


def render_dashboard(result: pd.DataFrame, segments: list):
    segments = [s for s in (segments or []) if s in result.columns]
    stats = compute_stats(result, segments)

    # ---- KPI tiles -----------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Reviews analysed", stats["total"])
    k2.metric("Negative", f"{stats['negative']}%")
    k3.metric("HIGH urgency", stats["urgent"])
    k4.metric("Average rating", "n/a" if stats["avg_rating"] is None else f"{stats['avg_rating']} / 5")
    with st.container(border=True):
        st.markdown("#### What stands out")
        for line in takeaways(result):
            st.markdown(f"- {line}")

    t_over, t_wrong, t_trend, t_urgent, t_report, t_all = st.tabs(
        ["📊 Overview", "🔎 What's going wrong", "📅 Trends & places", "🚨 Urgent",
         "📝 Weekly report", "📋 All reviews"])

    # ---- Overview ------------------------------------------------------------
    with t_over:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Sentiment split**")
            st.caption("How customers feel overall.")
            _plot(charts.sentiment_bar(result), "c_sent")
        with c2:
            st.markdown("**Reviews by category**")
            st.caption("What customers talk about most.")
            _plot(charts.category_bar(result), "c_cat")
        with c3:
            st.markdown("**Urgency**")
            st.caption("How many need a reply today.")
            _plot(charts.urgency_bar(result), "c_urg")

    # ---- What's going wrong --------------------------------------------------
    with t_wrong:
        neg = result[result["sentiment"] == "negative"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Complaints by category** (negative reviews only)")
            st.caption("Which department to fix first.")
            if len(neg):
                _plot(charts.category_bar(neg), "c_negcat")
            else:
                st.info("No negative reviews in this data.")
        with c2:
            st.markdown("**Sentiment inside each category**")
            st.caption("A big category with lots of red is a real problem area.")
            _plot(charts.stacked_by(result, "category"), "c_catsent")
        c3, c4 = st.columns(2)
        with c3:
            if result["platform"].nunique() > 1:
                st.markdown("**Sentiment by platform**")
                st.caption("Is one platform (Zomato, Swiggy, Google) unhappier than the others?")
                _plot(charts.stacked_by(result, "platform"), "c_platsent")
        with c4:
            fig = charts.rating_vs_sentiment(result)
            if fig is not None:
                st.markdown("**Star rating vs what customers wrote**")
                st.caption("Red on a 4★ or 5★ bar, or green on a 1★ bar, means the stars hide the real story.")
                _plot(fig, "c_rating")

    # ---- Trends & places -----------------------------------------------------
    with t_trend:
        fig = charts.daily_trend(result)
        if fig is not None:
            st.markdown("**Reviews per day, by sentiment**")
            st.caption("Spot the day things went wrong.")
            _plot(fig, "c_daily")
        else:
            st.info("Not enough dates in this file for a trend chart.")
        shown = False
        for i, seg in enumerate(segments):
            fig = charts.hotspot_bar(result, seg)
            if fig is not None:
                shown = True
                st.markdown(f"**Hotspots by {seg.replace('_', ' ')}**")
                st.caption("Share of reviews that are negative (only groups with 3+ reviews).")
                _plot(fig, f"c_hot_{i}")
        if not shown and segments == []:
            st.caption("Tip: choose a 'Group by' column such as branch or city in the column panel "
                       "to see which place has the most complaints.")
        fig = charts.language_bar(result)
        if fig is not None:
            st.markdown("**Languages in the reviews**")
            st.caption("The AI understood and summarised all of these in English.")
            _plot(fig, "c_lang")

    # ---- Urgent --------------------------------------------------------------
    with t_urgent:
        f1, f2, f3 = st.columns([1, 2, 2])
        only_high = f1.toggle("Only HIGH urgency", value=True)
        cats = f2.multiselect("Category", sorted(result["category"].unique()))
        plats = f3.multiselect("Platform", sorted(result["platform"].astype(str).unique()))
        view = result[result["urgency"] == "high"] if only_high else result
        if cats:
            view = view[view["category"].isin(cats)]
        if plats:
            view = view[view["platform"].astype(str).isin(plats)]
        rank = {"high": 0, "medium": 1, "low": 2}
        view = view.sort_values("urgency", key=lambda s: s.map(rank), kind="stable")
        st.write(f"**{len(view)} reviews shown.**")
        if only_high and len(view):
            st.caption("Reply to these today. The suggested reply is ready to copy.")
            _plot(charts.category_bar(view), "c_urgcat")
        st.dataframe(view[ORDERED], width="stretch", hide_index=True, column_config=COLUMN_CONFIG)
        st.download_button("⬇️ Download this list (CSV)", _csv(view[ORDERED]),
                           file_name="urgent_reviews.csv", mime="text/csv", key="dl_urgent")

    # ---- Weekly report -------------------------------------------------------
    with t_report:
        st.caption("A second AI step reads every result and writes the top complaints, praises and "
                   "3 action items. The numbers are counted by code, not guessed by the AI.")
        if st.button("Generate weekly report", key="btn_report"):
            try:
                with st.spinner("Reading all reviews and writing the report..."):
                    ins = generate_insights(result, segments=segments)
                    st.session_state["report_text"] = render_report(stats, ins)
            except Exception as e:
                st.error(f"Report failed. Is your OPENAI_API_KEY set?  Details: {e}")
        if "report_text" in st.session_state:
            st.code(st.session_state["report_text"], language=None)
            st.download_button("⬇️ Download report (TXT)", st.session_state["report_text"].encode("utf-8"),
                               file_name="weekly_report.txt", mime="text/plain", key="dl_report")

    # ---- All reviews ---------------------------------------------------------
    with t_all:
        show_orig = st.checkbox("Also show / include my original extra columns", value=False)
        shown = result if show_orig else result[ORDERED]
        st.dataframe(shown, width="stretch", hide_index=True, column_config=COLUMN_CONFIG)
        st.download_button("⬇️ Download enriched CSV (opens in Excel)", _csv(shown),
                           file_name="reviews_analysed.csv", mime="text/csv", key="dl_all")


# ======================= Data Insights mode (files with NO written feedback) =========
def render_data_insights(raw: pd.DataFrame, rating_col, date_col, id_col=None):
    st.info("**Data Insights mode.** This file has no written customer feedback, so there is no text for "
            "the AI to read. I'll analyse the ratings and numbers instead. If a comments column does exist, "
            "pick it under *Review text* above for the full review analysis.")
    cols = list(raw.columns)
    exclude = {c for c in (rating_col, date_col) if c}
    c1, c2 = st.columns(2)
    dims = c1.multiselect("Compare groups by", cols, default=insights.suggest_dims(raw, exclude),
                          max_selections=4, key="di_dims")
    num = insights.numeric_columns(raw, exclude)
    metrics = []
    if rating_col:
        metrics = c2.multiselect("Numbers to check against rating", num, default=num[:3], key="di_metrics")
    else:
        st.warning("No rating column chosen, so only counts per group are shown. "
                   "Pick one in the *Rating* box above for much richer insights.")
    t = insights.compute_tables(raw, rating_col, dims, metrics, date_col)

    k1, k2, k3 = st.columns(3)
    k1.metric("Rows", f"{t['rows']:,}")
    k2.metric("Average rating", "n/a" if t["overall"] is None else f"{t['overall']} / 5")
    k3.metric("Low ratings (2 or below)", "n/a" if t["low_pct"] is None else f"{t['low_pct']}%")

    if t["rating_dist"] is not None:
        st.markdown("**How ratings are spread**")
        _plot(charts.rating_hist(t["rating_dist"]), "di_hist")
    for i, (dim, g) in enumerate(t["dims"].items()):
        st.markdown(f"**By {dim.replace('_', ' ')}**")
        if t["has_rating"]:
            st.caption(f"Average rating per group, worst first (red = at least 0.2 below the overall "
                       f"{t['overall']}, green = at least 0.2 above, grey = no real difference). Groups under {t['min_n']} rows are hidden.")
            _plot(charts.avg_rating_by(g, t["overall"]), f"di_avg_{i}")
        else:
            _plot(charts.count_by(g), f"di_cnt_{i}")
    for i, (m, g) in enumerate(t["metrics"].items()):
        st.markdown(f"**Average rating by {m.replace('_', ' ')}** (split into 4 equal-sized ranges)")
        _plot(charts.metric_bars(g, t["overall"]), f"di_met_{i}")
    if t["trend"] is not None:
        st.markdown("**Average rating by month**")
        _plot(charts.monthly_line(t["trend"]), "di_trend")

    # ---- weekly report in the same shape as the text-review report ----------------
    st.markdown("---")
    st.markdown("#### Weekly report")
    if not rating_col:
        st.caption("Pick a rating column above to create this report.")
    else:
        st.caption("Same layout as the review report. Sentiment comes from the star rating "
                   "(4-5 positive, 3 neutral, 1-2 negative); problem areas are counted by code.")
        sub, dts = raw, None
        if date_col:       # read the dates ONCE from the full file, then slice
            dts = pd.to_datetime(pd.Series(insights._parse_dates(raw[date_col]), index=raw.index), errors="coerce")
            if dts.notna().any():
                lo, hi = dts.min().date(), dts.max().date()
                rng = st.date_input("Report period (choose one week for a weekly report)", value=(lo, hi),
                                    min_value=lo, max_value=hi, key="di_period")
                if isinstance(rng, (tuple, list)) and len(rng) == 2:
                    sub = raw[(dts >= pd.Timestamp(rng[0])) & (dts < pd.Timestamp(rng[1]) + pd.Timedelta(days=1))]
            else:
                dts = None
        if st.button("Generate weekly report", key="btn_dreport"):
            try:
                with st.spinner("Counting and writing the report..."):
                    stats_ = insights.build_data_report(sub, rating_col, dims, metrics, dts)
                    acts = insights.generate_report_actions(stats_)
                    st.session_state["di_report"] = insights.render_data_report(stats_, acts)
            except Exception as e:
                st.error(f"Report failed: {e}")
        if "di_report" in st.session_state:
            st.code(st.session_state["di_report"], language=None)
            st.download_button("⬇️ Download report (TXT)", st.session_state["di_report"].encode("utf-8"),
                               file_name="weekly_report.txt", mime="text/plain", key="dl_dreport")

    st.markdown("---")
    st.markdown("#### AI insights")
    st.caption("The AI reads the statistics above (all counted by code) and explains them in plain English.")
    if st.button("Write AI insights", key="btn_di"):
        try:
            with st.spinner("Reading the statistics..."):
                st.session_state["di_text"] = insights.generate_data_insights(t)
        except Exception as e:
            st.error(f"Could not write insights. Is your OPENAI_API_KEY set?  Details: {e}")
    r = st.session_state.get("di_text")
    if r is not None:
        st.success(r.headline)
        st.markdown("**Findings**")
        for f in r.findings:
            st.markdown(f"- {f}")
        st.markdown("**Suggested actions**")
        for n, a in enumerate(r.action_items, 1):
            st.markdown(f"{n}. {a}")
        with st.container(border=True):
            st.markdown("**Keep in mind**")
            for c in r.caveats:
                st.markdown(f"- {c}")

    # ---- follow-up messages for low-rated orders ---------------------------------
    st.markdown("---")
    st.markdown("#### Follow-up messages for low-rated orders")
    if not rating_col:
        st.caption("Pick a rating column above to use this.")
        return
    st.caption("There are no comments, so the AI cannot know what went wrong. It writes a polite, general "
               "message that apologises and asks the customer to tell you what happened. "
               "It only sees the facts you tick below.")
    f1, f2 = st.columns(2)
    max_r = f1.selectbox("Write to customers who rated", [1, 2, 3], index=1,
                         format_func=lambda x: f"{x} star{'s' if x > 1 else ''} or below", key="fu_max")
    limit = f2.number_input("How many messages (fewer = cheaper while testing)", 1, 200, 25, key="fu_n")
    options = [c for c in raw.columns if c not in (rating_col, id_col)]
    sensitive = {"gender", "age"}
    default_ctx = [c for c in list(t["dims"]) + list(t["metrics"]) if c in options
                   and not (insights._tokens(c) & sensitive)][:4]
    ctx = st.multiselect("Facts the AI may mention", options, default=default_ctx, max_selections=6, key="fu_ctx")
    n_low = insights.count_low(raw, rating_col, max_r)
    st.write(f"**{n_low} orders** have a rating of {max_r} or below.")
    if n_low and st.button("Write follow-up messages", key="btn_fu"):
        try:
            with st.spinner(f"Writing {min(int(limit), n_low)} messages..."):
                st.session_state["di_followups"] = insights.generate_followups(
                    raw, rating_col, id_col, ctx, max_r, int(limit))
        except Exception as e:
            st.error(f"Could not write messages. Is your OPENAI_API_KEY set?  Details: {e}")
    fu = st.session_state.get("di_followups")
    if fu is not None:
        table, total = fu
        st.write(f"Showing **{len(table)}** of {total} low-rated orders (lowest ratings first). "
                 "Read each message before sending.")
        st.dataframe(table, width="stretch", hide_index=True,
                     column_config={"follow_up_message": st.column_config.TextColumn("Follow-up message", width="large")})
        st.download_button("⬇️ Download messages (CSV)", _csv(table), file_name="follow_up_messages.csv",
                           mime="text/csv", key="dl_fu")
