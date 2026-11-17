"""ReviewLens: universal customer-feedback analyser (any CSV, any column names).
Run with:  streamlit run app.py
Flow: upload -> AI works out the columns -> you confirm -> analyse -> table -> download
"""
import streamlit as st

from analyzer import analyse_reviews
from ingest import ColumnMapping, detect_mapping, looks_like_feedback, normalise, read_csv_flexible
from background import inject_background
from dashboard import render_dashboard, render_data_insights

st.set_page_config(page_title="ReviewLens", page_icon="🔍", layout="wide")
inject_background()   # animated food-themed background

# ---- Colour palette (navy / orange / yellow) --------------------------------
AMAZON_CSS = """
<style>
[data-testid="stHeader"] { background: #131921; }
[data-testid="stToolbar"] * { color: #FFFFFF !important; }

.rl-banner {
    background: linear-gradient(90deg, #131921, #232F3E);
    border-bottom: 4px solid #FF9900;
    border-radius: 6px;
    padding: 18px 26px;
    margin-bottom: 18px;
}
.rl-banner h1 { color: #FFFFFF; margin: 0; padding: 0; font-size: 2.1rem; }
.rl-banner p  { color: #FEBD69; margin: 4px 0 0 0; font-size: 1rem; }

div.stButton > button, div.stDownloadButton > button {
    background: #FFD814; color: #0F1111; border: 1px solid #FCD200;
    border-radius: 20px; font-weight: 500;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    background: #F7CA00; border-color: #F2C200; color: #0F1111;
}
div.stDownloadButton > button { background: #FFA41C; border-color: #FF8F00; }
div.stDownloadButton > button:hover { background: #FA8900; border-color: #E47911; }

[data-testid="stFileUploader"] section { background: #FFFFFF; border: 1px dashed #888C8C; }
a { color: #007185 !important; }
</style>
"""
st.markdown(AMAZON_CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="rl-banner"><h1>🔍 ReviewLens</h1>'
    "<p>Upload any restaurant or food-delivery feedback CSV. The AI works out the columns "
    "and turns it into tagged, prioritised insights.</p></div>",
    unsafe_allow_html=True,
)

NONE = "(none)"

# ---- 1. Upload --------------------------------------------------------------
uploaded = st.file_uploader("Upload a customer-feedback CSV (any column names)", type="csv")

if uploaded is not None:
    file_key = (uploaded.name, uploaded.size)
    if st.session_state.get("file_key") != file_key:          # a NEW file: start fresh
        for k in [k for k in st.session_state if k.startswith("sel_")] + ["result", "segments", "report_text", "di_text", "di_followups", "di_report"]:
            st.session_state.pop(k, None)
        try:
            raw = read_csv_flexible(uploaded)
        except Exception as e:
            st.error(f"Could not read this file: {e}")
            st.stop()
        with st.spinner("Reading your file and working out what each column means..."):
            mapping, source = detect_mapping(raw)
        st.session_state.update(file_key=file_key, raw=raw, mapping=mapping, map_source=source)

    raw, ai_map = st.session_state["raw"], st.session_state["mapping"]
    cols = list(raw.columns)
    st.write(f"**{len(raw)} rows, {len(cols)} columns found.**")

    # ---- 2. Confirm the column mapping ---------------------------------------
    with st.expander("🧭 How I understood your file (change anything that looks wrong)", expanded=True):
        who = "the AI" if st.session_state["map_source"] == "ai" else "rule-based guessing"
        st.caption(f"Columns were detected by {who}. {ai_map.notes}")

        def pick(label, current, key, required=False):
            options = ["(choose a column)"] + cols if required else [NONE] + cols
            idx = options.index(current) if current in options else 0
            v = st.selectbox(label, options, index=idx, key=key)
            return None if v in (NONE, "(choose a column)") else v

        c1, c2, c3 = st.columns(3)
        with c1:
            text_col = pick("Review text (required)", ai_map.review_text, "sel_text", required=True)
            rating_col = pick("Rating", ai_map.rating, "sel_rating")
        with c2:
            date_col = pick("Date", ai_map.date, "sel_date")
            platform_col = pick("Platform / source", ai_map.platform, "sel_platform")
        with c3:
            id_col = pick("Review ID", ai_map.review_id, "sel_id")
            seg_cols = st.multiselect("Group by (e.g. branch, city)", cols,
                                      default=ai_map.segments, max_selections=2, key="sel_segs")

    if not text_col:                       # no written feedback anywhere -> numbers-only analysis
        st.session_state.pop("result", None)
        render_data_insights(raw, rating_col, date_col, id_col)
        st.stop()
    if text_col and not looks_like_feedback(raw[text_col]):
        st.warning(f"The column '{text_col}' does not look like written feedback (it looks like ids, "
                   "names or labels). The analysis will not be meaningful. Choose a column with real sentences.")
    picked = [c for c in (text_col, rating_col, date_col, platform_col, id_col) if c]
    if len(picked) != len(set(picked)):
        st.warning("The same column is chosen for two different roles. Please give each role its own column.")
    chosen = ColumnMapping(review_text=text_col, rating=rating_col, date=date_col,
                           platform=platform_col, review_id=id_col, segments=seg_cols)
    try:
        preview, _ = normalise(raw, chosen)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    st.write("**This is what the analyser will see:**")
    st.dataframe(preview.head(5), width="stretch")

    # ---- 3. Analyse ------------------------------------------------------------
    cap = min(len(preview), 1500)
    n_rows = st.number_input(
        f"How many reviews to analyse (max {cap}). Fewer = faster and cheaper while testing.",
        min_value=1, max_value=max(cap, 1), value=min(cap, 300), step=50)

    if st.button("Analyse reviews", type="primary"):
        try:
            table, segs = normalise(raw, chosen, max_rows=int(n_rows))
            with st.spinner(f"Analysing {len(table)} reviews... this can take a minute."):
                # session_state = the app's memory; without it results vanish on every click
                st.session_state["result"] = analyse_reviews(table)
                st.session_state["segments"] = segs
                st.session_state.pop("report_text", None)
        except Exception as e:
            st.error(f"Analysis failed. Is your OPENAI_API_KEY set in .env?  Details: {e}")

# ---- 4. Dashboard -------------------------------------------------------------
if "result" in st.session_state:
    st.success(f"Done. {len(st.session_state['result'])} reviews analysed.")
    render_dashboard(st.session_state["result"], st.session_state.get("segments", []))
