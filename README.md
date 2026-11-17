# 🔍 ReviewLens: Customer Feedback Analyzer

A universal AI tool that turns **any restaurant or food-delivery feedback CSV** into tagged, prioritised insights and a weekly report. Column names don't matter: the AI works out which column is which.

Built with **LangChain** (structured LLM output), **OpenAI**, **Streamlit**, **pandas** and **Plotly**.

## What it does

| Your file has... | ReviewLens gives you |
|---|---|
| Written reviews (English, Tamil, Tanglish) | Per review: sentiment, category, urgency, language, one-line English summary and a polite suggested reply |
| Only ratings and order details | **Data Insights mode**: rating-based charts, problem areas, a weekly report and follow-up messages for low-rated orders |

- **Universal ingestion.** The AI maps columns (review text, rating, date, platform, id) from the headers and sample values, with a rule-based fallback. You can correct any choice in dropdowns.
- **Structured output.** Every review goes through a LangChain chain (`prompt | llm.with_structured_output(...)`) that returns a validated Pydantic object, run in parallel batches.
- **Weekly insights report.** A second chain (map-then-reduce) writes top complaints, top praise and 3 action items. Counts are computed by code, never guessed by the AI.
- **Dashboard.** Sentiment split, category chart, urgency, trends, hotspots by restaurant/city, a HIGH-urgency filter, and an enriched CSV export.
- **Animated food-themed background** (pure CSS).

## Run it

```bash
git clone https://github.com/Anand2904/ReviewLens.git
cd ReviewLens
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
cp .env.example .env        # then put your OPENAI_API_KEY inside .env
streamlit run app.py
```

Try it with `data/demo_feedback_synthetic.csv` (300 synthetic reviews).

## Project layout

| File | Role |
|---|---|
| `app.py` | Streamlit page: upload, column mapping, run, dashboard |
| `ingest.py` | Flexible CSV reading, AI column mapping, cleaning and normalising |
| `analyzer.py` | Per-review LangChain chain (sentiment, category, urgency, summary, reply) |
| `report.py` | Weekly insights chain and Output-B style text report |
| `insights.py` | Data Insights mode: rating statistics, report, follow-up messages |
| `charts.py`, `dashboard.py` | Plotly charts and the dashboard layout |
| `background.py` | Animated background |
| `schemas.py`, `llm.py` | Pydantic models and LLM provider setup |

## Notes

- Never commit your `.env` file; it holds your API key (it is in `.gitignore`).
- Sample data in `data/` is synthetic or public test data.
