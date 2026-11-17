"""Terminal test of the report:  python run_step3.py   (run run_step2.py first)"""
import pandas as pd

from report import compute_stats, generate_insights, render_report

df = pd.read_csv("data/analysed_week40.csv")
stats = compute_stats(df)
insights = generate_insights(df)
text = render_report(stats, insights)

with open("data/weekly_report.txt", "w") as f:
    f.write(text)
print(text)
print("\nSaved data/weekly_report.txt")