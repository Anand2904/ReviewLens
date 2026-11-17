"""Test Step 2 from the terminal:  python run_step2.py [csv_path]"""
import sys

import pandas as pd

from analyzer import analyse_reviews

path = sys.argv[1] if len(sys.argv) > 1 else "data/reviews_week40.csv"
out = analyse_reviews(pd.read_csv(path))
out.to_csv("data/analysed_week40.csv", index=False)
cols = ["review_id", "language", "sentiment", "category", "urgency", "summary"]
print(out[cols].to_string(index=False))
print("\nSaved data/analysed_week40.csv")
