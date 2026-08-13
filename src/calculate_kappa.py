import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Load annotated files
df = pd.read_csv("data/mizolingo_dataset.csv")

# Compute Cohen's Kappa score to verify labeling alignment 
kappa = cohen_kappa_score(df['annotator_a'], df['annotator_b'])

print("==============================================")
print(f"📉 MizoLingo Inter-Annotator Agreement (Cohen's Kappa): {kappa:.3f}")
print("==============================================")
