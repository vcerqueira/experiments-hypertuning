from pathlib import Path

import pandas as pd

results_dir = Path() / 'assets' / 'results'

pattern = f"*,*,*,outer.csv"
config_files = list(results_dir.glob(pattern))

model_target_pairs = [(f.stem.split(',')[1], f.stem.split(',')[0]) for f in config_files]
df = pd.DataFrame(model_target_pairs, columns=['ds', 'model'])
pair_counts = df.value_counts().sort_values(ascending=False).sort_index()

print(pair_counts)
