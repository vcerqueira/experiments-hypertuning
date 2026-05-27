import pandas as pd

from src.config import RESULTS_DIR

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# results_dir = Path() / 'assets' / 'results'
# results_dir = Path().resolve().parent / 'hypertuning-files' / 'results-all-compiled'
print(RESULTS_DIR.absolute())

pattern = f"*,*,*,outer.csv"
config_files = list(RESULTS_DIR.glob(pattern))

model_target_pairs = [(f.stem.split(',')[1], f.stem.split(',')[0]) for f in config_files]
df = pd.DataFrame(model_target_pairs, columns=['ds', 'model'])
pair_counts = df.value_counts().sort_values(ascending=False).sort_index()

print(pair_counts)
