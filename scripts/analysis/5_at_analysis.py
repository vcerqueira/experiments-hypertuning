from pathlib import Path

import pandas as pd

N_TRIALS = [5, 25, 100]

MODEL_LIST = [
    'KAN',
    'MLP',
    'NHITS',
    'TFT',
    'PatchTST',
    'GRU',
]

RESULTS_DIR = Path('assets/results')

nt_scores = []
for n_trials in N_TRIALS:
    model_scores = []
    for model in MODEL_LIST:
        df_model = pd.read_csv(RESULTS_DIR / f'search,{n_trials},{model}.csv', index_col='Dataset')
        df_model = df_model.drop(columns=['AT']).rename(columns={'ATR': 'AT'})

        scr = df_model.mean()
        # scr = df.median()
        # scr = df.rank(axis=1).mean()

        data = {**scr.to_dict(), 'k': n_trials, 'model': model}

        model_scores.append(data)

    df = pd.DataFrame(model_scores).set_index(['k', 'model'])
    df.loc[(n_trials, 'Avg.'), :] = df.mean()
    df.loc[(n_trials, 'Avg. R'), :] = df.rank(axis=1).mean()

    nt_scores.append(df)

fdf = pd.concat(nt_scores)

print(fdf)
print(fdf.to_latex(caption='cap',
                   label='tab:search',
                   float_format="{:.3f}".format))
