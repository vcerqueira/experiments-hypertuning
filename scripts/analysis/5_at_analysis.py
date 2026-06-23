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
    'Informer'
]

RESULTS_DIR = Path('assets/results')

nt_scores = []
for n_trials in N_TRIALS:
    model_scores = []
    for model in MODEL_LIST:
        df_model = pd.read_csv(RESULTS_DIR / f'search,{n_trials},{model}.csv', index_col='Dataset')

        scr = df_model.mean()

        data = {**scr.to_dict(), 'k': n_trials, 'model': model}

        model_scores.append(data)

    df = pd.DataFrame(model_scores).set_index(['k', 'model'])

    mlp_rs = df.loc[(n_trials, 'MLP'), 'RS']
    mlp_at = df.loc[(n_trials, 'MLP'), 'AT']
    df['RS_diff'] = df['RS'] - mlp_rs
    # df['RS_diff'] = 100*((df['RS'] - mlp_rs)/mlp_rs)
    df['AT_diff'] = df['AT'] - mlp_at
    # df['AT_diff'] = 100*((df['AT'] - mlp_at)/mlp_at)

    avg_r = df[['RS', 'AT']].rank(axis=1).mean()
    df.loc[(n_trials, 'Avg.'), ['RS', 'AT']] = df[['RS', 'AT']].mean()
    df.loc[(n_trials, 'Avg.'), ['RS_diff', 'AT_diff']] = df[['RS_diff', 'AT_diff']].mean()
    df.loc[(n_trials, 'Avg. R'), ['RS', 'AT']] = avg_r

    nt_scores.append(df)

fdf = pd.concat(nt_scores)
fdf[['RS', 'AT']] = fdf[['RS', 'AT']].round(3)
fdf[['RS_diff', 'AT_diff']] = fdf[['RS_diff', 'AT_diff']].round(2)

print(fdf)

print(fdf.drop(columns=['RS_diff', 'AT_diff']).to_latex(caption='cap',
                                                        label='tab:search',
                                                        float_format="{:.3f}".format))

latex_df = fdf.copy()
latex_df['RS'] = [
    f'{val:.3f} ({diff:+.2f})' for val, diff in zip(fdf['RS'], fdf['RS_diff'])
]
latex_df['AT'] = [
    f'{val:.3f} ({diff:+.2f})' for val, diff in zip(fdf['AT'], fdf['AT_diff'])
]
latex_df = latex_df[['RS', 'AT']]

print(latex_df.to_latex(caption='cap', label='tab:search'))
