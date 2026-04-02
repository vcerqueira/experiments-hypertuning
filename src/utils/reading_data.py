import pandas as pd


def read_cv_results(dir, model, dataset_name, partition):
    pattern = f"{model},{dataset_name},*,{partition}.csv"
    key_cols = ['unique_id', 'ds', 'cutoff']

    config_files = list(dir.glob(pattern))
    config_ids = [f.stem.split(',')[2] for f in config_files]

    cv_df = None
    for file in config_files:
        config_id = file.stem.split(',')[2]
        cv_file = pd.read_csv(file).rename(columns={model: config_id})
        if cv_df is None:
            cv_df = cv_file
        else:
            cv_df = cv_df.merge(cv_file.drop(columns=['y']), on=key_cols, how='inner')

    return cv_df, config_ids
