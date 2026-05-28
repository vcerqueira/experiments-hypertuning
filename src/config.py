from pathlib import Path

DRY_RUN = False

RESULTS_DIR = Path().resolve().parent.parent / 'hypertuning-files' / 'results-all-compiled'

SEED = 123
TRY_MPS = False
if DRY_RUN:
    LIMIT_EPOCHS = True
    N_SAMPLES = 100
    MAX_SAMPLES = 50
else:
    LIMIT_EPOCHS = False
    N_SAMPLES = 3000
    MAX_SAMPLES = 500
