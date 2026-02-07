DRY_RUN = False

SEED = 123
TRY_MPS = True
if DRY_RUN:
    N_SAMPLES = 2
    LIMIT_EPOCHS = True
else:
    N_SAMPLES = 10
    LIMIT_EPOCHS = False
