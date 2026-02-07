from sklearn.model_selection import ParameterSampler

from src.config import SEED


ParameterSampler(param_distributions={}, n_iter=10, random_state=SEED)