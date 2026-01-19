import warnings

import matplotlib.pyplot as plt
import numpy as np


def configure_environment(seed: int = 42) -> None:
    """Configure warnings, random seed, and plotting style."""
    warnings.filterwarnings('ignore')
    np.random.seed(seed)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'figure.figsize': (10, 6),
    })
