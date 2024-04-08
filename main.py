import numpy as np
from app import run_app

if __name__ == "__main__":
    linspace = np.linspace(0, 10, 1000)
    signal = np.sin(linspace)
    run_app(linspace, signal)
