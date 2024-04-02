import cupy as cp

def seed_cupy(seed=1000):
    cp.random.seed(seed)