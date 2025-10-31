import numpy as np

def fundamental_to_essential(F, K1, K2=None):
    if K2 is None:
        K2 = K1
    E = K2.T @ F @ K1
    return E


def