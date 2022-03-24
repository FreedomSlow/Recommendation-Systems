import pandas as pd
from utils import *


if __name__ == '__main__':
    # Test load data
    ml_100 = load_dataset("ml_100k")
    assert (isinstance(ml_100, pd.DataFrame))
    assert (len(ml_100.columns) == 4)
    ml_100.info()
    print(ml_100.head())

    ml_small = load_dataset("ml_small")
    assert (isinstance(ml_small, pd.DataFrame))
    assert (len(ml_small.columns) == 4)
    ml_small.info()
    print(ml_small.head())


