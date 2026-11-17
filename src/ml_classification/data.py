from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from . import config

def load_data():
    X, y = load_wine(return_X_y=True)
    print("Data shape:", X.shape, "| Classes:", set(y))
    return X, y


def split_data(X, y):
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )