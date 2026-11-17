from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from . import config

PCA_GRID = [0.90, 0.95]


def build_pipeline(clf):
    """scale -> PCA -> model. Everything is learned inside each CV fold,
    so there is no data leakage."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("clf", clf),
    ])


def get_models():
    """Returns {name: (model, grid)}. Add a new model by adding an entry."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=5000),
            {"pca__n_components": PCA_GRID,
             "clf__C": [0.1, 1, 10]},
        ),
        "SVM (RBF)": (
            SVC(kernel="rbf"),
            {"pca__n_components": PCA_GRID,
             "clf__C": [0.1, 1, 10],
             "clf__gamma": ["scale", 0.01]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=config.RANDOM_STATE),
            {"pca__n_components": PCA_GRID,
             "clf__n_estimators": [100, 300],
             "clf__max_depth": [None, 5]},
        ),
    }