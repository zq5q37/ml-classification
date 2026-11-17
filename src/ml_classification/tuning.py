from sklearn.model_selection import GridSearchCV
from .models import build_pipeline
from . import config   


def tune_model(clf, grid, X_train, y_train):
    pipe = build_pipeline(clf)
    search = GridSearchCV(
        pipe, grid,
        cv=config.CV_FOLDS,
        scoring=config.SCORING,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search