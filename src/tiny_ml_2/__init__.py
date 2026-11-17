# simple_ml.py
# A minimal end-to-end classification example:
# split -> scale -> PCA -> tune with cross-validation -> test

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def main():
    # 1. Load the data (X = features, y = labels)
    X, y = load_wine(return_X_y=True)
    print("Data shape:", X.shape, "| Classes:", set(y))

    # 2. Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 3. The models to try, each with its own "dials" (hyperparameters).
    #    "clf__" means "a setting of the model step in the pipeline".
    #    "pca__" means "a setting of the PCA step".
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=5000),
            {"pca__n_components": [0.90, 0.95],
             "clf__C": [0.1, 1, 10]},
        ),
        "SVM (RBF)": (
            SVC(kernel="rbf"),
            {"pca__n_components": [0.90, 0.95],
             "clf__C": [0.1, 1, 10],
             "clf__gamma": ["scale", 0.01]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {"pca__n_components": [0.90, 0.95],
             "clf__n_estimators": [100, 300],
             "clf__max_depth": [None, 5]},
        ),
    }

    best_name, best_cv_score, best_search = None, -1, None

    for name, (clf, grid) in models.items():
        # 4. Chain: scale -> PCA -> model.
        #    Inside a Pipeline, scaling and PCA are learned only from the
        #    training part of each CV round, so there is no data leakage.
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA()),
            ("clf", clf),
        ])

        # 5. Try every combination of dials using 5-fold cross-validation
        search = GridSearchCV(pipe, grid, cv=5, scoring="accuracy", n_jobs=-1)
        search.fit(X_train, y_train)

        print(f"\n{name}")
        print("  Best settings:", search.best_params_)
        print(f"  CV accuracy:   {search.best_score_:.3f}")

        if search.best_score_ > best_cv_score:
            best_name, best_cv_score, best_search = name, search.best_score_, search

    # 6. Pick the winner using CV scores, then test it ONCE on the unseen test set
    print(f"\nBest model by cross-validation: {best_name}")
    y_pred = best_search.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}\n")
    print(classification_report(y_test, y_pred))
