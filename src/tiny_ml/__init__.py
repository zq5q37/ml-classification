from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

def main() -> None:
    # 1. Load data
    X, y = load_wine(return_X_y=True)

    # 2. Split: 80% to learn from, 20% to test on
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Chain of steps: scale -> PCA -> SVM
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=5)),
        ("svm", SVC()), #"svm" is just a label you choose
    ])

    # Debug statement to find what parameters to tune
    print(SVC().get_params())

    # For everything in pipe, gives valid keys for your grid.
    print(pipe.get_params().keys())

    # 4. Dials to try for the SVM
    # The grid key is label__parameter (two underscores):
    # "svm__C" means the C setting of the step labelled svm
    grid = {"svm__C": [0.1, 1, 10]} # try these values

    # 5. Try each dial value with 5-fold cross-validation
    search = GridSearchCV(pipe, grid, cv=5)
    search.fit(X_train, y_train)

    # 6. Results
    print("Best C:", search.best_params_)
    print("CV score:", search.best_score_)
    print("Test score:", search.score(X_test, y_test))
