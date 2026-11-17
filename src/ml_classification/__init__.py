from .data import load_data, split_data
from .models import get_models
from .tuning import tune_model
from .evaluation import print_cv_result, final_test


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    best_name, best_search = None, None

    for name, (clf, grid) in get_models().items():
        search = tune_model(clf, grid, X_train, y_train)
        print_cv_result(name, search)

        if best_search is None or search.best_score_ > best_search.best_score_:
            best_name, best_search = name, search

    final_test(best_name, best_search, X_test, y_test)


if __name__ == "__main__":
    main()