from sklearn.metrics import classification_report, accuracy_score


def print_cv_result(name, search):
    print(f"\n{name}")
    print("  Best settings:", search.best_params_)
    print(f"  CV accuracy:   {search.best_score_:.3f}")


def final_test(name, search, X_test, y_test):
    y_pred = search.predict(X_test)
    print(f"\nBest model by cross-validation: {name}")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}\n")
    print(classification_report(y_test, y_pred))