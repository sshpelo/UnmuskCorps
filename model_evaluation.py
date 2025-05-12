from sklearn.model_selection import cross_val_score

def evaluate_model(model, X, y):
    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")