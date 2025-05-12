def plot_results(y_test, y_pred, feature_names, model):
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()

    # Feature Importance (Top 20)
    if hasattr(model, 'coef_'):
        coefficients = pd.Series(model.coef_[0], index=feature_names)
        plt.figure(figsize=(10,6))
        coefficients.sort_values(ascending=False)[:20].plot(kind='bar')
        plt.title('Top 20 Important Features')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()