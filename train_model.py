def train_model(X_train, y_train):
    try:
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise