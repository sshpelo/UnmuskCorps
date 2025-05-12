def main():
    # Load and prepare data
    df = load_data()
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    # Feature extraction
    tfidf_features, manual_features, tfidf_vectorizer = extract_features(df)
    X = np.hstack([tfidf_features.toarray(), manual_features])
    y = df['label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visualize
    feature_names = list(tfidf_vectorizer.get_feature_names_out()) + list(manual_features.columns)
    plot_results(y_test, y_pred, feature_names, model)