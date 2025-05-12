def extract_features(df):
    # Basic text features
    df['char_count'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['exclamation_count'] = df['text'].apply(lambda x: x.count('!'))
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
    tfidf_features = tfidf.fit_transform(df['clean_text'])
    
    return tfidf_features, df[['char_count', 'word_count', 'exclamation_count']], tfidf