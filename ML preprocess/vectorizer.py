from sklearn.feature_extraction.text import TfidfVectorizer

corpus = (file_path)

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())