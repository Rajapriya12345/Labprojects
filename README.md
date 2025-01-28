import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset of text documents
data = {
    "Document": [
        "Natural Language Processing is a field of AI.",
        "TF-IDF stands for Term Frequency-Inverse Document Frequency.",
        "It is widely used in Information Retrieval and Text Mining.",
        "This is an example of text processing using Python.",
        "Machine Learning and NLP are interconnected fields."
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Step 1: Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Step 2: Fit and transform the text data
tfidf_matrix = tfidf_vectorizer.fit_transform(df["Document"])

# Step 3: Display the TF-IDF matrix as a DataFrame
feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
print("TF-IDF Matrix:")
print(tfidf_df)

# Step 4: Query the TF-IDF model
query = "text mining and NLP"
query_vector = tfidf_vectorizer.transform([query])

# Step 5: Compute cosine similarity between the query and documents
similarity_scores = cosine_similarity(query_vector, tfidf_matrix)

# Step 6: Display the similarity scores
df["Similarity"] = similarity_scores[0]
print("\nDocuments ranked by similarity to the query:")
print(df.sort_values(by="Similarity", ascending=False))
# Labprojects
