# train_pipeline.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Example training data
descriptions = ["pizza", "uber ride", "groceries", "movie ticket"]
categories = ["Food", "Transport", "Grocery", "Entertainment"]

# Create a pipeline
pipeline = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

# Train the pipeline
pipeline.fit(descriptions, categories)

# Save the trained pipeline
joblib.dump(pipeline, "budgetbee_pipeline.joblib")
print("Pipeline saved as budgetbee_pipeline.joblib")
