#!/usr/bin/env python
# coding: utf-8

# In[2]:


# !pip install streamlit
import streamlit as st
import pickle

# Load the model, vectorizer, and SVD
with open('modelFinal.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('svd.pkl', 'rb') as svd_file:
    svd = pickle.load(svd_file)

# Define a function to preprocess and predict review sentiment
def predict_sentiment(review_text):
    # Transform the review using the vectorizer
    X_new = vectorizer.transform([review_text])
    # Apply SVD transformation
    X_new = svd.transform(X_new)
    # Make prediction
    prediction = model.predict(X_new)[0]
    #Map prediction to label
    labels = {0: 'Negative', 1: 'Positive'}
    return labels[prediction]

# Streamlit application
def main():
    st.title('Hotel Review Sentiment Classifier')

    st.write("Enter a hotel review below to classify its sentiment as 'Positive' or 'Negative'.")

    # Text input for user review
    review_text = st.text_area("Review Text:")

    if st.button('Classify'):
        if review_text:
            # Predict sentiment
            sentiment = predict_sentiment(review_text)
            st.write(f"Predicted Sentiment: {sentiment}")
        else:
            st.write('Please enter a review.')

if __name__ == '__main__':
    main()


# In[ ]:




