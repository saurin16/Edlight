import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Knowledge base categories
knowledge_base = [
    "Category 1 - Login Issues - Login issues often occur due to incorrect passwords or account lockouts.",
    "Category 2 - App Functionality - App crashes can be caused by outdated software or device incompatibility.",
    "Category 3 - Billing - Billing discrepancies may result from processing errors or duplicate transactions.",
    "Category 4 - Account Management - Account management includes tasks such as changing profile information, linking social media accounts, and managing privacy settings.",
    "Category 5 - Performance Issues - Performance issues can be related to device specifications, network connectivity, or app optimization."
]

# Function to get text embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Precompute embeddings for the knowledge base
knowledge_base_embeddings = np.array([get_embedding(cat) for cat in knowledge_base])

# Function to retrieve the most relevant category
def retrieve_most_relevant(query_embedding, top_k=1):
    similarities = cosine_similarity([query_embedding], knowledge_base_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [knowledge_base[i] for i in top_indices]

# Function to classify a ticket
def classify_ticket(ticket_text):
    ticket_embedding = get_embedding(ticket_text)
    relevant_category = retrieve_most_relevant(ticket_embedding)[0]
    # Extract only the category number and title
    return ' - '.join(relevant_category.split(' - ')[:2])

# Streamlit UI
st.title("Support Ticket Classifier")

ticket_text = st.text_area("Enter the support ticket text:")
if st.button("Classify Ticket"):
    if ticket_text:
        classification = classify_ticket(ticket_text)
        st.write(f"Predicted Classification: {classification}")
    else:
        st.write("Please enter a support ticket text.")

if st.checkbox("Evaluate Classifier"):
    test_tickets = [
        {"text": "I can't log in to my account. It says my password is incorrect.", "label": "Category 1 - Login Issues"},
        {"text": "The app keeps crashing when I open it on my iPhone.", "label": "Category 2 - App Functionality"},
        {"text": "I was charged $19.99 twice this month for my subscription.", "label": "Category 3 - Billing"},
        {"text": "How do I change my profile picture? I can't find the option.", "label": "Category 4 - Account Management"},
        {"text": "Videos are buffering constantly and the quality is poor.", "label": "Category 5 - Performance Issues"}
    ]
    
    correct = 0
    total = len(test_tickets)
    
    for ticket in test_tickets:
        predicted_label = classify_ticket(ticket["text"])
        st.write(f"Ticket: {ticket['text']}\nExpected: {ticket['label']}\nPredicted: {predicted_label}\n")
        if predicted_label == ticket["label"]:
            correct += 1
    
    accuracy = correct / total
    st.write(f"Accuracy: {accuracy:.2f}")
