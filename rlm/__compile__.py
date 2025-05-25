import json
import random
import nltk
import torch
from textblob import TextBlob
from transformers import pipeline

nltk.download('punkt')

FILE_NAME = "memoryCache32.json"

gpt2 = pipeline("text-generation", model="gpt2")

def load_data():
    try:
        with open(FILE_NAME, "r") as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        return {}

def save_data(question, answer):
    data = load_data()
    data[question] = answer

    with open(FILE_NAME, "w") as file:
        json.dump(data, file)

def sentiment_analysis(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity

    if sentiment_score > 0:
        return "You seem happy! 😊"
    elif sentiment_score < 0:
        return "You seem sad. Can I help? 😞"
    else:
        return "You seem neutral. 🤔"

def find_best_answer(user_input, data):
    if not data:
        return None

    for question, answer in data.items():
        if question.lower() in user_input.lower() or user_input.lower() in question.lower():
            return answer
    return None

def generate_gpt2_response(user_input):
    result = gpt2(user_input, max_length=50, num_return_sequences=1)
    return result[0]['generated_text']

print("Chatbot: Hello! You can teach me something. Type 'exit' to quit.")

data = load_data()

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chatbot: See you later! 👋")
        break

    sentiment_response = sentiment_analysis(user_input)
    
    response = find_best_answer(user_input, data)
    
    if response:
        print(f"Chatbot: {sentiment_response} Also, {response}")
    else:
        print("Chatbot: I don't know about this topic. Let me think... 🤔")
        gpt2_response = generate_gpt2_response(user_input)
        print(f"Chatbot: {gpt2_response}")

        new_answer = input("You tell me, how should I answer this? ")
        save_data(user_input, new_answer)
        print("Chatbot: Thanks! Now I know this. 😊 Restart me, and I will answer correctly!")
