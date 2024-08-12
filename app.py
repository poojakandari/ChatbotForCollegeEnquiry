import nltk
from nltk.corpus import stopwords
import json
import random
import numpy as np
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify,session
from flask_mysqldb import MySQL
from MySQLdb import IntegrityError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker  # Use pyspellchecker for spelling correction

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'random'

# MySQL configurations
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'pwd'
app.config['MYSQL_DB'] = 'login'
mysql = MySQL(app)

# Initialize NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Initialize spell checker
spell = SpellChecker()

# Load intents file
with open('intents.json') as file:
    data = json.load(file)

# Preprocess data
ignore_words = ['?', '!', '.', ',']
corpus = []
tags = []

def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word not in ignore_words and word.lower() not in stop_words]
    return ' '.join(tokens)

for intent in data['intents']:
    for pattern in intent['patterns']:
        corpus.append(preprocess_text(pattern))
        tags.append(intent['tag'])

# Vectorize the corpus
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Function to correct spelling
def correct_spelling(text):
    words = text.split()
    corrected_words = [spell.candidates(word).pop() if spell.candidates(word) else word for word in words]
    corrected_text = ' '.join(corrected_words)
    return corrected_text

# Chatbot response function
def get_bot_response(user_input):
    user_input_str = preprocess_text(user_input)
    user_input_vec = vectorizer.transform([user_input_str])

    # Calculate cosine similarity
    similarities = cosine_similarity(user_input_vec, X).flatten()
    max_similarity_index = np.argmax(similarities)

    if similarities[max_similarity_index] > 0.4:  # Adjusted threshold
        best_match_tag = tags[max_similarity_index]
        for intent in data['intents']:
            if intent['tag'] == best_match_tag:
                response = random.choice(intent['responses'])
                return response

    # If no response is found, suggest spelling corrections
    corrected_input = correct_spelling(user_input)
    corrected_input_str = preprocess_text(corrected_input)
    corrected_input_vec = vectorizer.transform([corrected_input_str])

    # Recalculate cosine similarity with corrected input
    similarities = cosine_similarity(corrected_input_vec, X).flatten()
    max_similarity_index = np.argmax(similarities)

    if similarities[max_similarity_index] > 0.4:  # Adjusted threshold
        best_match_tag = tags[max_similarity_index]
        for intent in data['intents']:
            if intent['tag'] == best_match_tag:
                response = random.choice(intent['responses'])
                return f"Did you mean: '{corrected_input}'? {response}"

    return "Sorry, I don't understand your question."

@app.route('/')
def index():
    session.clear()
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form['id']
        password = request.form['password']

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE id = %s AND password = %s", (user_id, password))
        user = cur.fetchone()
        cur.close()

        if user:
            flash('Logged in successfully!', 'success')
            return render_template('chat.html', user_id=user_id)
        else:
            flash('Invalid ID or Password', 'danger')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_id = request.form['id']
        password = request.form['password']

        cur = mysql.connection.cursor()
        try:
            cur.execute("INSERT INTO users (id, password) VALUES (%s, %s)", (user_id, password))
            mysql.connection.commit()
            flash('You have successfully registered!', 'success')
            return render_template('chat.html', user_id=user_id)
        except IntegrityError:
            mysql.connection.rollback()
            flash('This ID is already taken. Please choose a different one.', 'danger')
            return redirect(url_for('register'))
        finally:
            cur.close()
    return render_template('register.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    response = get_bot_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
