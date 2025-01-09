from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

app = Flask(__name__)

data = pd.read_csv('data/data.csv')
data['combined_text'] = data['review_title'].fillna('') + ' ' + data['review_body']

# Define language-specific stopwords
stopwords_dict = {
    'en': stopwords.words('english'),
    'de': stopwords.words('german'),
    'fr': stopwords.words('french'),
    'es': stopwords.words('spanish'),
    'ja': None,
    'zh': None
}

# We create separate datasets and vectorizers for each language
languages = ['en', 'de', 'fr', 'es', 'ja', 'zh']
vectorizers = {}
transformed_data = {}
lang_datasets = {}

for lang in languages:
    lang_data = data[data['language'] == lang]
    vectorizer = TfidfVectorizer(stop_words=stopwords_dict.get(lang))
    vectorizers[lang] = vectorizer
    transformed_data[lang] = vectorizer.fit_transform(lang_data['combined_text'])
    lang_datasets[lang] = lang_data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    user_input = request.json.get('query', '')
    language = request.json.get('language', 'en')
    sentiment = request.json.get('sentiment', 'positive')
    model = request.json.get('model', 'lstm')

    # Select vectorizer and data for the given language
    lang_data = lang_datasets[language]
    
    # Vectorize the user input
    user_vec = vectorizers[language].transform([user_input])

    # Compute cosine similarity
    similarities = cosine_similarity(user_vec, transformed_data[language]).flatten()

    # Retrieve the top 100 results based on cosine similarity
    top_indices = similarities.argsort()[-100:][::-1]

    # Filter top results
    results_df = lang_data.iloc[top_indices].copy()
    results_df['cosine_score'] = similarities[top_indices]

    # Adjust ranking based on sentiment - we chose to have a lower impact for sentiment
    if sentiment == 'positive':
        results_df['final_score'] = (
            0.7 * results_df['cosine_score'] + 0.3 * results_df['sentiment_score_lstm']
        )
    elif sentiment == 'negative':
        results_df['final_score'] = (
            0.7 * results_df['cosine_score'] + 0.3 * (1 - results_df['sentiment_score_lstm'])
        )

    # Sort results by the final score
    results_df = results_df.sort_values(by='final_score', ascending=False).head(5)

    results = results_df[['review_id', 'product_id', 'review_body', 'review_title', 'final_score', 'cosine_score', 'sentiment_score_lstm']].to_dict(orient='records')

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
