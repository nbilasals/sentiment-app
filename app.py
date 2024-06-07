import os
import pandas as pd
from flask import Flask, render_template, request, send_file
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
import csv
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from io import BytesIO
import base64

sns.set_theme(color_codes=True)

app = Flask(__name__)

# Load the saved vectorizer
with open('data/tokenizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load logistic regression model
model = joblib.load(os.path.join('data', 'model.joblib'))

# Load lexicon
lexicon_positive = {}
with open(os.path.join('data', 'lexicon_positive_ver1.csv'), 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        lexicon_positive[row[0]] = int(row[1])

lexicon_negative = {}
with open(os.path.join('data', 'lexicon_negative_ver1.csv'), 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        lexicon_negative[row[0]] = int(row[1])


def sentiment_analysis_lexicon_indonesia(text):
    score = 0
    for word in text:
        if word in lexicon_positive:
            score += lexicon_positive[word]
        if word in lexicon_negative:
            score -= lexicon_negative[word]
    if score > 0:
        polarity = 'positive'
    elif score < 0:
        polarity = 'negative'
    else:
        polarity = 'neutral'
    return score, polarity


def preprocess_text(text, indo_stopwords):
    text = text.lower()
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub("@[A-Za-z0-9_]+", "", text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    superscript_pattern = re.compile("["u"\U00002070"
                                     u"\U000000B9"
                                     u"\U000000B2-\U000000B3"
                                     u"\U00002074-\U00002079"
                                     u"\U0000207A-\U0000207E"
                                     u"\U0000200D"
                                     "]+", flags=re.UNICODE)
    text = superscript_pattern.sub(r'', text)
    text = re.sub("#[A-Za-z0-9_]+", "", text)
    text = re.sub(r'(.)\1+', r'\1', text)
    tokens = word_tokenize(text)
    tokens = [
        word for word in tokens if word not in indo_stopwords and word not in string.punctuation]
    return tokens


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        if 'Komentar' not in df.columns:
            return "CSV must contain 'Komentar' column"

        nltk.download('punkt')
        nltk.download('stopwords')
        indo_stopwords = stopwords.words('indonesian')
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        df['processed_text'] = df['Komentar'].apply(
            lambda x: preprocess_text(x, indo_stopwords))

        df['sentiment_lr'] = df['processed_text'].apply(
            lambda x: model.predict(vectorizer.transform([' '.join(x)]))[0])

        # Map numeric labels to string labels
        label_mapping = {0: 'neutral', 1: 'negative', 2: 'positive'}
        df['sentiment_lr'] = df['sentiment_lr'].map(label_mapping)

        df['sentiment_lexicon'] = df['processed_text'].apply(
            lambda x: sentiment_analysis_lexicon_indonesia(x)[1])

        # Convert columns to string type to avoid comparison errors
        df['sentiment_lr'] = df['sentiment_lr'].astype(str)
        df['sentiment_lexicon'] = df['sentiment_lexicon'].astype(str)

        # Create charts and tables
        charts_data = create_charts(df)

        # Return rendered template with data
        return render_template('index.html', charts=charts_data)


def create_charts(df):
    # Convert columns to string type if necessary
    df['sentiment_lexicon'] = df['sentiment_lexicon'].astype(str)
    df['sentiment_lr'] = df['sentiment_lr'].astype(str)

    sentiment_counts_lexicon = df['sentiment_lexicon'].value_counts()
    sentiment_counts_lr = df['sentiment_lr'].value_counts()

    # Value Counts Comparison
    value_counts_comparison = pd.DataFrame({
        'Lexicon Based': sentiment_counts_lexicon,
        'Logistic Regression': sentiment_counts_lr
    }).fillna(0)

    # Chart 1: Sentiment Analysis Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.countplot(x='sentiment_lr', data=df, ax=ax1, palette='Set2')
    ax1.set_title('Logistic Regression')
    sns.countplot(x='sentiment_lexicon', data=df, ax=ax2, palette='Set3')
    ax2.set_title('Lexicon Based')
    buf = BytesIO()
    plt.savefig(buf, format="png")
    data_distribution = base64.b64encode(buf.getbuffer()).decode("ascii")
    buf.close()

    # Chart 2: Pie Chart for Sentiment Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    # Logistic Regression
    ax1.pie(sentiment_counts_lr, labels=sentiment_counts_lr.index,
            autopct='%1.1f%%', colors=sns.color_palette('Set2'))
    ax1.set_title('Logistic Regression')
    # Lexicon Based
    ax2.pie(sentiment_counts_lexicon, labels=sentiment_counts_lexicon.index,
            autopct='%1.1f%%', colors=sns.color_palette('Set3'))
    ax2.set_title('Lexicon Based')
    buf = BytesIO()
    plt.savefig(buf, format="png")
    data_pie = base64.b64encode(buf.getbuffer()).decode("ascii")
    buf.close()

    # Chart 3: Word Clouds for Logistic Regression
    wordclouds_lr = {}
    for sentiment in ['positive', 'neutral', 'negative']:
        text = ' '.join([' '.join(
            tokens) for tokens in df[df['sentiment_lr'] == sentiment]['processed_text']])
        if text:
            wordcloud = WordCloud(width=400, height=400,
                                  background_color='white').generate(text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            buf = BytesIO()
            plt.savefig(buf, format="png")
            wordclouds_lr[sentiment] = base64.b64encode(
                buf.getbuffer()).decode("ascii")
            buf.close()

    # Chart 4: Word Clouds for Lexicon-based
    wordclouds_lexicon = {}
    for sentiment in ['positive', 'neutral', 'negative']:
        text = ' '.join([' '.join(
            tokens) for tokens in df[df['sentiment_lexicon'] == sentiment]['processed_text']])
        if text:
            wordcloud = WordCloud(width=400, height=400,
                                  background_color='white').generate(text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            buf = BytesIO()
            plt.savefig(buf, format="png")
            wordclouds_lexicon[sentiment] = base64.b64encode(
                buf.getbuffer()).decode("ascii")
            buf.close()

    # Top 10 Reviews for each sentiment (Logistic Regression vs Lexicon)
    top_reviews_lr = {'positive': df[df['sentiment_lr'] == 'positive'].head(10),
                      'negative': df[df['sentiment_lr'] == 'negative'].head(10),
                      'neutral': df[df['sentiment_lr'] == 'neutral'].head(10)}

    top_reviews_lexicon = {'positive': df[df['sentiment_lexicon'] == 'positive'].head(10),
                           'negative': df[df['sentiment_lexicon'] == 'negative'].head(10),
                           'neutral': df[df['sentiment_lexicon'] == 'neutral'].head(10)}

    # Convert DataFrame to HTML
    df_html = df.to_html(classes='dataframe', header="true", index=False)
    value_counts_html = value_counts_comparison.to_html(
        classes='dataframe', header="true", index=True)
    top_reviews_lr_html = {}
    top_reviews_lexicon_html = {}

    for sentiment in ['positive', 'negative', 'neutral']:
        top_reviews_lr_html[sentiment] = top_reviews_lr[sentiment][[
            'Komentar', 'sentiment_lr']].to_html(classes='dataframe', header="true", index=False)
        top_reviews_lexicon_html[sentiment] = top_reviews_lexicon[sentiment][[
            'Komentar', 'sentiment_lexicon']].to_html(classes='dataframe', header="true", index=False)

    # Return the encoded images and the HTML table
    return {
        "data_distribution": f"data:image/png;base64,{data_distribution}",
        "data_pie": f"data:image/png;base64,{data_pie}",
        "wordclouds_lr": wordclouds_lr,
        "wordclouds_lexicon": wordclouds_lexicon,
        "table": df_html,
        "value_counts": value_counts_html,
        "top_reviews_lr": top_reviews_lr_html,
        "top_reviews_lexicon": top_reviews_lexicon_html
    }


if __name__ == '__main__':
    app.run(debug=True)
