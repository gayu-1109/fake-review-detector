from flask import Flask, request, render_template, send_file
import joblib
import re
import pandas as pd
import io
from textblob import TextBlob

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

history = []

def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

@app.route('/')
def home():
    return render_template("index.html", history=history)

@app.route('/predict', methods=["POST"])
def predict():
    global history
    results = []

    if 'review' in request.form and request.form['review']:
        review = request.form['review']
        cleaned = clean_text(review)
        vectorized = vectorizer.transform([cleaned])
        pred = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized).max() * 100
        sentiment = get_sentiment(review)
        result = {
            "review": review,
            "result": pred.upper(),
            "confidence": f"{prob:.2f}",
            "sentiment": sentiment
        }
        history.insert(0, result)
        return render_template(
            "index.html",
            history=history,
            prediction_text=f"{pred.upper()} ({prob:.2f}%) - {sentiment}",
            download_ready=True
        )

    if 'file' in request.files and request.files['file']:
        file = request.files['file']
        filename = file.filename.lower()

        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(file)
        elif filename.endswith('.csv'):
            try:
                df = pd.read_csv(file, encoding='utf-8')
            except UnicodeDecodeError:
                file.seek(0)
                try:
                    df = pd.read_csv(file, encoding='cp1252')
                except UnicodeDecodeError:
                    file.seek(0)
                    df = pd.read_csv(file, encoding='ISO-8859-1')
        else:
            return "Unsupported file type. Please upload CSV or Excel."

        col_name = df.columns[0]
        fake_count = 0
        genuine_count = 0

        for review in df[col_name]:
            review_str = str(review)
            cleaned = clean_text(review_str)
            vectorized = vectorizer.transform([cleaned])
            pred = model.predict(vectorized)[0]
            prob = model.predict_proba(vectorized).max() * 100
            sentiment = get_sentiment(review_str)
            results.append({
                "review": review_str,
                "result": pred.upper(),
                "confidence": f"{prob:.2f}",
                "sentiment": sentiment
            })
            if pred.upper() == "FAKE":
                fake_count += 1
            else:
                genuine_count += 1

        history = results + history

        summary = {"FAKE": fake_count, "GENUINE": genuine_count}
        return render_template(
            "index.html",
            history=history,
            summary=summary,
            download_ready=True
        )

    return render_template("index.html", history=history)

@app.route('/delete/<int:index>', methods=['POST'])
def delete(index):
    global history
    if 0 <= index < len(history):
        history.pop(index)
    return render_template("index.html", history=history)

@app.route('/clear', methods=['POST'])
def clear():
    global history
    history.clear()
    return render_template("index.html", history=history)

@app.route('/download', methods=['GET'])
def download():
    output_df = pd.DataFrame(history, columns=["review", "result", "confidence", "sentiment"])
    buffer = io.StringIO()
    output_df.to_csv(buffer, index=False)
    buffer.seek(0)
    return send_file(
        io.BytesIO(buffer.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='predictions.csv'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
