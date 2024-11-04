from flask import Flask, request, jsonify
from sentiment_classifier import SentimentClassifier

app = Flask(__name__)
classifier = SentimentClassifier.load()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    prediction = classifier.predict(text)
    return jsonify({
        'text': text,
        'sentiment': 'Positive' if prediction == 1 else 'Negative'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)