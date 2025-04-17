from transformers import pipeline

# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text):
    """
    Analyze sentiment of the given text.
    :param text: Input string
    :return: Sentiment result
    """
    result = sentiment_analyzer(text)
    return result


