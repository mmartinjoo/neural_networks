from sentiment_nn.networks import simple_classifier, sentence_transformer, aspect_sentiment

def run_simple_classifier():
    simple_classifier.run()
    
def run_sentence_transformer():
    sentence_transformer.run()
    
def run_aspect_sentiment():
    aspect_sentiment.run()