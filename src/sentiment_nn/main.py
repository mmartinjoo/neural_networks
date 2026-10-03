from sentiment_nn.networks import aspect_sentiment_sentence, simple_classifier, sentence_transformer, aspect_sentiment

def run_simple_classifier():
    simple_classifier.run()
    
def run_sentence_transformer():
    sentence_transformer.run()
    
def run_aspect_sentiment():
    aspect_sentiment.run()
    
def train_aspect_sentiment_sentence():
    aspect_sentiment_sentence.train()
    
def inference_aspect_sentiment_sentence():
    res = aspect_sentiment_sentence.inference([
        "Trello is a great product.",
        "I use it for my solo projects and it's quite a great product.",
        "It's really easy to use.",
        "I like the fact that i can use it for free.",
        "However, handling multiple projects is a bit tricky.",
        "It misses some crucial features such as an overall view for all of my projects and tasks.",
        "I've never experienced any performance problems.",
        "It's very fast even with thousands of tasks.",
    ])
    print(res)