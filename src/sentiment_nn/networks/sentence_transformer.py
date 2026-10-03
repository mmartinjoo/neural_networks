import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sentence_transformers import SentenceTransformer
from sentiment_nn.services import reports

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

def run():
    df = pd.read_csv("/Users/cheppers-one/code/sentiment-nn/data/tech_product_reviews_labeled.csv")
    
    df["text"] = df["title"] + ". " + df["review_text"]
    
    print(f"Total reviews: {len(df)}")
    print(df["label"].value_counts(dropna=False))

    labaled = df[df["label"].notna()]
    unlabeled = df[df["label"].isna()]

    X = labaled["text"]     # input
    y = labaled["label"]    # output
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    
    print("Embedding training set...")
    X_train_vec = embedder.encode(X_train.tolist(), show_progress_bar=True)
    X_test_vec = embedder.encode(X_test.tolist())
    
    print(f"Training on {len(X_train)} reviews, testing on {len(X_test)}")
    
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    
    # TfidfVectorizer settings:
    #   ngram_range=(1, 2): look at single words ("expensive") AND word pairs ("not worth").
    #                       Pairs matter: "not worth" means the opposite of "worth".
    #   min_df=2:           ignore words that appear in only one review (noise, typos).
    baseline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3),
        LogisticRegression(max_iter=1000),
    )
    
    # i built the example, it was quite easy to follow
    # i have questions, however:
    # - what does the TfidVectorizer does? i thought it was a quite basic vectorizer that will now work well in real scnearios.
    # - if i want to use a more advanced vectorizer that also understands meaning and context, what are my choices?
    # - explain the MLPClassifier in more detail. i thought it is used to assign many labels to a sentence, for example. but here it only gives one label, right?
    
    classifier = MLPClassifier(
        hidden_layer_sizes=(64,),
        alpha=1e-3,
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42,
    )   
    
    print("training the logistic regression baseline...")
    baseline.fit(X_train, y_train)
    
    print("training the neural network...")
    classifier.fit(X_train_vec, y_train)
    print()
    
    predictions = baseline.predict(X_test)
    reports.report_accuracy("baseline", predictions, y_test)
    
    predictions = classifier.predict(X_test_vec)
    reports.report_accuracy("classifier", predictions, y_test)
    
    print("=" * 53)
    print(f"Dummy") 
    print(f"accuracy: {dummy.score(X_test, y_test):.3f}")
    print("=" * 53)
    print()
        
    predictions = classifier.predict(embedder.encode(X_test.tolist()))
    reports.report_mistakes("classifier", predictions, X_test, y_test)
    reports.review_baseline(baseline)
        
    print("Predictions for the first 5 unlabeled reviews:")
    probabilities = classifier.predict_proba(embedder.encode(unlabeled["text"].tolist()))
    classes = list(classifier.classes_)  # ["negative", "positive"]
    for text, probs in list(zip(unlabeled["text"], probabilities))[:5]:
        label = classes[probs.argmax()]
        print(f"  [{label} {probs.max():.0%}] {text}")
    print()
    
    my_sentences = [
        "Support never replied and the app keeps crashing.",
        "Setup took ten minutes and my team loves it.",
        "It's okay, but way too expensive for a small team.",
        "Not bad at all, actually.",                            # tricky: negation
        "Great, another price increase. Just what we needed.",  # tricky: sarcasm
    ]
    print("Predictions for my own sentences:")
    for text, probs in zip(my_sentences, classifier.predict_proba(embedder.encode(my_sentences))):
        print(f"  [{classes[probs.argmax()]} {probs.max():.0%}] {text}")