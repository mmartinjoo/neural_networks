import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report

USE_EMBEDDINGS = True

def run():
    df = pd.read_csv("/Users/cheppers-one/code/sentiment-nn/data/aspect_reviews.csv")
    labeled = df[df["labels"].notna()].reset_index(drop=True)
    unlabeled = df[df["labels"].isna()].reset_index(drop=True)
    
    labeled["label_list"] = labeled["labels"].str.split(";")
    # print(labeled[["review_text", "label_list"]].head(3).to_string(), "\n")
    
    print("Label list")
    print(labeled["label_list"].head(3))
    
    # List -> 0/1 table
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(labeled["label_list"])
    print("Labels: ", ", ".join(mlb.classes_), "\n")
    print(f"Labels turned into 0/1 tables for each review. {Y.shape[0]} reviews X {Y.shape[1]} binary labels")
    print(Y)
    print()
    
    # ---------------------------------------------------------------------------
    # 3. Three sets:
    #    train      (70%) - the model learns from these
    #    validation (15%) - used to tune thresholds; never used for training
    #    test       (15%) - the final exam, touched only at the very end
    # ---------------------------------------------------------------------------
    texts = labeled["review_text"].tolist()
    idx = np.arange(len(texts))
    idx_train, idx_rest = train_test_split(idx, test_size=0.3, random_state=42)
    idx_val, idx_test = train_test_split(idx_rest, test_size=0.5, random_state=42)
    Y_train, Y_val, Y_test = Y[idx_train], Y[idx_val], Y[idx_test]

    print(f"train={len(idx_train)}  validation={len(idx_val)}  test={len(idx_test)}\n")
    
    if USE_EMBEDDINGS:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
        
        def vectorize(text_list):
            return embedder.encode(text_list, normalize_embeddings=True, show_progress_bar=True)
        
        print("embedding text")
        X_all = vectorize(texts)
    else:
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        tfidf.fit([texts[i] for i in idx_train])
        
        def vectorize(text_list):
            return tfidf.transform(text_list)
        
        X_all = vectorize(texts)
        
    print("X_all")
    print(X_all[:3])
    
    X_train, X_val, X_test = X_all[idx_train], X_all[idx_val], X_all[idx_test]
    
    # train models
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, Y_train)
    baseline = OneVsRestClassifier(LogisticRegression(max_iter=1000)).fit(X_train, Y_train)
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(256,),
        alpha=1e-3,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42,
    ).fit(X_train, Y_train)
    
    # ---------------------------------------------------------------------------
    # Evaluate
    #    micro F1:        F1 over all individual label decisions together
    #    macro F1:        F1 per label, then averaged (rare labels count as much as common ones)
    #    exact match:     % of reviews where ALL labels were exactly right (harsh!)
    # ---------------------------------------------------------------------------
    
    def report(name, Y_true, Y_pred):
        print(f"{name:<28} micro F1={f1_score(Y_true, Y_pred, average='micro', zero_division=0):.3f}  "
              f"macro F1={f1_score(Y_true, Y_pred, average='macro', zero_division=0):.3f}  "
              f"exact match={accuracy_score(Y_true, Y_pred):.3f}")
        
    print("\nResults on the test set (threshold 0.5):")
    report("Dummy", Y_test, dummy.predict(X_test))
    report("Logistic regression", Y_test, baseline.predict(X_test))
    report("Neural net (MLP)", Y_test, mlp.predict(X_test))
    
    # Threshold tuning
    
    val_probs = mlp.predict_proba(X_val)          # shape: (reviews, 16) probabilities
    thresholds = np.full(Y.shape[1], 0.5)
    for j in range(Y.shape[1]):
        best_f1 = -1
        for t in np.arange(0.3, 0.81, 0.05):
            f1 = f1_score(Y_val[:, j], (val_probs[:, j] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, thresholds[j] = f1, t
    
    test_probs = mlp.predict_proba(X_test)
    Y_pred = (test_probs >= thresholds).astype(int)
    report("Neural net + tuned thresholds", Y_test, Y_pred)
    
    print("\nPer-label results (neural net, tuned thresholds):")
    print(classification_report(Y_test, Y_pred, target_names=mlb.classes_, zero_division=0))
    
    def predict_aspects(text_list):
        probs = mlp.predict_proba(vectorize(text_list))
        results = []
        for p in probs:
            best = {}  # aspect -> (sentiment, probability)
            for j, label in enumerate(mlb.classes_):
                if p[j] >= thresholds[j]:
                    aspect, sentiment = label.split(":")
                    # If an aspect comes out both + and -, keep the more confident one.
                    # (Simple rule; a review CAN be mixed on one aspect, see the notes.)
                    if aspect not in best or p[j] > best[aspect][1]:
                        best[aspect] = ("+" if sentiment == "positive" else "-", round(float(p[j]), 2))
            pairs = [(aspect, s, prob) for aspect, (s, prob) in best.items()]
            results.append(sorted(pairs, key=lambda x: -x[2]))
        return results
    
    my_sentences = [
        "Cheap, but support never answers.",
        "The dashboards are gorgeous and it never goes down.",
        "We pay a fortune and it still crashes every week.",
        "Onboarding was a breeze, though the API docs are a mess.",
        "Highly recommend.",   # general praise, no specific aspect -> ideally no labels
    ]
    print("\nMy own sentences:")
    for text, pairs in zip(my_sentences, predict_aspects(my_sentences)):
        print(f"  {text}\n     -> {pairs or 'no aspect detected'}")
    
    print("\nFirst 3 unlabeled reviews:")
    for text, pairs in zip(unlabeled["review_text"][:3], predict_aspects(unlabeled["review_text"][:3].tolist())):
        print(f"  {text}\n     -> {pairs}")