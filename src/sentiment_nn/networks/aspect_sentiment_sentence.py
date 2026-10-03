
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"

def train():
    df = pd.read_csv("/Users/cheppers-one/code/sentiment-nn/data/aspect_sentences.csv", keep_default_na=False)
    df["label_list"] = df["labels"].apply(lambda s: s.split(";") if s else [])
    
    print(f"{len(df)} sentences from {df['review_id'].nunique()} reviews")
    print(f"{(df['labels'] == '').mean():.0%} of sentences have no aspect")
    print()
    
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["label_list"])
    print(f"{Y.shape[1]} labels: {', '.join(mlb.classes_)}")
    print()
    
    groups = df["review_id"].values
    split1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    idx_train, idx_rest = next(split1.split(df, groups=groups))
    split2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    rest_val, rest_test = next(split2.split(idx_rest, groups=groups[idx_rest]))
    idx_val, idx_test = idx_rest[rest_val], idx_rest[rest_test]
    
    Y_train, Y_val, Y_test = Y[idx_train], Y[idx_val], Y[idx_test]
    print(f"sentences: train={len(idx_train)}  validation={len(idx_val)}  test={len(idx_test)}")
    assert not set(groups[idx_train]) & set(groups[idx_test]), "a review leaked into both sets"
    
    sentences = df["sentence"].tolist()
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    print("embedding sentences")
    X_all = _vectorize(embedder, sentences)
    X_train, X_val, X_test = X_all[idx_train], X_all[idx_val], X_all[idx_test]

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
    
    def report(name, Y_true, Y_pred, show_no_aspect=True):
        empty = Y_true.sum(axis=1) == 0
        no_aspect = f"{(Y_pred[empty].sum(axis=1) == 0).mean():.3f}" if empty.any() else "n/a"
        print(f"  {name:<30} micro F1={f1_score(Y_true, Y_pred, average='micro', zero_division=0):.3f}  "
            f"macro F1={f1_score(Y_true, Y_pred, average='macro', zero_division=0):.3f}  "
            f"exact={accuracy_score(Y_true, Y_pred):.3f}" + (f"  no-aspect acc={no_aspect}" if show_no_aspect else ""))

    print("\nSENTENCE level, test set (threshold 0.5):")
    report("Lazy floor", Y_test, dummy.predict(X_test))
    report("Logistic regression", Y_test, baseline.predict(X_test))
    report("Neural net (MLP)", Y_test, mlp.predict(X_test))
    
    val_probs = mlp.predict_proba(X_val)
    thresholds = np.full(Y.shape[1], 0.5)
    for j in range(Y.shape[1]):
        best_f1 = -1
        for t in np.arange(0.3, 0.81, 0.05):
            f1 = f1_score(Y_val[:, j], (val_probs[:, j] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, thresholds[j] = f1, round(t, 2)

    def apply_thresholds(probs):
        pred = (probs >= thresholds).astype(int)
        aspects = [c.split(":")[0] for c in mlb.classes_]
        for aspect in set(aspects):
            cols = [j for j, a in enumerate(aspects) if a == aspect]   # e.g. pricing:neg, pricing:pos
            both = pred[:, cols].sum(axis=1) > 1
            for row in np.where(both)[0]:
                weaker = min(cols, key=lambda j: probs[row, j])
                pred[row, weaker] = 0
        return pred
    
    test_probs = mlp.predict_proba(X_test)
    Y_pred = apply_thresholds(test_probs)
    report("Neural net (MLP) + tuned thresholds", Y_test, Y_pred)
    print("\nPer-label results (sentence level):")
    print(classification_report(Y_test, Y_pred, target_names=mlb.classes_, zero_division=0))
    
    test_reviews = groups[idx_test]
    true_by_review = pd.DataFrame(Y_test, columns=mlb.classes_).groupby(test_reviews).max()
    pred_by_review = pd.DataFrame(Y_pred, columns=mlb.classes_).groupby(test_reviews).max()

    print("REVIEW level, test set (sentence predictions combined):")
    report("Neural net + tuned thresholds", true_by_review.values, pred_by_review.values, show_no_aspect=False)
    
    joblib.dump({
        "model": mlp,
        "mlb": mlb,
        "thresholds": thresholds,
        "embedding_model": EMBEDDING_MODEL,        
    }, "aspect_model.joblib")
    print("saved model to aspect_model.joblib")

def inference(sentences: list[str]):
    bundle = joblib.load("aspect_model.joblib")
    model, mlb, thresholds = bundle["model"], bundle["mlb"], bundle["thresholds"]
    embedder = SentenceTransformer(bundle["embedding_model"])
    
    probs = model.predict_proba(_vectorize(embedder, sentences))
    results = []
    for p in probs:
        best = {}
        for j, label in enumerate(mlb.classes_):
            if p[j] > thresholds[j]:
                aspect, sentiment = label.split(":")
                if aspect not in best or p[j] > best[aspect][1]:
                    best[aspect] = (sentiment, float(p[j]))
        results.append(best)
    return results
    
def _vectorize(embedder: SentenceTransformer, texts: list[str]):
    return embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)