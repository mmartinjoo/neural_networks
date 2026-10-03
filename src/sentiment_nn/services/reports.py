import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

def report_accuracy(name: str, predictions, y_test):
    print("=" * 53)
    print(name) 
    print(f"accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("=" * 53)
    print(classification_report(y_test, predictions))
    
    cm = confusion_matrix(y_test, predictions, labels=["negative", "positive"])
    print("Confusion matrix (rows = truth, columns = prediction)")
    print(pd.DataFrame(cm, index=["is negative", "is positive"],
                            columns=["said negative", "said positive"]), "\n")
    
def report_mistakes(name, predictions, X_test, y_test):
    mistakes = X_test[predictions != y_test]
    print(f"The {name} got {len(mistakes)} test reviews wrong")
    for text in mistakes.head(5):
        print(" -", text)
    print()
    
def review_baseline(baseline: Pipeline):
    # ---------------------------------------------------------------------------
    # 7. Look inside the baseline: logistic regression gives every word a weight.
    #    Large positive weight -> pushes towards "positive"; large negative -> "negative".
    #    (This is easy for logistic regression; neural nets are harder to inspect.)
    # ---------------------------------------------------------------------------
    vectorizer = baseline.named_steps["tfidfvectorizer"]
    classifier = baseline.named_steps["logisticregression"]
    weights = pd.Series(classifier.coef_[0], index=vectorizer.get_feature_names_out())

    print("Words that push most towards POSITIVE:")
    print(", ".join(weights.sort_values(ascending=False).head(15).index), "\n")
    print("Words that push most towards NEGATIVE:")
    print(", ".join(weights.sort_values().head(15).index), "\n")
    