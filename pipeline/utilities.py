from tools import *

# ------Utility Functions------

def sample_resources():
    p = psutil.Process(os.getpid())
    return {"rss_mb": p.memory_info().rss / (1024**2), "cpu_percent": psutil.cpu_percent(interval=None)}

def resources(fn):
    def wrapper(*args, **kwargs):
        before = sample_resources()
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        t1 = time.perf_counter()
        after = sample_resources()
        return result, t1-t0, before, after
    return wrapper

def eval(clf, X_test, y_test, proba=True):
    y_pred = clf.predict(X_test)
    if proba and hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_test)[:, 1]
    elif proba and hasattr(clf, "decision_function"):
        try:
            y_proba = clf.decision_function(X_test)
        except:
            y_score = None
    else:
        y_score = None

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    roc_auc = roc_auc_score(y_test, y_score) if y_score is not None else None
    avg_precision = average_precision_score(y_test, y_score) if y_score is not None else None
    confusion_matrix_res = confusion_matrix(y_test, y_pred)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "confusion_matrix": confusion_matrix_res
    }