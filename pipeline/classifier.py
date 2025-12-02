from preprocess import *
from utilities import resources, eval

X_train, X_test, y_train, y_test = preprocess_data()
results = {}

# ------Logistic Regression Classifier------
@resources
def train_lr(X_tr, y_tr):
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
    clf.fit(X_tr, y_tr)
    return clf

(lr_clf, lr_time, lr_before, lr_after) = train_lr(X_train, y_train)
lr_eval = eval(lr_clf, X_test, y_test)
results['Logistic Regression'] = {"model": lr_clf, "eval": lr_eval, "time": lr_time, "before": lr_before, "after": lr_after}
print("Logistic Regression Results:", lr_eval)

# ------Random Forest Classifier------
@resources
def train_rf(X_tr, y_tr):
    clf = RandomForestClassifier(n_estimators=100, n_jobs=1, class_weight='balanced', random_state=RANDOM_STATE)
    clf.fit(X_tr, y_tr)
    return clf

(rf_clf, rf_time, rf_before, rf_after) = train_rf(X_train, y_train)
rf_eval = eval(rf_clf, X_test, y_test)
results['Random Forest'] = {"model": rf_clf, "eval": rf_eval, "time": rf_time, "before": rf_before, "after": rf_after}
print("Random Forest Results:", rf_eval)

# ------XGBoost Classifier------
@resources
def train_xgb(X_tr, y_tr):
    scale_pos = sum(y_tr == 0) / (sum(y_tr == 1) + 1e9)
    clf = XGBClassifier(\
        n_estimators = 300,
        max_depth = 5,
        learning_rate = 0.05,
        subsample = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = scale_pos,
        use_label_encoder = False,
        n_jobs = 1,
        random_state = RANDOM_STATE,
    )
    clf.fit(X_tr, y_tr)
    return clf
(xgb_clf, xgb_time, xgb_before, xgb_after) = train_xgb(X_train, y_train)
xgb_eval = eval(xgb_clf, X_test, y_test)
results['XGBoost'] = {"model": xgb_clf, "eval": xgb_eval, "time": xgb_time, "before": xgb_before, "after": xgb_after}
print("XGBoost Results:", xgb_eval)

# ------LSTM Classifier------
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

def build_lstm(input_shape, lstm_units=LSTM_UNITS):
    model = Sequential()
    model.add(LSTM(units=lstm_units, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['AUC'])
    return model

@resources
def train_lstm(X_tr, y_tr, X_val=None, y_val=None, epochs=LSTM_EPOCHS, batch_size=BATCH_SIZE):
    model = build_lstm((X_tr.shape[1], X_tr.shape[2]), units=LSTM_UNITS)
    model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=0)
    return model

(lstm_model, lstm_time, lstm_before, lstm_after) = train_lstm(X_train_lstm, y_train, X_test_lstm, y_test)
y_probs_lstm = lstm_model.predict(X_test_lstm).ravel()
y_pred_lstm = (y_probs_lstm >= 0.5).astype(int)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_lstm, average='binary')
lstm_eval = {
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc_score(y_test, y_probs_lstm) if len(np.unique(y_test)) > 1 else None,
    "prc_auc": average_precision_score(y_test, y_probs_lstm) if len(np.unique(y_test)) > 1 else None,
    "confusion_matrix": confusion_matrix(y_test, y_pred_lstm),
    "y_pred": y_pred_lstm,
    "y_score": y_probs_lstm
}
results['LSTM'] = {"model": lstm_model, "eval": lstm_eval, "time": lstm_time, "before": lstm_before, "after": lstm_after}
print("LSTM Results:", lstm_eval)

# ------Summary of Results------
def summarize_results(results):
    rows = []
    for name, info in results.items():
        metrics = info.get("eval", {})
        time_tr = info.get("time", None)
        mem_before = info.get("before", {})
        mem_after = info.get("after", {})
        rows.append({
            "Model": name,
            "Precision": metrics.get("precision"),
            "Recall": metrics.get("recall"),
            "F1-Score": metrics.get("f1"),
            "ROC-AUC": metrics.get("roc_auc"),
            "PRC-AUC": metrics.get("avg_precision"),
            "Training Time (s)": time_tr,
            "Memory Before (MB)": mem_before.get("rss_mb"),
            "Memory After (MB)": mem_after.get("rss_mb"),
        })
    return pd.DataFrame(rows).sort_values(by="F1-Score", ascending=False)
summary_df = summarize_results(results)
print("\nSummary of Classifier Results:")
print(summary_df.to_string(index=False))

for name, info in results.items():
    print(f"\n--- {name} Detailed Evaluation ---")
    metrics = info.get("eval", {})
    print(f"Precision: {metrics.get('precision'):.4f}", 
          f"Recall: {metrics.get('recall'):.4f}", 
          f"F1-Score: {metrics.get('f1'):.4f}", 
          f"ROC-AUC: {metrics.get('roc_auc'):.4f}" if metrics.get('roc_auc') is not None else "ROC-AUC: N/A", 
          f"PRC-AUC: {metrics.get('avg_precision'):.4f}" if metrics.get('avg_precision') is not None else "PRC-AUC: N/A", sep="\n")
    print("Confusion Matrix:\n", metrics.get("confusion_matrix"))