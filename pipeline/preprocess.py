from tools import *

def preprocess_data():
    data = pd.read_csv('credit.csv')
    assert TARGET_COL in data.columns, f"Target column '{TARGET_COL}' not found in data"

    X = data.drop(TARGET_COL, axis=1)
    y = data[TARGET_COL]

    X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # RFE Feature Selection
    lr_rfe = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear', random_state=RANDOM_STATE)
    rfe = RFE(estimator=lr_rfe, n_features_to_select=N_FEATURES_TO_SELECT, step=1)
    rfe.fit(X_train, y_train)
    mask = rfe.support_
    selected_features = X.columns[mask].tolist()
    print(f"Selected Features: {selected_features}")

    return X_train[:, mask], X_test[:, mask], y_train, y_test