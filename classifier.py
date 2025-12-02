import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

def preprocess_data(data):
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler
    oversample = SMOTE()
    X, y  = oversample.fit_resample(data.drop('Class', axis=1), data['Class'])
    X = StandardScaler().fit_transform(X)
    X = pd.DataFrame(X, columns=data.drop('Class', axis=1).columns)
    data = pd.concat([X, y], axis=1)
    return data

# ------Decision Tree Classifier------
def decision_tree(data):
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import classification_report, confusion_matrix

    print("-----Decision Tree Classifier-----")
    data = data.fillna(0)
    X = data.drop('Class', axis=1)
    y = data['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# ------Support Vector Machine Classifier------
def svmachine(data):
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report, confusion_matrix

    print("-----Support Vector Machine Classifier-----")
    data = data.fillna(0)
    X = data.drop('Class', axis=1)
    y = data['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    svc = SVC(random_state=42)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# ------Logistic Regression Classifier------
def logistic_regression(data):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix

    print("-----Logistic Regression Classifier-----")
    data = data.fillna(0)
    X = data.drop('Class', axis=1)
    y = data['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# ------Random Forest Classifier------
def random_forest(data):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix

    data = data.fillna(0)
    X = data.drop('Class', axis=1)
    y = data['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# ------LSTM Classifier------
def lstm(data):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import classification_report, confusion_matrix

    print("-----LSTM Classifier-----")

    def create_sequences(X, y, seq_length=1):
        sequences = []
        labels = []
        for i in range(len(X) - seq_length):
            seq = X[i:i+seq_length].values
            label = y[i+seq_length]
            sequences.append(seq)
            labels.append(label)
        return np.array(sequences), np.array(labels)
    
    X_train, y_train = create_sequences(data.drop('Class', axis=1), data['Class'], seq_length=5)
    X_test, y_test = create_sequences(data.drop('Class', axis=1), data['Class'], seq_length=5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class LSTMClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(LSTMClassifier, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            out = self.sigmoid(out)
            return out
        
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = LSTMClassifier(input_size=X_train.shape[2], hidden_size=50, num_layers=2, output_size=1).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}, RMSE: {torch.sqrt(loss).item():.4f}')

    with torch.no_grad():
        model.eval()
        all_preds = []
        all_labels = []
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (outputs.cpu().numpy() > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
        print(f'Final Training RMSE: {torch.sqrt(criterion(torch.tensor(all_preds, dtype=torch.float32), torch.tensor(all_labels, dtype=torch.float32))).item():.4f}')
