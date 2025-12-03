from preprocess import *
from classifier import lstm_model, X_train_lstm, X_test_lstm, summarize_results
from utilities import resources, eval

X_train, X_test, y_train, y_test = preprocess_data()
results = {}

def build_lstm_encoder(input_shape, latent_dim=32):
    input = Input(shape=input_shape)
    x = LSTM(LSTM_UNITS, name="encoder_lstm")(input)
    x = Dense(latent_dim, activation='relu', name="encoder_dense")(x)
    model = Model(inputs=input, outputs=x)
    return model

latent_dim = 32
encoder = build_lstm_encoder((1, X_train.shape[1]), latent_dim)

try:
    lstm_layer = None
    for layer in lstm_model.layers:
        if isinstance(layer, LSTM):
            lstm_layer = layer
            break
    if lstm_layer is not None:
        encoder.get_layer("encoder_lstm").set_weights(lstm_layer.get_weights())
        print("LSTM weights transferred successfully.")
except Exception as e:
    print(f"Error transferring weights: {e}")

def build_dqn(latent_dim, action_size=2):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(latent_dim,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=DQN_LR), loss='mse')
    return model

dqn_model = build_dqn(latent_dim, action_size=2)
target_model = build_dqn(latent_dim, action_size=2)
target_model.set_weights(dqn_model.get_weights())
replay = deque(maxlen=DQN_MEMORY_SIZE)

def reward(action, true_label):
    if action == true_label:
        return 1.0
    else:
        if true_label == 1 and action == 0:
            return -5.0
        else:
            return -1.0
        
def experience(state, action, reward, next_state, done):
    replay.append((state, action, reward, next_state, done))
def sample(batch_size):
    batch = random.sample(replay, min(len(replay), batch_size))
    return batch

X_train_encoded = encoder.predict(X_train_lstm, batch_size=BATCH_SIZE)
X_test_encoded = encoder.predict(X_test_lstm, batch_size=BATCH_SIZE)

@resources
def train_dqn(encodings, labels, enc_next=None, steps=DQN_UPDATES):
    epsilon = DQN_EPS_START
    step = 0
    updates = 0
    idx = np.arange(encodings.shape[0])

    for i in range(min(5000, encodings.shape[0])):
        state = encodings[i % encodings.shape[0]]
        true = int(labels.iloc[i % encodings.shape[0]])
        a = random.choice([0,1])
        r = reward(a, true)
        next_state = state
        experience(state, a, r, next_state, False)

    total_steps = steps
    while step < total_steps:
        i = random.randrange(encodings.shape[0])
        state = encodings[i]
        true = int(labels.iloc[i])

        if random.random() < epsilon:
            action = random.choice([0,1])
        else:
            qstate = dqn_model.predict(state.reshape(1, -1), verbose=0)[0]
            a = int(np.argmax(qstate))
        r = reward(action, true)
        next_state = encodings[(i+1) % encodings.shape[0]]
        experience(state, action, r, next_state, False)

        if len(replay) >= DQN_BATCH and step % DQN_TRAIN_EVERY == 0:
            batch = sample(DQN_BATCH)
            states = np.array([b[0] for b in batch])
            actions = np.array([b[1] for b in batch])
            rewards = np.array([b[2] for b in batch])
            next_states = np.array([b[3] for b in batch])
            dones = np.array([b[4] for b in batch])

            q_next = target_model.predict(next_states, verbose=0)
            q_target = dqn_model.predict(states, verbose=0)

            for idx in range(len(batch)):
                a = actions[idx]
                q_target[idx][a] = rewards[idx] + (0 if dones[idx] else DQN_GAMMA * np.amax(q_next[idx]))

            dqn_model.train_on_batch(states, q_target)
            updates += 1

            if updates % 50 == 0:
                target_model.set_weights(dqn_model.get_weights())

        epsilon = max(DQN_EPS_END, epsilon * DQN_EPS_DECAY)
        step += 1

    return dqn_model

(dqn_trained, dqn_time, dqn_before, dqn_after) = train_dqn(X_train_encoded, y_train, steps=DQN_UPDATES)
results["DQN+LSTM"] = {"model": dqn_trained, "time": dqn_time, "before": dqn_before, "after": dqn_after}

q_vals_test = dqn_trained.predict(X_test_encoded, verbose=0)
y_pred_dqn = np.argmax(q_vals_test, axis=1)

from scipy.special import softmax
y_score_dqn = softmax(q_vals_test, axis=1)[:, 1]

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_dqn, average='binary')
dqn_eval = {
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc_score(y_test, y_score_dqn) if len(np.unique(y_test)) > 1 else None,
    "prc_auc": average_precision_score(y_test, y_score_dqn) if len(np.unique(y_test)) > 1 else None,
    "confusion_matrix": confusion_matrix(y_test, y_pred_dqn),
    "y_pred": y_pred_dqn,
    "y_score": y_score_dqn
}
results['DQN+LSTM']["eval"] = dqn_eval
print("DQN+LSTM Results:", dqn_eval)

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