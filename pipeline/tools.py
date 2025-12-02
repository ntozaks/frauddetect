import time
import os
import numpy as np
import pandas as pd
from collections import deque
import random
import psutil

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import make_pipeline

from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout
from tensorflow.keras.optimizers import Adam


# ---------Hyperparameters---------
DATA_PATH = "../creditcard.csv"
TARGET_COL = "Class"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_FEATURES_TO_SELECT = 10
LSTM_UNITS = 64
LSTM_EPOCHS = 10
BATCH_SIZE = 256

DQN_GAMMA = 0.99
DQN_LR = 1e-3
DQN_EPS_START = 1.0
DQN_EPS_END = 0.01
DQN_EPS_DECAY = 0.995
DQN_BATCH = 128
DQN_MEMORY_SIZE = 10000
DQN_UPDATES = 10000 
DQN_TRAIN_EVERY = 16