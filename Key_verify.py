# Key verification model

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import OneClassSVM
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# Load dataset
df = pd.read_csv('New_DDOS_.csv', low_memory=False)

# Optional: Sample if too large
# df = df.sample(n=10000, random_state=42)

# Preprocess
df = df.drop('Label_old', axis=1)
X = df.drop('Label', axis=1)
y = df['Label']

# Handle inf and nan
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to compute metrics
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    dr = recall_score(y_true, y_pred, pos_label=1)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    return dr, far, acc

results = {}

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)
results['Decision Tree'] = compute_metrics(y_test, pred_dt)

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)
results['KNN'] = compute_metrics(y_test, pred_knn)

# Naive Bayesian
nb = GaussianNB()
nb.fit(X_train, y_train)
pred_nb = nb.predict(X_test)
results['Naive Bayesian'] = compute_metrics(y_test, pred_nb)

# One-Class SVM
ocsvm = OneClassSVM(nu=0.1, gamma='scale')
normal_train = X_train[y_train == 0]
if len(normal_train) > 0:
    ocsvm.fit(normal_train)
    pred_oc = ocsvm.predict(X_test)
    pred_oc = np.array([1 if p == -1 else 0 for p in pred_oc])
    results['One-Class SVM'] = compute_metrics(y_test, pred_oc)
else:
    results['One-Class SVM'] = (0,0,0)

# DNN
class DNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
dataset = TensorDataset(X_train_t, y_train_t)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = DNN(X_train.shape[1])
optim = Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

for epoch in range(10):
    for batch_x, batch_y in loader:
        optim.zero_grad()
        out = model(batch_x)
        loss = loss_fn(out, batch_y)
        loss.backward()
        optim.step()

X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
with torch.no_grad():
    pred_dnn = model(X_test_t).round().squeeze().numpy()

results['Proposed DNN'] = compute_metrics(y_test, pred_dnn)

# Output results as table
print("ML Model\tDR\tFAR\tAccuracy")
for model_name, (dr, far, acc) in results.items():
    print(f"{model_name}\t{dr:.4f}\t{far:.4f}\t{acc:.4f}")
