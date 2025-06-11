import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from ripser import ripser
from persim import plot_diagrams

#Load and preprocess data

df1 = pd.read_excel('pems409529b.xlsx').iloc[:,0:2]
df2 = pd.read_excel('pems409529c.xlsx').iloc[:,0:2]

df3 = pd.read_excel('pems716887b.xlsx').iloc[:,[0,2]]
df4 = pd.read_excel('pems716887c.xlsx').iloc[:,0:2]

df5 = pd.read_excel('pems759259b.xlsx').iloc[:,[0,4]]
df6 = pd.read_excel('pems759259c.xlsx').iloc[:,0:2]

df7 = pd.read_excel('pems774244b.xlsx').iloc[:,0:2]
df8 = pd.read_excel('pems774244c.xlsx').iloc[:,0:2]

df9 = pd.read_excel('pems1119241b.xlsx').iloc[:,0:2]
df10 = pd.read_excel('pems1119241c.xlsx').iloc[:,0:2]

#Add binary labels for incident days

def incidents(df1, df2):
    df1['Hour'] = pd.to_datetime(df1['Hour'])
    df2['Time'] = pd.to_datetime(df2['Time'])

    df1['date'] = df1['Hour'].dt.date
    df2['date'] = df2['Time'].dt.date

    incident_dates = set(df2['date'])
    df1['incident_occurred'] = df1['date'].isin(incident_dates).astype(int)
    df1.drop('date', axis=1, inplace=True)

incidents(df1, df2)
incidents(df3, df4)
incidents(df5, df6)
incidents(df7, df8)
incidents(df9, df10)

#Normalize flow data

all_dfs = [df1, df3, df5, df7, df9]

for df in all_dfs:
    df['Flow (Veh/Hour)'] = StandardScaler().fit_transform(df[['Flow (Veh/Hour)']])

#TDA Processing Functions

def sliding_windows(series, window_size, step):
    return [series[i:i+window_size].values for i in range(0, len(series) - window_size + 1, step)]

def takens_embedding(signal, d, tau):
    n_points = len(signal) - (d - 1) * tau
    if n_points <= 0:
        return None
    return np.array([signal[i : i + d * tau : tau] for i in range(n_points)])

def generate_tda_windows(df, window_size=48, step=6, d=3, tau=2):
    flow_series = df['Flow (Veh/Hour)']
    label_series = df['incident_occurred']

    flow_windows = sliding_windows(flow_series, window_size, step)
    label_windows = sliding_windows(label_series, window_size, step)

    embedded_point_clouds = []
    incident_labels = []

    for i in range(len(flow_windows)):
        window = flow_windows[i]
        label_window = label_windows[i]

        emb = takens_embedding(window, d=d, tau=tau)
        if emb is not None:
            embedded_point_clouds.append(emb)
            incident_labels.append(int(np.any(label_window)))

    return embedded_point_clouds, incident_labels

def compute_persistence_diagrams(point_clouds, maxdim=1):
    h1_diagrams = []
    for pc in point_clouds:
        dgms = ripser(pc, maxdim=maxdim)['dgms']
        h1_diagrams.append(dgms[1])
    return h1_diagrams

#Convert persistence diagrams to features for ML

def diagram_features(dgm):
    if len(dgm) == 0:
        return [0, 0]
    persistences = dgm[:, 1] - dgm[:, 0]
    return [len(dgm), np.mean(persistences)]

#Wrap whole pipeline for one dataframe

def process_dataframe(df, window_size=48, step=6, d=3, tau=2):
    embedded, labels = generate_tda_windows(df, window_size, step, d, tau)
    diagrams = compute_persistence_diagrams(embedded)
    X = np.array([diagram_features(dgm) for dgm in diagrams])
    y = np.array(labels)
    return X, y

#Apply to all dataframes

X_all, y_all = [], []

for df in all_dfs:
    X, y = process_dataframe(df)
    X_all.append(X)
    y_all.append(y)

X_all = np.vstack(X_all)
y_all = np.concatenate(y_all)

print("Final feature matrix shape:", X_all.shape)
print("Final label vector shape:", y_all.shape)

#Train a classifier

clf = RandomForestClassifier(n_estimators=100, random_state=0)
scores = cross_val_score(clf, X_all, y_all, cv=5, scoring='roc_auc')

print("Cross-validated AUCs:", scores)
print("Mean AUC:", np.mean(scores))
