import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from ripser import ripser
from persim import plot_diagrams
from persim import PersistenceLandscaper

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df409529 = pd.read_excel('pems409529.xlsx')
df767523 = pd.read_excel('pems767523.xlsx').iloc[0:2132,:]
df1221136 = pd.read_excel('pems1221136.xlsx').iloc[0:2132,:]
df500010072 = pd.read_excel('pems500010072.xlsx').iloc[0:2132,:]
df500013111 = pd.read_excel('pems500013111.xlsx').iloc[0:2132,:]

hour_column = df409529[['Hour']]
df_sensors = pd.concat([
    df409529.iloc[:,4],
    df767523.iloc[:,5],
    df1221136.iloc[:,2],
    df500010072.iloc[:,3],
    df500013111.iloc[:,3]
], axis=1)
df_sensors.columns = ["sensor1", "sensor2", "sensor3", "sensor4", "sensor5"]
df = pd.concat([hour_column, df_sensors], axis=1)




df.set_index('Hour', inplace=True)

sensor_columns = [col for col in df.columns if 'sensor' in col]
X_raw = df[sensor_columns]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
df_scaled = pd.DataFrame(X_scaled, index=df.index, columns=sensor_columns)



N_SENSORS = 5
LOOKBACK_PERIOD = 12
FORECAST_PERIOD = 6


RESOLUTION = 100
NUM_LANDSCAPES_TO_USE = 3
LANDSCAPE_CALCULATOR = PersistenceLandscaper(hom_deg=1)



def create_topological_features_landscape(window_df):
    point_cloud = window_df.values
    if point_cloud.shape[0] < 2:
        return None

    try:
        dgms = ripser(point_cloud, maxdim=1)['dgms']
        
        landscapes = LANDSCAPE_CALCULATOR.fit_transform([dgms], resolution=RESOLUTION)
        

        num_actual_landscapes = landscapes.shape[0]
        
        fixed_size_landscapes = np.zeros((NUM_LANDSCAPES_TO_USE, RESOLUTION))
        
        len_to_copy = min(num_actual_landscapes, NUM_LANDSCAPES_TO_USE)
        if len_to_copy > 0:
            fixed_size_landscapes[:len_to_copy, :] = landscapes[:len_to_copy, :]
            
        return fixed_size_landscapes.flatten()

    except Exception as e:
        return np.zeros(NUM_LANDSCAPES_TO_USE * RESOLUTION)


topological_features = []
targets = []
feature_timestamps = []

for t in tqdm(range(LOOKBACK_PERIOD, len(df_scaled) - FORECAST_PERIOD)):
    lookback_window = df_scaled.iloc[t - LOOKBACK_PERIOD : t]
    forecast_window = df_scaled.iloc[t : t + FORECAST_PERIOD]
    target_y = forecast_window.values.mean()
    tda_vector = create_topological_features_landscape(lookback_window)
    
    if tda_vector is not None:
        topological_features.append(tda_vector)
        targets.append(target_y)
        feature_timestamps.append(lookback_window.index[-1])

X_tda_features = np.array(topological_features)
y_targets = np.array(targets)





time_df = pd.DataFrame(index=feature_timestamps)
time_df['hour_of_day'] = time_df.index.hour
time_df['day_of_week'] = time_df.index.dayofweek
X_time_features = pd.get_dummies(time_df, columns=['hour_of_day', 'day_of_week'], drop_first=True).astype(int)

X_baseline = X_time_features.values
X_tda_enhanced = np.hstack([X_tda_features, X_time_features.values])





split_index = int(len(y_targets) * 0.8)
X_train_tda, X_test_tda = X_tda_enhanced[:split_index], X_tda_enhanced[split_index:]
y_train, y_test = y_targets[:split_index], y_targets[split_index:]
X_train_base, X_test_base = X_baseline[:split_index], X_baseline[split_index:]


model_baseline = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_baseline.fit(X_train_base, y_train)
preds_baseline = model_baseline.predict(X_test_base)
mae_baseline = mean_absolute_error(y_test, preds_baseline)



model_tda = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_tda.fit(X_train_tda, y_train)
preds_tda = model_tda.predict(X_test_tda)
mae_tda = mean_absolute_error(y_test, preds_tda)

print("\n--- Model Evaluation Results ---")
print(f"Baseline Model MAE:   {mae_baseline:.4f}")
print(f"TDA-Enhanced Model MAE: {mae_tda:.4f}")

if mae_tda < mae_baseline:
    improvement = (mae_baseline - mae_tda) / mae_baseline * 100
    print(f"\nSuccess! The TDA features improved the model's MAE by {improvement:.2f}%.")
else:
    print("\nThe TDA features did not improve the model's performance in this configuration.")



tda_feature_labels = [f'TDA_{i}' for i in range(X_tda_features.shape[1])]
time_feature_labels = list(X_time_features.columns)
all_feature_labels = tda_feature_labels + time_feature_labels

importances = model_tda.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': all_feature_labels,
    'importance': importances
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20))
plt.title('Top 20 Feature Importances in TDA-Enhanced Model')
plt.show()

plt.figure(figsize=(15, 6))
plt.plot(y_test, label='Actual Values', color='black', alpha=0.7)
plt.plot(preds_baseline, label=f'Baseline Predictions (MAE: {mae_baseline:.4f})', color='orange', linestyle='--')
plt.plot(preds_tda, label=f'TDA Predictions (MAE: {mae_tda:.4f})', color='blue', linestyle='-')
plt.title('Prediction Performance on Test Set')
plt.xlabel('Time Steps in Test Set')
plt.ylabel('Scaled Average Flow')
plt.legend()
plt.grid(True)
plt.show()
