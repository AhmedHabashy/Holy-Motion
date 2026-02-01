import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.fft import fft
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

BASE_PATH = Path('days')
WINDOW_BEFORE_MS = 200
WINDOW_AFTER_MS = 800
WINDOW_SIZE_MS = WINDOW_BEFORE_MS + WINDOW_AFTER_MS

SESSION_INFO = {
    1: {'name': 'idle', 'day': 1, 'punch_type': 'idle', 'use': 'train'},
    2: {'name': 'reference', 'day': 1, 'punch_type': 'mixed', 'use': 'train'},
    3: {'name': 'jab_light_slow', 'day': 2, 'punch_type': 'jab', 'use': 'train'},
    4: {'name': 'jab_medium', 'day': 2, 'punch_type': 'jab', 'use': 'train'},
    5: {'name': 'jab_power_fast', 'day': 2, 'punch_type': 'jab', 'use': 'train'},
    6: {'name': 'jab_natural', 'day': 2, 'punch_type': 'jab', 'use': 'train'},
    7: {'name': 'jab_mixed', 'day': 2, 'punch_type': 'jab', 'use': 'train'},
    8: {'name': 'jab_power_natural', 'day': 2, 'punch_type': 'jab', 'use': 'train'},
    9: {'name': 'hook_light_slow', 'day': 3, 'punch_type': 'hook', 'use': 'train'},
    10: {'name': 'hook_medium', 'day': 3, 'punch_type': 'hook', 'use': 'train'},
    11: {'name': 'hook_power_fast', 'day': 3, 'punch_type': 'hook', 'use': 'train'},
    12: {'name': 'hook_natural', 'day': 3, 'punch_type': 'hook', 'use': 'train'},
    13: {'name': 'hook_mixed', 'day': 3, 'punch_type': 'hook', 'use': 'train'},
    14: {'name': 'hook_power_natural', 'day': 3, 'punch_type': 'hook', 'use': 'train'},
    15: {'name': 'mixed', 'day': 4, 'punch_type': 'mixed', 'use': 'train'},
    16: {'name': 'mixed', 'day': 4, 'punch_type': 'mixed', 'use': 'train'},
    17: {'name': 'holdout_jabs', 'day': 5, 'punch_type': 'jab', 'use': 'holdout'},
    18: {'name': 'holdout_hooks', 'day': 5, 'punch_type': 'hook', 'use': 'holdout'},
    19: {'name': 'blind_mixed', 'day': 5, 'punch_type': 'mixed', 'use': 'holdout'},
    20: {'name': 'fatigued', 'day': 6, 'punch_type': 'mixed', 'use': 'train'},
    21: {'name': 'hard_negatives', 'day': 6, 'punch_type': 'hard_negative', 'use': 'train'},
    22: {'name': 'edge_cases', 'day': 6, 'punch_type': 'edge', 'use': 'train'},
}
DAY_FOLDERS = {1: 'day1_baseline', 2: 'day2_jabs', 3: 'day3_hooks', 4: 'day4_mixed', 5: 'day5_validation', 6: 'day6_stress'}

def find_session_files(session_id):
    info = SESSION_INFO[session_id]
    day_folder = DAY_FOLDERS[info['day']]
    day_path = BASE_PATH / day_folder
    session_folders = list(day_path.glob(f'session_{session_id:02d}_*'))
    session_folder = session_folders[0]
    data_files = list(session_folder.glob('*_data.csv'))
    beep_files = list(session_folder.glob('*_beeps.csv'))
    return data_files[0], beep_files[0]

def load_session(session_id):
    data_file, beep_file = find_session_files(session_id)
    return pd.read_csv(data_file), pd.read_csv(beep_file)

def extract_punch_windows(data_df, beep_df, session_info, session_id):
    windows = []
    for idx, beep in beep_df.iterrows():
        beep_time = beep['timestamp_ms']
        window_start = beep_time - WINDOW_BEFORE_MS
        window_end = beep_time + WINDOW_AFTER_MS
        mask = (data_df['timestamp_ms'] >= window_start) & (data_df['timestamp_ms'] <= window_end)
        window_data = data_df[mask].copy()
        if len(window_data) < 10:
            continue
        announcement = beep.get('announcement', '')
        if pd.notna(announcement) and announcement:
            announcement = str(announcement).upper()
            if 'JAB' in announcement:
                label = 'jab'
            elif 'HOOK' in announcement:
                label = 'hook'
            else:
                label = 'other'
        else:
            punch_type = session_info['punch_type']
            if punch_type in ['jab', 'hook']:
                label = punch_type
            elif punch_type == 'hard_negative':
                label = 'idle'
            elif punch_type == 'edge':
                label = 'edge'
            else:
                label = 'other'
        windows.append({'session_id': session_id, 'label': label, 'data': window_data})
    return windows

def extract_idle_windows(data_df, beep_df, session_id, num_windows=5):
    windows = []
    if len(beep_df) < 2:
        data_len = len(data_df)
        step = data_len // (num_windows + 1)
        for i in range(num_windows):
            start_idx = step * (i + 1) - 12
            end_idx = start_idx + 24
            if end_idx < data_len and start_idx >= 0:
                window_data = data_df.iloc[start_idx:end_idx].copy()
                windows.append({'session_id': session_id, 'label': 'idle', 'data': window_data})
        return windows

    beep_times = beep_df['timestamp_ms'].values
    for i in range(len(beep_times) - 1):
        gap_start = beep_times[i] + WINDOW_AFTER_MS + 500
        gap_end = beep_times[i + 1] - WINDOW_BEFORE_MS - 500
        if gap_end - gap_start < WINDOW_SIZE_MS:
            continue
        mid_time = (gap_start + gap_end) / 2
        window_start = mid_time - WINDOW_SIZE_MS / 2
        window_end = mid_time + WINDOW_SIZE_MS / 2
        mask = (data_df['timestamp_ms'] >= window_start) & (data_df['timestamp_ms'] <= window_end)
        window_data = data_df[mask].copy()
        if len(window_data) < 10:
            continue
        windows.append({'session_id': session_id, 'label': 'idle', 'data': window_data})
        if len(windows) >= num_windows:
            break
    return windows

def extract_features(window_data):
    features = {}
    accel_cols = ['accel_x', 'accel_y', 'accel_z']
    gyro_cols = ['gyro_x', 'gyro_y', 'gyro_z']

    accel_mag = np.sqrt(sum(window_data[col]**2 for col in accel_cols))
    gyro_mag = np.sqrt(sum(window_data[col]**2 for col in gyro_cols))

    for col in accel_cols + gyro_cols:
        data = window_data[col].values
        prefix = col
        features[f'{prefix}_mean'] = np.mean(data)
        features[f'{prefix}_std'] = np.std(data)
        features[f'{prefix}_min'] = np.min(data)
        features[f'{prefix}_max'] = np.max(data)
        features[f'{prefix}_range'] = np.max(data) - np.min(data)
        features[f'{prefix}_median'] = np.median(data)
        features[f'{prefix}_iqr'] = np.percentile(data, 75) - np.percentile(data, 25)
        features[f'{prefix}_skew'] = stats.skew(data)
        features[f'{prefix}_kurtosis'] = stats.kurtosis(data)
        zero_crossings = np.sum(np.diff(np.sign(data - np.mean(data))) != 0)
        features[f'{prefix}_zero_cross'] = zero_crossings

        fft_vals = np.abs(fft(data))
        fft_vals = fft_vals[:len(fft_vals)//2]
        if len(fft_vals) > 0:
            features[f'{prefix}_fft_mean'] = np.mean(fft_vals)
            features[f'{prefix}_fft_std'] = np.std(fft_vals)
            features[f'{prefix}_fft_max'] = np.max(fft_vals)
            features[f'{prefix}_fft_energy'] = np.sum(fft_vals**2)
            features[f'{prefix}_fft_dom_idx'] = np.argmax(fft_vals) / len(fft_vals)

    for name, mag in [('accel_mag', accel_mag), ('gyro_mag', gyro_mag)]:
        features[f'{name}_mean'] = np.mean(mag)
        features[f'{name}_std'] = np.std(mag)
        features[f'{name}_max'] = np.max(mag)
        features[f'{name}_min'] = np.min(mag)
        features[f'{name}_range'] = np.max(mag) - np.min(mag)
        features[f'{name}_peak_idx'] = np.argmax(mag) / len(mag)

    features['accel_xy_corr'] = np.corrcoef(window_data['accel_x'], window_data['accel_y'])[0, 1]
    features['accel_xz_corr'] = np.corrcoef(window_data['accel_x'], window_data['accel_z'])[0, 1]
    features['accel_yz_corr'] = np.corrcoef(window_data['accel_y'], window_data['accel_z'])[0, 1]
    features['gyro_xy_corr'] = np.corrcoef(window_data['gyro_x'], window_data['gyro_y'])[0, 1]
    features['gyro_xz_corr'] = np.corrcoef(window_data['gyro_x'], window_data['gyro_z'])[0, 1]
    features['gyro_yz_corr'] = np.corrcoef(window_data['gyro_y'], window_data['gyro_z'])[0, 1]

    for key in features:
        if np.isnan(features[key]):
            features[key] = 0

    return features

if __name__ == '__main__':
    # Extract all windows
    print("="*60)
    print("STEP 1: EXTRACTING WINDOWS")
    print("="*60)
    all_windows = []
    for session_id, info in SESSION_INFO.items():
        data_df, beep_df = load_session(session_id)
        if info['punch_type'] != 'idle':
            punch_windows = extract_punch_windows(data_df, beep_df, info, session_id)
            all_windows.extend(punch_windows)
        idle_windows = extract_idle_windows(data_df, beep_df, session_id, num_windows=5)
        all_windows.extend(idle_windows)

    labels = [w['label'] for w in all_windows]
    print(f'Total windows: {len(all_windows)}')
    print(f'Label distribution:')
    for label in sorted(set(labels)):
        count = labels.count(label)
        print(f'  {label}: {count} ({100*count/len(labels):.1f}%)')

    # Extract features
    print("\n" + "="*60)
    print("STEP 2: EXTRACTING FEATURES")
    print("="*60)
    feature_records = []
    for window in all_windows:
        features = extract_features(window['data'])
        features['label'] = window['label']
        features['session_id'] = window['session_id']
        features['use'] = SESSION_INFO[window['session_id']]['use']
        feature_records.append(features)

    features_df = pd.DataFrame(feature_records)
    print(f'Feature matrix shape: {features_df.shape}')
    print(f'Number of features: {features_df.shape[1] - 3}')

    # Prepare data
    print("\n" + "="*60)
    print("STEP 3: PREPARING DATA")
    print("="*60)
    valid_labels = ['jab', 'hook', 'idle']
    df_filtered = features_df[features_df['label'].isin(valid_labels)].copy()

    print(f'Filtered to {len(df_filtered)} samples (jab/hook/idle only)')
    print(f'Label distribution:')
    print(df_filtered['label'].value_counts())

    train_df = df_filtered[df_filtered['use'] == 'train'].copy()
    holdout_df = df_filtered[df_filtered['use'] == 'holdout'].copy()

    print(f'\nTraining samples: {len(train_df)}')
    print(f'Hold-out samples: {len(holdout_df)}')

    # Prepare features
    feature_cols = [col for col in train_df.columns if col not in ['label', 'session_id', 'use']]

    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values

    X_holdout = holdout_df[feature_cols].values
    y_holdout = holdout_df['label'].values

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_holdout_encoded = label_encoder.transform(y_holdout)

    print(f'Classes: {label_encoder.classes_}')

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_holdout_scaled = scaler.transform(X_holdout)

    # Train models
    print("\n" + "="*60)
    print("STEP 4: TRAINING MODELS")
    print("="*60)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_train_encoded,
        test_size=0.2, random_state=42, stratify=y_train_encoded
    )

    # Random Forest
    print('\nTraining Random Forest...')
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_tr, y_tr)
    rf_val_pred = rf_model.predict(X_val)
    rf_val_acc = accuracy_score(y_val, rf_val_pred)
    print(f'Random Forest Validation Accuracy: {rf_val_acc:.4f}')

    cv_scores = cross_val_score(rf_model, X_train_scaled, y_train_encoded, cv=5)
    print(f'CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})')

    # Gradient Boosting
    print('\nTraining Gradient Boosting...')
    gb_model = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
    gb_model.fit(X_tr, y_tr)
    gb_val_pred = gb_model.predict(X_val)
    gb_val_acc = accuracy_score(y_val, gb_val_pred)
    print(f'Gradient Boosting Validation Accuracy: {gb_val_acc:.4f}')

    cv_scores_gb = cross_val_score(gb_model, X_train_scaled, y_train_encoded, cv=5)
    print(f'CV Mean: {cv_scores_gb.mean():.4f} (+/- {cv_scores_gb.std()*2:.4f})')

    # Select best
    if rf_val_acc >= gb_val_acc:
        best_model = rf_model
        best_name = "Random Forest"
    else:
        best_model = gb_model
        best_name = "Gradient Boosting"

    print(f'\nBest model: {best_name}')
    best_model.fit(X_train_scaled, y_train_encoded)

    # Evaluate
    print("\n" + "="*60)
    print("STEP 5: HOLD-OUT EVALUATION")
    print("="*60)

    y_holdout_pred = best_model.predict(X_holdout_scaled)
    holdout_acc = accuracy_score(y_holdout_encoded, y_holdout_pred)

    print(f'\nHold-out Accuracy: {holdout_acc:.4f}')
    print(f'\nClassification Report:')
    print(classification_report(
        y_holdout_encoded, y_holdout_pred,
        target_names=label_encoder.classes_
    ))

    print('\nConfusion Matrix:')
    cm = confusion_matrix(y_holdout_encoded, y_holdout_pred)
    print(f'           {label_encoder.classes_}')
    for i, row in enumerate(cm):
        print(f'{label_encoder.classes_[i]:>10}: {row}')

    # Feature importance
    print('\nTop 10 Most Important Features:')
    importance = best_model.feature_importances_
    indices = np.argsort(importance)[::-1][:10]
    for i in range(10):
        print(f'  {i+1}. {feature_cols[indices[i]]}: {importance[indices[i]]:.4f}')

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f'Model: {best_name}')
    print(f'Total features: {len(feature_cols)}')
    print(f'Training samples: {len(train_df)}')
    print(f'Hold-out samples: {len(holdout_df)}')
    print(f'Hold-out Accuracy: {holdout_acc:.2%}')
