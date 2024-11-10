from imblearn.over_sampling import SMOTE

def balance_dataset(features, labels):
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(features, labels)
    return X_resampled, y_resampled
