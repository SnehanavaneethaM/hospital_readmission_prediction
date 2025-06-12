import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Load Data
df = pd.read_csv("cleaned_diabetic_data.csv", encoding='latin1')

# 2. Map target values
df['readmitted'] = df['readmitted'].map({'<30': 1, '>30': 0, 'NO': 0})
df = df[df['readmitted'].notnull()]

# 3. Drop columns
if 'readmit_flag' in df.columns:
    df.drop(['readmit_flag'], axis=1, inplace=True)
df.dropna(axis=1, thresh=len(df)*0.8, inplace=True)

# 4. Select only 8 features
selected_features = ['age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
                     'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient']
X = df[selected_features]
y = df['readmitted']

# 5. Encode non-numeric
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# 6. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 7. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Models
lr = LogisticRegression(solver='saga', max_iter=3000)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

# 9. Accuracy
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))

# 10. Feature Importance
feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
top_features = feature_importance.sort_values(ascending=False)

# 11. Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 8 Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()

# 7. Save model and scaler
import joblib
import pickle
joblib.dump(rf, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and Scaler saved with 8 input features")