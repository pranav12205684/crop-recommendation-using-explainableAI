import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("C:/Users/Asus/OneDrive/Documents/B Tech/Capstone/Crop_recommendation.csv")

X = df.drop("label", axis=1)
y = df["label"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
class_names = encoder.classes_

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

models = {
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append([name, acc, cv_scores.mean(), cv_scores.std(), prec, rec, f1, roc, kappa, mcc])

df_results = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "CV Mean", "CV Std", "Precision", "Recall", "F1", "ROC-AUC", "Kappa", "MCC"
])

for _, row in df_results.iterrows():
    print(f"Model: {row['Model']}")
    print(f" Accuracy  : {row['Accuracy']:.6f}")
    print(f" CV Mean   : {row['CV Mean']:.6f}")
    print(f" Precision : {row['Precision']:.6f}")
    print(f" Recall    : {row['Recall']:.6f}")
    print(f" F1        : {row['F1']:.6f}")
    print(f" ROC-AUC   : {row['ROC-AUC']:.6f}")
    print(f" Kappa     : {row['Kappa']:.6f}")
    print(f" MCC       : {row['MCC']:.6f}")
    print("-" * 50)

param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [None, 20, 30],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
print("\nBest RF Params:", grid.best_params_)

ensemble = VotingClassifier(
    estimators=[
        ('rf', best_rf),
        ('svm', SVC(probability=True)),
        ('xgb', XGBClassifier(eval_metric='mlogloss', use_label_encoder=False))
    ],
    voting='soft'
)

ensemble.fit(X_train, y_train)

y_pred = ensemble.predict(X_test)
y_prob = ensemble.predict_proba(X_test)

print("\nEnsemble Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

y_cat = to_categorical(y_encoded)

X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
    X_scaled, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
)

model_nn = Sequential([
    Input(shape=(X_train_nn.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(y_cat.shape[1], activation='softmax')
])

model_nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history_obj = model_nn.fit(
    X_train_nn, y_train_nn,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early],
    verbose=1
)

loss, acc = model_nn.evaluate(X_test_nn, y_test_nn)
print("\nNeural Network Accuracy:", acc)

shap.initjs()

explainer = shap.TreeExplainer(best_rf)
X_sample = X_test[:200]
feature_names = X.columns.tolist()

shap_values = explainer.shap_values(X_sample)

if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
    shap_values = np.transpose(shap_values, (2, 0, 1))
    shap_values = [shap_values[i] for i in range(shap_values.shape[0])]

shap_values_array = np.array(shap_values)
global_shap_values = np.abs(shap_values_array).mean(axis=0)

shap.summary_plot(global_shap_values, X_sample, feature_names=feature_names)
crop_to_explain = 21
shap.summary_plot(shap_values[crop_to_explain], X_sample, feature_names=feature_names)


# Model Performance (RF, XGB, NB)
selected_models = ["Random Forest", "XGBoost", "Naive Bayes"]
df_filtered = df_results[df_results["Model"].isin(selected_models)]

x = np.arange(len(df_filtered))

plt.figure(figsize=(10,6))
plt.bar(x, df_filtered["Accuracy"], width=0.4, label="Accuracy")
plt.plot(x, df_filtered["F1"], marker='o', linewidth=2, label="F1 Score")
plt.plot(x, df_filtered["ROC-AUC"], marker='s', linewidth=2, label="ROC-AUC")

plt.xticks(x, df_filtered["Model"])
plt.ylabel("Score")
plt.title("Model Performance Comparison (RF vs XGB vs NB)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# MCC & Kappa
x = np.arange(len(df_filtered))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, df_filtered["MCC"], width, label="MCC")
plt.bar(x + width/2, df_filtered["Kappa"], width, label="Kappa")

plt.xticks(x, df_filtered["Model"])
plt.ylabel("Score")
plt.title("MCC & Kappa Comparison (RF vs XGB vs NB)")
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Grouped Confusion Matrix
crop_groups = {
    "Cereals": ["rice", "maize"],
    "Pulses": ["chickpea", "lentil", "blackgram", "mungbean", "pigeonpeas"],
    "Fruits": ["apple", "banana", "grapes", "mango", "orange", "papaya"],
    "Cash Crops": ["cotton", "jute", "coffee"],
    "Vegetables": ["watermelon", "muskmelon"],
    "Plantation": ["coconut"],
    "Spices": ["kidneybeans"]
}

label_to_group = {}
for group, crops in crop_groups.items():
    for crop in crops:
        if crop in class_names:
            idx = np.where(class_names == crop)[0][0]
            label_to_group[idx] = group

y_test_group = [label_to_group.get(i, "Other") for i in y_test]
y_pred_group = [label_to_group.get(i, "Other") for i in y_pred]

group_names = list(set(label_to_group.values()))
cm_group = confusion_matrix(y_test_group, y_pred_group, labels=group_names)

plt.figure(figsize=(8,6))
disp = ConfusionMatrixDisplay(cm_group, display_labels=group_names)
disp.plot(xticks_rotation=30)
plt.title("Grouped Confusion Matrix")
plt.tight_layout()
plt.show()

# Training vs Validation Accuracy
history = history_obj.history

plt.figure(figsize=(8,5))
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Training vs Validation Loss
plt.figure(figsize=(8,5))
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()