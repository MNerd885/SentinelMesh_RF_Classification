from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

ACTIONS = ["walking", "bending", "drinking", "lying", "sitting", "falling"]

X = pd.read_csv("dataset-a121/synth_dataset_A121.csv")
Y = pd.read_csv("dataset-a121/synth_dataset_A121_targets.csv")

# Split Train/Test
X_train, X_test, y_train, y_test= train_test_split(X, Y, test_size=0.25, stratify=Y)

print("\n--- 2. Addestramento Classificatore Random Forest ---")
clf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=38,min_samples_leaf=30)
clf.fit(X_train, y_train)

# Valutazione Modello
y_pred = clf.predict(X_test)

print("\n--- 3. Risultati della Classificazione in training ---")
print(classification_report(y_test, y_pred, target_names=ACTIONS))

# Confusion matrix
cfm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix on the test dataset
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cfm, display_labels=ACTIONS)
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)

plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()