from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ACTIONS = ["walking", "bending", "drinking", "lying", "sitting", "falling"]

NUM_TREES = [10, 20, 30, 40, 50, 100, 150, 200]
oobscore = []

X = pd.read_csv("dataset-a121/synth_dataset_A121.csv")
Y = pd.read_csv("dataset-a121/synth_dataset_A121_targets.csv")

# Split Train/Test
X_train, X_test, y_train, y_test= train_test_split(X, Y, test_size=0.3, stratify=Y)

print("\n--- Addestramento Classificatore Random Forest ---")

for i in NUM_TREES:

    clf = RandomForestClassifier(n_estimators=i, max_depth=20, random_state=38,min_samples_leaf=30, bootstrap=True, oob_score=True)
    clf.fit(X_train, y_train)


    # Testing on the model
    y_pred = clf.predict(X_test)

    print("\n--- Risultati della Classificazione in training ---")
    print(classification_report(y_test, y_pred, target_names=ACTIONS))

    # Confusion matrix
    cfm = confusion_matrix(y_test, y_pred)

    # Normalize by rows (True class)
    cfm_norm = cfm.astype("float") / cfm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix on the test dataset
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cfm_norm, display_labels=ACTIONS, )
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()