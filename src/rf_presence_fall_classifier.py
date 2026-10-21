from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import emlearn
import os

# Function to get a unique filename for the converted model in C code
def get_unique_filename(base_name, extension='h'):
    counter = 1
    new_filename = f"{base_name}.{extension}"
    
    while os.path.exists("model/"+new_filename):
        new_filename = f"{base_name}_{counter}.{extension}"
        counter += 1
        
    return new_filename

ACTIONS = ["walking", "bending", "drinking", "lying", "sitting", "falling"]
MIN_LEAF = 0.05

X = pd.read_csv("dataset-a121/synth_dataset_A121.csv")
Y = pd.read_csv("dataset-a121/synth_dataset_A121_targets.csv")

# Split Train/Test
X_train, X_test, y_train, y_test= train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=36)

print("\n--- Training of the RF classifier ---")


clf = RandomForestClassifier(n_estimators=10,max_depth=10, warm_start=True, random_state=48, min_samples_leaf=MIN_LEAF, bootstrap=True)
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
disp = ConfusionMatrixDisplay(confusion_matrix=cfm_norm, display_labels=ACTIONS)
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)

plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

trees = []
for t in clf.estimators_:
    assert isinstance(t, DecisionTreeClassifier)
    trees.append(t.tree_)          # <-- serve .tree_

n_nodes  = sum(t.node_count for t in trees)
n_leaves = sum(int((t.children_left == -1).sum()) for t in trees)
n_int    = n_nodes - n_leaves
flash_bytes = n_int * 7 + n_leaves * 6 * 4
print(f"Number of nodes: {n_nodes} \n Number of leaves: {n_leaves} \n Memory footprint: {flash_bytes}")

# Trained model dump in order to be loaded into the MCU
filename = get_unique_filename("fall_clf","h")
embedded_rf = emlearn.convert(clf, method="inline")
embedded_rf.save(file="model/"+filename, name="fall_clf")