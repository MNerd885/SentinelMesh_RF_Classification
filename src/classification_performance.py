from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ACTIONS = ["walking", "bending", "drinking", "lying", "sitting", "falling"]
features_names = [
        "mean_total_energy",
        "mean_doppler_centroid",
        "mean_doppler_spread",
        "mean_max_doppler_bin",
        "mean_spectral_entropy",
        "mean_high_doppler_ratio",
        "mean_mean_range",
        "std_total_energy",
        "std_doppler_centroid",
        "std_doppler_spread",
        "std_max_doppler_bin",
        "std_spectral_entropy",
        "std_high_doppler_ratio",
        "std_mean_range",
        "max_total_energy",
        "max_doppler_centroid",
        "max_doppler_spread",
        "max_max_doppler_bin",
        "max_spectral_entropy",
        "max_high_doppler_ratio",
        "max_mean_range",
        "raw_mean_range", 
        "raw_range_spread", 
        "raw_range_span", 
        "range_excursion"
    ]

oobscore = []

num_Trees = [10, 30, 50, 60, 70, 100, 120, 150, 200]
MIN_LEAF = 5

X = pd.read_csv("dataset-a121/synth_dataset_A121_2.csv")
Y = pd.read_csv("dataset-a121/synth_dataset_A121_targets_2.csv")

# Split Train/Test
X_train, X_test, y_train, y_test= train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=36)

print("\n--- Training of the RF classifier ---")

clf = RandomForestClassifier(max_depth=10, warm_start=True, random_state=48,min_samples_leaf=MIN_LEAF, bootstrap=True, oob_score=True)
for trees in num_Trees:

    clf.set_params(n_estimators=trees)
    clf.fit(X_train, y_train.values.ravel())

    # Record of the oob error for every number of Trees
    oob_error = 1 - clf.oob_score_
    oobscore.append((trees,oob_error))


# Testing on the model
y_pred = clf.predict(X_test)

print("\n--- Risultati della Classificazione in testing ---")
print(classification_report(y_test, y_pred, target_names=ACTIONS))

# Confusion matrix
cfm = confusion_matrix(y_test, y_pred)

# Normalize by rows (True class)
cfm_norm = cfm.astype("float") / cfm.sum(axis=1)[:, np.newaxis]

# Plot the OOB score 
_, _ = plt.subplots(figsize=(8, 6))
x, y = zip(*oobscore)
plt.plot(x,y)

plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")

# Plot confusion matrix on the test dataset
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cfm_norm, display_labels=ACTIONS, )
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)

plt.title("Confusion Matrix")
plt.tight_layout()

# Feature importance plot
importance = clf.feature_importances_
idxs = np.argsort(importance)

plt.figure(3)
plt.title("Feature importance")
plt.barh(range(len(idxs)), importance[idxs], color='b', align="center")
plt.yticks(range(len(idxs)), [features_names[i] for i in idxs])
plt.xlabel('Relative Importance')
plt.show()