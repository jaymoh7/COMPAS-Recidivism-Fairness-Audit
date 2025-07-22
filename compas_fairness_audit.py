
# COMPAS Dataset Fairness Audit using AI Fairness 360

import pandas as pd
from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the COMPAS dataset
dataset = CompasDataset()

# Display basic information
print(f"Dataset features: {dataset.features.shape}")
print(f"Protected attribute: {dataset.protected_attribute_names}")
print(f"Favorable label: {dataset.favorable_label}")
print(f"Unfavorable label: {dataset.unfavorable_label}")

# Measure original bias
metric = BinaryLabelDatasetMetric(dataset, unprivileged_groups=[{'race': 1}], privileged_groups=[{'race': 0}])
print("Disparate impact (original):", metric.disparate_impact())

# Reweighing for bias mitigation
RW = Reweighing(unprivileged_groups=[{'race': 1}], privileged_groups=[{'race': 0}])
dataset_transf = RW.fit_transform(dataset)

# Prepare data for model training
X = StandardScaler().fit_transform(dataset_transf.features)
y = dataset_transf.labels.ravel()

# Train a simple classifier
clf = LogisticRegression(solver='liblinear')
clf.fit(X, y)
y_pred = clf.predict(X)

# Create predicted dataset
dataset_pred = dataset_transf.copy()
dataset_pred.labels = y_pred

# Fairness evaluation post-mitigation
classified_metric = ClassificationMetric(
    dataset_transf, dataset_pred,
    unprivileged_groups=[{'race': 1}],
    privileged_groups=[{'race': 0}]
)

print("Disparate impact (after mitigation):", classified_metric.disparate_impact())
print("False positive rate difference:", classified_metric.false_positive_rate_difference())

# Visualization
metrics = {
    "Disparate Impact": [metric.disparate_impact(), classified_metric.disparate_impact()],
    "FPR Difference": [0, classified_metric.false_positive_rate_difference()]
}

df_metrics = pd.DataFrame(metrics, index=["Original", "After Mitigation"])
df_metrics.plot(kind="bar", figsize=(8, 6), title="Fairness Metrics Before and After Mitigation")
plt.xticks(rotation=0)
plt.ylabel("Metric Value")
plt.grid(True)
plt.tight_layout()
plt.savefig("compas_fairness_metrics.png")
plt.show()
