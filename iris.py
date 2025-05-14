# Iris Dataset Classification with Visualization and Analysis

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

# Set up the plotting style
plt.style.use('seaborn-v0_8')
colors = ['navy', 'turquoise', 'darkorange']

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create a DataFrame for easier data handling
df = pd.DataFrame(data=np.c_[X, y], columns=feature_names + ['target'])
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.fit_transform(X)  # Scale the entire dataset for cross-validation

# Function to display confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.tight_layout()
    return cm

# Function to plot decision regions
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02, title="Decision Boundary"):
    # Setup marker generator and color map
    markers = ('o', 's', '^')
    
    # Plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=ListedColormap(colors))
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # Plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                   alpha=0.8, c=[colors[idx]], marker=markers[idx],
                   label=target_names[cl], edgecolor='black')
    
    # Highlight test samples if provided
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0,
                   edgecolor='black', linewidth=1, marker='o',
                   s=100, label='test set')
    
    plt.title(title)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend()
    
# Initialize different classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(kernel='rbf', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Train models and store results
results = {}
confusion_matrices = {}

for name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'classification_report': classification_report(y_test, y_pred, target_names=target_names),
        'classifier': clf,
        'predictions': y_pred
    }
    
    # Store confusion matrix
    confusion_matrices[name] = confusion_matrix(y_test, y_pred)

# Exploratory Data Analysis
plt.figure(figsize=(20, 15))

# Scatter plot matrix
pd.plotting.scatter_matrix(df[feature_names], figsize=(15, 15), 
                          c=y, marker='o', hist_kwds={'bins': 20}, 
                          s=60, alpha=0.8, cmap='viridis')  # Fixed cmap usage
plt.suptitle('Scatter Plot Matrix of Iris Dataset Features', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.95)

# Boxplots for each feature by species
plt.figure(figsize=(14, 10))
for i, feature in enumerate(feature_names):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='species', y=feature, data=df)
    plt.title(f'Boxplot of {feature} by Species')
plt.tight_layout()

# Violin plots for each feature by species
plt.figure(figsize=(14, 10))
for i, feature in enumerate(feature_names):
    plt.subplot(2, 2, i+1)
    sns.violinplot(x='species', y=feature, data=df)
    plt.title(f'Violin Plot of {feature} by Species')
plt.tight_layout()

# Pairplot
sns.pairplot(df, hue='species', vars=feature_names)
plt.suptitle('Pairplot of Iris Dataset', y=1.02, fontsize=16)

# Perform PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_

# Plot PCA results
plt.figure(figsize=(10, 8))
for i, target_name in enumerate(target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], alpha=0.8, 
                color=colors[i], label=target_name, edgecolor='k')
plt.xlabel(f'First Principal Component (Explains {explained_variance[0]:.2%} of variance)')
plt.ylabel(f'Second Principal Component (Explains {explained_variance[1]:.2%} of variance)')
plt.title('PCA of Iris Dataset')
plt.legend()
plt.tight_layout()

# Find the best classifier based on accuracy
best_classifier_name = max(results, key=lambda k: results[k]['accuracy'])
best_classifier = results[best_classifier_name]['classifier']
best_accuracy = results[best_classifier_name]['accuracy']

# Visualize decision boundaries using the best classifier for two features
# We'll use sepal length and sepal width, as they are the most commonly visualized
X_subset = X[:, :2]  # First two features

# Plot model comparison
plt.figure(figsize=(12, 8))
model_names = list(results.keys())
accuracy_scores = [results[name]['accuracy'] for name in model_names]
bar_colors = ['blue', 'green', 'red', 'purple', 'orange']
plt.bar(model_names, accuracy_scores, color=bar_colors)
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Model Comparison by Accuracy')
plt.ylim(0.7, 1.0)  # Adjust the y-axis for better visualization
for i, v in enumerate(accuracy_scores):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
plt.tight_layout()

# Visualize decision boundaries for best classifier
plt.figure(figsize=(12, 6))

# First two features (sepal length and width)
plt.subplot(1, 2, 1)
best_clf_subset = classifiers[best_classifier_name]
best_clf_subset.fit(X_subset, y)
plot_decision_regions(X_subset, y, best_clf_subset, 
                     title=f'Decision Boundary ({best_classifier_name})\nSepal Features')

# Perform PCA and use the top two components for another decision boundary
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.subplot(1, 2, 2)
best_clf_pca = classifiers[best_classifier_name]
best_clf_pca.fit(X_pca, y)
plot_decision_regions(X_pca, y, best_clf_pca, 
                     title=f'Decision Boundary ({best_classifier_name})\nPCA Components')
plt.tight_layout()

# Plot confusion matrices for all classifiers
plt.figure(figsize=(15, 12))
for i, (name, cm) in enumerate(confusion_matrices.items()):
    plt.subplot(3, 2, i+1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix: {name}')
plt.tight_layout()

# Calculate cross-validation scores for all classifiers
cv_results = {}
for name, clf in classifiers.items():
    cv_scores = cross_val_score(clf, X_scaled, y, cv=5)
    cv_results[name] = cv_scores
    
# Plot cross-validation results
plt.figure(figsize=(12, 6))
plt.boxplot([cv_results[name] for name in cv_results.keys()], labels=cv_results.keys())
plt.title('Cross-Validation Results (5-fold)')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()

# Display feature importances for tree-based models
plt.figure(figsize=(12, 5))
tree_models = ['Decision Tree', 'Random Forest']
for i, model_name in enumerate(tree_models):
    if model_name in results:
        model = results[model_name]['classifier']
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.subplot(1, 2, i+1)
        plt.bar(range(X.shape[1]), importances[indices], align='center')
        plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
        plt.title(f'Feature Importance: {model_name}')
plt.tight_layout()

# Function to analyze and interpret the confusion matrix
def analyze_confusion_matrix(cm, target_names):
    n_classes = len(target_names)
    # Calculate metrics per class
    metrics = {cls: {} for cls in target_names}
    
    for i in range(n_classes):
        # True positives, false positives, false negatives, true negatives
        tp = cm[i, i]
        fp = sum(cm[:, i]) - tp
        fn = sum(cm[i, :]) - tp
        tn = cm.sum() - tp - fp - fn
        
        # Store metrics
        metrics[target_names[i]] = {
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'misclassification_rate': (fp + fn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
        }
    
    return metrics

# Analyze the confusion matrix of the best classifier
best_cm = confusion_matrices[best_classifier_name]
cm_analysis = analyze_confusion_matrix(best_cm, target_names)

# Learning curves for the best model
plt.figure(figsize=(10, 6))
train_sizes, train_scores, test_scores = learning_curve(
    best_classifier, X_scaled, y, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='blue', marker='o', 
         markersize=5, label='Training accuracy')
plt.fill_between(train_sizes, train_mean - train_std, 
                 train_mean + train_std, alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean, color='green', marker='s', 
         markersize=5, label='Validation accuracy')
plt.fill_between(train_sizes, test_mean - test_std,
                 test_mean + test_std, alpha=0.15, color='green')

plt.title('Learning Curves for Best Classifier')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()

# Display results
print(f"Best Classifier: {best_classifier_name} with accuracy: {best_accuracy:.4f}")
print("\nClassification Report:")
print(results[best_classifier_name]['classification_report'])

# Print results for all models
print("\nResults for all models:")
for name in results:
    print(f"{name}: {results[name]['accuracy']:.4f}")

# Print confusion matrix analysis
print("\nConfusion Matrix Analysis for the Best Classifier:")
for cls, metrics in cm_analysis.items():
    print(f"\nMetrics for {cls}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

# Calculate overall accuracy
overall_accuracy = np.trace(best_cm) / np.sum(best_cm)
print(f"\nOverall Accuracy: {overall_accuracy:.4f}")

# Show all plots
plt.show()
