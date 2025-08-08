import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from group_lasso import LogisticGroupLasso
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress potential convergence warnings from the solver
warnings.filterwarnings("ignore", category=UserWarning)

# --- Data Simulation & Preparation Functions ---

def simulate_fmri_data(n_subs=100, n_features=50, n_groups=5, random_state=42):
    """
    Creates a sample DataFrame that mimics fMRI data structure for testing.
    """
    print("Step 1: Simulating fMRI data...")
    np.random.seed(random_state)
    
    feature_names = [f'ROI_{i+1}' for i in range(n_features)]
    X_data = np.random.randn(n_subs, n_features)
    df = pd.DataFrame(X_data, columns=feature_names)

    y_data = (0.5 * df['ROI_1'] - 0.8 * df['ROI_15'] + np.random.randn(n_subs)) > 0
    df['depression'] = y_data.astype(int)

    group_labels = {f'Group_{g+1}': [] for g in range(n_groups)}
    feature_indices = np.arange(n_features)
    np.random.shuffle(feature_indices)
    group_assignments = np.array_split(feature_indices, n_groups)

    for i, group in enumerate(group_assignments):
        group_name = f'Group_{i+1}'
        group_features = [feature_names[j] for j in group]
        group_labels[group_name] = group_features

    print(f"   - Created DataFrame with shape: {df.shape}")
    print("-" * 30)
    return df, group_labels

def prepare_group_lasso_data(df, group_labels_dict, target_col='depression'):
    """
    Separates features and target, and creates the numerical group array.
    """
    print("Step 2: Preparing data and group structure...")
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    feature_map = {feature: i for i, feature in enumerate(X.columns)}
    groups = np.zeros(len(X.columns), dtype=int)
    
    for group_id, group_name in enumerate(sorted(group_labels_dict.keys())):
        features_in_group = group_labels_dict[group_name]
        for feature in features_in_group:
            if feature in feature_map:
                groups[feature_map[feature]] = group_id
    
    print(f"   - Converted group dictionary to a numerical array of length {len(groups)}.")
    print("-" * 30)
    return X, y, groups

# --- NEW: Core Hold-Out Evaluation Function ---

def run_holdout_evaluation(X, y, groups, param_grid, test_size=0.25, random_state=42, cv=5):
    """
    Splits data into a train and test set, tunes hyperparameters on the train set
    using cross-validation, and evaluates the final model on the hold-out test set.

    This is a standard alternative to nested cross-validation.
    """
    print("Step 4: Running Hold-Out Evaluation...")

    # 1. Split data into a training set and a single hold-out test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"   - Data split into {len(X_train)} training samples and {len(X_test)} test samples.")

    # 2. Set up preprocessor and scale data
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), list(X.columns))],
        remainder='passthrough'
    )
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    print("   - Scaler fitted on training data and applied to both sets.")

    # 3. Tune hyperparameters using cross-validation on the training set
    inner_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    lgl = LogisticGroupLasso(groups=groups, supress_warning=True, n_iter=2000)

    print("   - Tuning hyperparameters with GridSearchCV on the training data...")
    clf = GridSearchCV(
        estimator=lgl, param_grid=param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)

    print(f"   - Best hyperparameter found: {clf.best_params_}")
    print(f"   - Best CV score on training data: {clf.best_score_:.4f}")

    # 4. Evaluate the best model on the hold-out test set
    # clf.best_estimator_ is automatically refit on the entire X_train_scaled set
    final_model = clf.best_estimator_
    y_pred = final_model.predict(X_test_scaled)
    y_pred_proba = final_model.predict_proba(X_test_scaled)[:, 1]

    print("   - Evaluating final model on the hold-out test set.")
    report = get_classification_report(y_test, y_pred, y_pred_proba)

    # Return all necessary components for further analysis
    return final_model, preprocessor, report, (y_test, y_pred, y_pred_proba)


# --- Model Evaluation Functions ---

def get_selected_groups(model, groups, group_labels_dict):
    """
    Identifies the feature groups selected by the fitted GroupLasso model.
    """
    group_names = sorted(group_labels_dict.keys())
    selected_groups = []
    
    for group_id, group_name in enumerate(group_names):
        feature_indices_in_group = np.where(groups == group_id)[0]
        group_coef_sum = np.sum(np.abs(model.coef_[feature_indices_in_group]))
        
        if group_coef_sum > 1e-6:
            selected_groups.append(group_name)
            
    return selected_groups

def get_classification_report(y_true, y_pred, y_pred_proba):
    """
    Calculates and returns a dictionary of key classification metrics.
    """
    report = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, y_pred_proba)
    }
    return report

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Generates and displays a confusion matrix plot.
    """
    if class_names is None:
        class_names = ['Negative', 'Positive']
        
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def plot_roc_curve(y_true, y_pred_proba):
    """
    Generates and displays an ROC curve plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()
