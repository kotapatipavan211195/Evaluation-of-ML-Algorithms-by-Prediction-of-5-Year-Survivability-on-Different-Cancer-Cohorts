# Description: This code is used to evaluate the performance of different machine learning models on different datasets.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import roc_auc_score, auc, roc_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Define paths
path_in = "/Users/kotapatitejavenkatpavan/Downloads/"
path_out = "/Users/kotapatitejavenkatpavan/Desktop/python/Projects/"
results = "/Users/kotapatitejavenkatpavan/Desktop/python/Projects/"

# Load datasets
datasets = {
    "Breast Cancer": pd.read_csv(path_in + "breast_spr.csv")
    # ,
    # "Colrectal Cancer": pd.read_csv(path_in + "colrect_spr.csv"),
    # "Digothr Cancer": pd.read_csv(path_in + "digothr_spr.csv"),
    # "Female Genital Cancer": pd.read_csv(path_in + "femgen_spr.csv"),
    # "Male Genital Cancer": pd.read_csv(path_in + "malegen_spr.csv"),
    # "Respiratory Cancer": pd.read_csv(path_in + "respir_spr.csv"),
    # "Urinary Cancer": pd.read_csv(path_in + "urinary_spr.csv")
}

# Define models
models = {
    "DT_gini": DecisionTreeClassifier(criterion="gini", min_samples_leaf=20),
    "DT_entropy": DecisionTreeClassifier(criterion="entropy", min_samples_leaf=10),
    "LR": LogisticRegression(max_iter=10000),
    "NB": GaussianNB(),
    "SVM": LinearSVC(dual=False, max_iter=10000),
    "RF_gini": RandomForestClassifier(criterion='gini', n_estimators=500, max_leaf_nodes=1000, n_jobs=-1),
    "RF_entropy": RandomForestClassifier(criterion='entropy', n_estimators=100, max_leaf_nodes=1000, n_jobs=-1),
    "MLP": MLPClassifier(solver='adam', hidden_layer_sizes=(10,10), max_iter=10000),
    "ADA": AdaBoostClassifier(),
    "BGC": BaggingClassifier(),
    "GBC": GradientBoostingClassifier()
}

# Define sampling techniques
sampling_techniques = ["No_Sampling", "Over_Sampling", "Under_Sampling"]

# Define stratified 10-fold cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=False)

def get_data(data_key):
    return datasets[data_key].values[:, :-1], datasets[data_key].values[:, -1]

def apply_sampling(X_train, Y_train, sample):
    if sample == "Over_Sampling":
        ros = RandomOverSampler()
        return ros.fit_resample(X_train, Y_train)
    elif sample == "Under_Sampling":
        rus = RandomUnderSampler()
        return rus.fit_resample(X_train, Y_train)
    return X_train, Y_train

def train_and_evaluate_model(X, Y, model, sample):
    nrl_confusion_matrix = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0

    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        X_train, Y_train = apply_sampling(X_train, Y_train, sample)
        
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        
        nrl_confusion_matrix.append(confusion_matrix(Y_test, Y_pred))
        fpr, tpr, _ = roc_curve(Y_test, Y_pred)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.3f)' % (i, roc_auc))
        i += 1

    plot_roc_curve(tprs, mean_fpr, aucs)
    calculate_metrics(nrl_confusion_matrix)

def plot_roc_curve(tprs, mean_fpr, aucs):
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
    plt.plot(mean_fpr, mean_tpr, color='blue',
             label=r'Mean ROC (AUC = %0.3f $\pm$ %0.4f)' % (mean_auc, std_auc), lw=3, alpha=.8)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

def calculate_metrics(confusion_matrices):
    nrl_matrix_score = np.sum(confusion_matrices, axis=0)
    TN, FP, FN, TP = nrl_matrix_score.ravel()
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    f1_score = 2 * TP / (2 * TP + FP + FN)

    print(f'Sensitivity: {sensitivity:.5f}')
    print(f'Accuracy: {accuracy:.5f}')
    print(f'Specificity: {specificity:.5f}')
    print(f'Precision: {precision:.5f}')
    print(f'F1_score: {f1_score:.5f}')

def main():
    for data_key in datasets.keys():
        X, Y = get_data(data_key)
        for sample in sampling_techniques:
            for model_key, model in models.items():
                print(f"Evaluating {data_key} with {sample} and {model_key}")
                train_and_evaluate_model(X, Y, model, sample)

if __name__ == "__main__":
    main()
