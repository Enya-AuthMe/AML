import itertools

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_score, recall_score)
from sklearn.model_selection import KFold, cross_val_score, train_test_split

trainset_path = './Class_Dataset/trainset/trainset_ex1.csv'
testset_path = './Class_Dataset/testset/testset_ex1.csv'
submi_path = './submission_format.csv'
df = pd.read_csv(trainset_path)  # 22744
df_test_org = pd.read_csv(testset_path)
df_test = df_test_org.drop(['alert_key'], axis=1, inplace=False)
submi = pd.read_csv(submi_path)


def printing_Kfold_scores(x_train_data, y_train_data):
    kf = KFold(n_splits=5)
    recall_accs = []
    precision_accs = []
    for iteration, indices in enumerate(kf.split(y_train_data), start=1):
        model = GradientBoostingClassifier(learning_rate=0.1, max_depth=100)
        model.fit(x_train_data.iloc[indices[0], :].values,
                  y_train_data.iloc[indices[0], :].values.ravel())
        y_pred = model.predict(x_train_data.iloc[indices[1], :].values)
        # breakpoint()
        recall_acc = recall_score(
            y_train_data.iloc[indices[1], :].values, y_pred)
        precision_acc = precision_score(
            y_train_data.iloc[indices[1], :].values, y_pred)
        recall_accs.append(recall_acc)
        precision_accs.append(precision_acc)
        print('Iteration %d : recall score = %.3f precision score = %.3f' %
              (iteration, recall_acc, precision_acc))
    joblib.dump(model, 'ex1_2_GBDT')
    mean_recall_score = np.mean(recall_accs)
    mean_precision_score = np.mean(precision_accs)
    print('==='*15)
    print('Mean recall score %.3f' % mean_recall_score)
    print('Mean precision score %.3f' % mean_precision_score)
    print('==='*15)
    return mean_recall_score


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def split_original():
    X = df.loc[:, df.columns != 'sar']
    Y = df.loc[:, df.columns == 'sar']

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=None, shuffle=True)
    return X_train, X_test, Y_train, Y_test


def split_undersampling():
    df_split = pd.concat([X_train, Y_train], axis=1)
    num_abn = len(df_split.loc[df_split['sar'] == 1])
    num_nor = len(df_split.loc[df_split['sar'] == 0])
    idx_abn = df_split.loc[df_split['sar'] == 1].index.values
    idx_nor = df_split.loc[df_split['sar'] == 0].index.values
    rand_idx_nor = np.random.choice(idx_nor, num_abn, replace=False)

    undersample_idx = np.concatenate([idx_abn, rand_idx_nor])
    undersample_df = df_split.loc[undersample_idx]

    X_undersample = undersample_df.loc[:, undersample_df.columns != 'sar']
    Y_undersample = undersample_df.loc[:, undersample_df.columns == 'sar']

    X_train_undersample, X_test_undersample, Y_train_undersample, Y_test_undersample = train_test_split(
        X_undersample, Y_undersample, test_size=0.00001, random_state=None, shuffle=True)
    return X_train_undersample, X_test_undersample, Y_train_undersample, Y_test_undersample


# original
X_train, X_test, Y_train, Y_test = split_original()
origin = printing_Kfold_scores(X_train, Y_train)

# undersample
X_train_undersample, X_test_undersample, Y_train_undersample, Y_test_undersample = split_undersampling()
# undersample = printing_Kfold_scores(X_train_undersample, Y_train_undersample)

# testset prediction
trained_model = joblib.load('ex1_2_GBDT')
Y_pred = trained_model.predict(X_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test, Y_pred)
np.set_printoptions(precision=2)

print("------>Recall_0 :    %.3f" % recall_score(Y_test, Y_pred, labels=0))

test_recall_0 = cnf_matrix[0, 0]/(cnf_matrix[0, 0]+cnf_matrix[0, 1])
test_precision_0 = cnf_matrix[0, 0]/(cnf_matrix[0, 0]+cnf_matrix[1, 0])
test_recall_1 = cnf_matrix[1, 1]/(cnf_matrix[1, 1]+cnf_matrix[1, 0])
test_precision_1 = cnf_matrix[1, 1]/(cnf_matrix[1, 1]+cnf_matrix[0, 1])
print("testing dataset :")
print("Recall_0 :    %.3f" % test_recall_0)
print("Precision_0 : %.3f" % test_precision_0)
print("Recall_1 :    %.3f" % test_recall_1)
print("Precision_1 : %.3f" % test_precision_1)
print('==='*15)

# Plot non-normalized confusion matrix
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
# plt.show()
breakpoint()

# # contest prediction
# X_contest = df_test
# Y_contest_pred = lr.predict_proba(X_contest.values)[:, 1]

# for i, prob in enumerate(Y_contest_pred):
#     key = df_test_org.loc[i, 'alert_key']
#     submi.loc[submi['alert_key'] == key, 'probability'] = prob


breakpoint()
