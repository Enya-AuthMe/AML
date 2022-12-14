import pandas as pd
import numpy as np
import itertools
import joblib
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

trainset_path = './Class_Dataset/trainset/trainset_ex1.csv'
testset_path = './Class_Dataset/testset/testset_ex1.csv'
submi_path = '/Users/yuj/repo/SAR/submi/submission_format.csv'
df = pd.read_csv(trainset_path)  # 22744
df_test_org = pd.read_csv(testset_path)
df_test = df_test_org.drop(['alert_key'], axis=1, inplace=False)
submi = pd.read_csv(submi_path)
model_name = './ex1_1_LR'


def contest_eval(y_pred, y_proba, y_test):
    dict_eval = {'predict': y_pred,
                 'probability': y_proba[:, 1],
                 'real': y_test}
    df_eval = pd.DataFrame(dict_eval)
    N = len(df_eval.loc[df_eval['real'] == 1])
    df_eval.sort_values('probability', ascending=False, inplace=True)
    df_eval.reset_index(drop=True, inplace=True)
    pred = df_eval[df_eval['real'] == 1].index[-2] + 1
    evaluation = (N-1) / (pred)
    return evaluation


def printing_Kfold_scores(x_train_data, y_train_data):
    kf = KFold(n_splits=5)
    recalls = []
    precisions = []
    f1s = []
    evaluations = []
    high_score = 0
    print('Iteration | recall score | precision score |   f1   | contest score')
    for iteration, indices in enumerate(kf.split(y_train_data), start=1):
        model = LogisticRegression()
        model.fit(x_train_data.iloc[indices[0], :].values,
                  y_train_data.iloc[indices[0], :].values.ravel())
        y_pred = model.predict(x_train_data.iloc[indices[1], :].values)
        y_proba = model.predict_proba(x_train_data.iloc[indices[1], :].values)
        recall = recall_score(y_train_data.iloc[indices[1], :].values, y_pred)
        precision = precision_score(
            y_train_data.iloc[indices[1], :].values, y_pred)
        evaluation = contest_eval(
            y_pred, y_proba, y_train_data.iloc[indices[1], :].values.ravel())
        f1 = f1_score(y_train_data.iloc[indices[1], :].values, y_pred)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
        evaluations.append(evaluation)
        print('%d           %.3f           %.3f      %.3f       %.3f' %
              (iteration, recall, precision, f1, evaluation))
        now_score = evaluation
        if now_score > high_score:
            joblib.dump(model, model_name)
            high_score = now_score
    mean_recall_score = np.mean(recalls)
    mean_precision_score = np.mean(precisions)
    mean_contest_eval_score = np.mean(evaluations)
    print('==='*15)
    print('Mean recall score %.3f' % mean_recall_score)
    print('Mean precision score %.3f' % mean_precision_score)
    print('Mean contest evaluation score %.3f' % mean_contest_eval_score)
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


def plt_pca(X_train):
    pca = PCA(n_components=3, random_state=9527)
    X_pca = pca.fit_transform(X_train)
    fig = plt.figure(figsize=(6, 6))
    pca_plt = fig.add_subplot(111, projection='3d')
    pca_plt.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], linewidths=1,
                    alpha=.7, edgecolors='k', s=200, c=Y_train.values.tolist())


def plt_tsne(X_train):
    tsne = manifold.TSNE(n_components=3, init='random',
                         random_state=5, verbose=1)
    X_tsne = tsne.fit_transform(X_train)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    fig = plt.figure(figsize=(6, 6))
    tsne_plt = fig.add_subplot(111, projection='3d')
    tsne_plt.scatter(X_norm[:, 0], X_norm[:, 1], X_norm[:, 2], linewidths=1,
                     alpha=.7, edgecolors='k', s=200, c=Y_train.values.tolist())


def print_result(real_label, pred_label, evaluation):
    print("testing dataset :")
    print("--- pos_label=0 ---")
    print("Recall    : %.3f" % recall_score(
        real_label, pred_label, pos_label=0))
    print("Precision : %.3f" % precision_score(
        real_label, pred_label, pos_label=0))
    print('f1        : %.3f' % f1_score(real_label, pred_label, pos_label=0))
    print("--- pos_label=1 ---")
    print("Recall    : %.3f" % recall_score(
        real_label, pred_label, pos_label=1))
    print("Precision : %.3f" % precision_score(
        real_label, pred_label, pos_label=1))
    print("f1        : %.3f" % f1_score(real_label, pred_label, pos_label=1))
    print("--- Evaluation ---")
    print("eval      : %.3f" % evaluation)
    print('==='*15)


# original
X_train, X_test, Y_train, Y_test = split_original()
# origin = printing_Kfold_scores(X_train, Y_train)

# undersample
X_train_undersample, X_test_undersample, Y_train_undersample, Y_test_undersample = split_undersampling()
undersample = printing_Kfold_scores(X_train_undersample, Y_train_undersample)

# t-SNE
# plt_tsne(X_train=X_train)

# PCA
# plt_pca(X_train=X_train)

# testset prediction
trained_model = joblib.load(model_name)
Y_pred = trained_model.predict(X_test.values)
Y_proba = trained_model.predict_proba(X_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test, Y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
# plt.show()

# contest eval
evaluation = contest_eval(Y_pred, Y_proba, Y_test.to_numpy().reshape(-1))

print_result(Y_test, Y_pred, evaluation)

# contest prediction
X_contest = df_test
Y_contest_pred = trained_model.predict_proba(X_contest.values)[:, 1]

for i, prob in enumerate(Y_contest_pred):
    key = df_test_org.loc[i, 'alert_key']
    submi.loc[submi['alert_key'] == key, 'probability'] = prob


breakpoint()
