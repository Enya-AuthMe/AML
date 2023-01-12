import pandas as pd
import numpy as np
import authmecv as acv
import itertools
import joblib
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from pretty_confusion_matrix import pp_matrix

DIR = acv.get_curdir(__file__)
DATA_ROOT = DIR.parent / '訓練資料集_first'
DATA = {
    'train': 'ex1/trainset/trainset_ex1_minmax_2.csv',
    'test': 'ex1/testset/testset_ex1_minmax_2.csv',
    'submi': 'submi/submission_format.csv',
    'answer': 'public_y_answer.csv'
}
df_list = acv.PowerDict({k: pd.read_csv(str(DATA_ROOT / v))
                        for k, v in DATA.items()})

df_org = df_list.train
df = df_org.drop(columns=['alert_key'])
df_test_org = df_list.test
df_test = df_test_org.drop(columns=['alert_key'])
submi = df_list.submi
model_name = './ex1_1_2'
np.random.seed(0)


def contest_eval(y_proba, y_test):
    dict_eval = {'probability': y_proba[:, 1],
                 'real': y_test}
    df_eval = pd.DataFrame(dict_eval)
    N = len(df_eval[df_eval['real'] == 1])
    df_eval.sort_values('probability', ascending=False, inplace=True)
    df_eval.reset_index(drop=True, inplace=True)
    pred = df_eval[df_eval['real'] == 1].index[-2] + 1
    evaluation = (N-1) / (pred)
    return evaluation


def printing_Kfold_scores(x_train_data, y_train_data):
    # skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=False)
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    recalls = []
    precisions = []
    f1s = []
    evaluations = []
    high_score = 0
    print('fold | Recall | Precision |   F1   | Contest')
    for iteration, indices in enumerate(kf.split(x_train_data, y_train_data), start=1):
        model = LogisticRegression()
        model.fit(x_train_data.iloc[indices[0], :].values,
                  y_train_data.iloc[indices[0], :].values.ravel())
        y_pred = model.predict(x_train_data.iloc[indices[1], :].values)
        y_proba = model.predict_proba(x_train_data.iloc[indices[1], :].values)
        recall = recall_score(y_train_data.iloc[indices[1], :].values, y_pred)
        precision = precision_score(
            y_train_data.iloc[indices[1], :].values, y_pred)
        evaluation = contest_eval(y_proba, y_train_data.iloc[indices[1], :].values.ravel())
        f1 = f1_score(y_train_data.iloc[indices[1], :].values, y_pred)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
        evaluations.append(evaluation)
        print('%d      %.3f     %.3f      %.3f     %.3f' %
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


def split_original():
    X = df.loc[:, df.columns != 'sar_flag']
    Y = df.loc[:, df.columns == 'sar_flag']

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=None, shuffle=True)
    return X_train, X_test, Y_train, Y_test


def split_undersampling():
    df_split = pd.concat([X_train, Y_train], axis=1)
    num_abn = len(df_split.loc[df_split['sar_flag'] == 1])
    num_nor = len(df_split.loc[df_split['sar_flag'] == 0])
    idx_abn = df_split.loc[df_split['sar_flag'] == 1].index.values
    idx_nor = df_split.loc[df_split['sar_flag'] == 0].index.values
    rand_idx_nor = np.random.choice(idx_nor, num_abn, replace=False)

    undersample_idx = np.concatenate([idx_abn, rand_idx_nor])
    undersample_df = df_split.loc[undersample_idx]

    X_undersample = undersample_df.loc[:, undersample_df.columns != 'sar_flag']
    Y_undersample = undersample_df.loc[:, undersample_df.columns == 'sar_flag']

    X_train_undersample, X_test_undersample, Y_train_undersample, Y_test_undersample = train_test_split(
        X_undersample, Y_undersample, test_size=0.2, random_state=None, shuffle=True)
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

# train
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

# test
trained_model = joblib.load(model_name)
Y_pred = trained_model.predict(X_test_undersample.values)
Y_proba = trained_model.predict_proba(X_test_undersample.values)
test_eval = contest_eval(Y_proba, Y_test_undersample.sar_flag.values)
print('Testing Score : %f ' % test_eval)

# confusion matrix
cnf_matrix = confusion_matrix(Y_test_undersample, Y_pred)
np.set_printoptions(precision=2)
df_cnf = pd.DataFrame(data=cnf_matrix, index=[0, 1], columns=[0, 1])
# pp_matrix(df_cnf, cmap='YlGn')
# plt.savefig('./tmp.jpg')

# contest
X_contest = df_test
Y_contest_pred = trained_model.predict_proba(X_contest.values)
submi_eval = contest_eval(Y_contest_pred, df_list.answer.sar_flag.values)
print('Public Score : %f ' % submi_eval)

breakpoint()
