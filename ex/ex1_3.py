import itertools
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import authmecv as acv
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from pretty_confusion_matrix import pp_matrix
from xgboost import XGBClassifier

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
model_name = './ex1_3_2'
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
    x_train, x_calib, y_train, y_calib = train_test_split(x_train_data, y_train_data, test_size=0.2, random_state=42)
    # params = {'max_depth': [2,4,6,8,10,12],
    #      'min_samples_split': [2,3,4],
    #      'min_samples_leaf': [1,2]}
    # dt = tree.DecisionTreeClassifier()
    # model = GridSearchCV(estimator=dt,param_grid=params)

    model = tree.DecisionTreeClassifier(max_depth=2, min_samples_leaf=2, random_state=0) # max_depth=2, min_samples_leaf=2, random_state=0
    # model = RandomForestClassifier(n_estimators=150, max_samples=0.8, random_state=0)
    model.fit(x_train.values, y_train.values.ravel())
    calib_model = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
    calib_model.fit(x_calib.values, y_calib.values.ravel())
    y_pred = calib_model.predict(x_calib.values)
    y_proba = calib_model.predict_proba(x_calib.values)
    recall = recall_score(y_calib.values, y_pred)
    precision = precision_score(y_calib.values, y_pred)
    f1 = f1_score(y_calib.values, y_pred)
    evaluation = contest_eval(y_proba, y_calib.values.ravel())
    print('--- Training ---')
    print('recall %.3f | precision %.3f | f1 %.3f | contest %.3f' %
          (recall, precision, f1, evaluation))
    joblib.dump(calib_model, model_name)


def split_original():
    X = df.loc[:, df.columns != 'sar_flag']
    Y = df.loc[:, df.columns == 'sar_flag']

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, shuffle=True)
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
        X_undersample, Y_undersample, test_size=0.2, random_state=42, shuffle=True)
    return X_train_undersample, X_test_undersample, Y_train_undersample, Y_test_undersample


def tree_importance_tabel(x_test, trained_model):
    dict_imrt = {'name': x_test.columns, 'importance': trained_model.feature_importances_}
    df_imrt = pd.DataFrame(dict_imrt)
    df_imrt.sort_values('importance', ascending=False, inplace=True)
    return df_imrt

# train
# original
X_train, X_test, Y_train, Y_test = split_original()
# origin = printing_Kfold_scores(X_train, Y_train)

# undersample
X_train_undersample, X_test_undersample, Y_train_undersample, Y_test_undersample = split_undersampling()
undersample = printing_Kfold_scores(X_train_undersample, Y_train_undersample)

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

# impt_table = tree_importance_tabel(X_test, trained_model)

# If you want to plot tree graph, change print_Kfold_scores return form dump(calib.model) to dump(model)
# tree.plot_tree(trained_model, filled=True, feature_names=X_test.columns.values.tolist(), fontsize=5)
breakpoint()

