import itertools
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from pretty_confusion_matrix import pp_matrix
from xgboost import XGBClassifier


trainset_path = './Class_Dataset/trainset/trainset_ex1_minmax.csv'
testset_path = './Class_Dataset/testset/testset_ex1_minmax.csv'
submi_path = '/Users/yuj/repo/SAR/submi/submission_format.csv'
df = pd.read_csv(trainset_path)  # 22744
df_test_org = pd.read_csv(testset_path)
df_test = df_test_org.drop(['alert_key'], axis=1, inplace=False)
submi = pd.read_csv(submi_path)
model_name = './ex1_3_tree'


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
    evaluation = contest_eval(y_pred, y_proba, y_calib.values.ravel())
    print('--- Training ---')
    print('recall %.3f | precision %.3f | f1 %.3f | contest %.3f' %
          (recall, precision, f1, evaluation))
    joblib.dump(calib_model, model_name)


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
        X, Y, test_size=0.2, random_state=42, shuffle=True)
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
        X_undersample, Y_undersample, test_size=0.00001, random_state=42, shuffle=True)
    return X_train_undersample, X_test_undersample, Y_train_undersample, Y_test_undersample


def print_result(real_label, pred_label, evaluation):
    print("--- Testing ---")
    print("[ pos_label=0 ]")
    print("Recall    : %.3f" % recall_score(
        real_label, pred_label, pos_label=0))
    print("Precision : %.3f" % precision_score(
        real_label, pred_label, pos_label=0))
    print('f1        : %.3f' % f1_score(real_label, pred_label, pos_label=0))
    print("[ pos_label=1 ]")
    print("Recall    : %.3f" % recall_score(
        real_label, pred_label, pos_label=1))
    print("Precision : %.3f" % precision_score(
        real_label, pred_label, pos_label=1))
    print("f1        : %.3f" % f1_score(real_label, pred_label, pos_label=1))
    print("[ Evaluation ]")
    print("eval      : %.3f" % evaluation)
    print('==='*15)

def tree_importance_tabel(x_test, trained_model):
    dict_imrt = {'name': x_test.columns, 'importance': trained_model.feature_importances_}
    df_imrt = pd.DataFrame(dict_imrt)
    df_imrt.sort_values('importance', ascending=False, inplace=True)
    return df_imrt

# original
X_train, X_test, Y_train, Y_test = split_original()
# origin = printing_Kfold_scores(X_train, Y_train)

# undersample
X_train_undersample, X_test_undersample, Y_train_undersample, Y_test_undersample = split_undersampling()
undersample = printing_Kfold_scores(X_train_undersample, Y_train_undersample)

# testset prediction
trained_model = joblib.load(model_name)
Y_pred = trained_model.predict(X_test.values)
Y_proba = trained_model.predict_proba(X_test.values)

# Confusion matrix
cnf_matrix = confusion_matrix(Y_test, Y_pred)
np.set_printoptions(precision=2)
df_cnf = pd.DataFrame(data=cnf_matrix, index=[0, 1], columns=[0, 1])
# pp_matrix(df_cnf, cmap='YlOrBr')

# contest eval
evaluation = contest_eval(Y_pred, Y_proba, Y_test.to_numpy().reshape(-1))
print_result(Y_test, Y_pred, evaluation)

# contest prediction
X_contest = df_test
Y_contest_pred = trained_model.predict_proba(X_contest.values)[:, 1]

for i, prob in enumerate(Y_contest_pred):
    key = df_test_org.loc[i, 'alert_key']
    submi.loc[submi['alert_key'] == key, 'probability'] = prob

impt_table = tree_importance_tabel(X_test, trained_model)

# If you want to plot tree graph, change print_Kfold_scores return form dump(calib.model) to dump(model)
# tree.plot_tree(trained_model, filled=True, feature_names=X_test.columns.values.tolist(), fontsize=5)
breakpoint()

