# %load test.py
import math
import os
import random
import time

from numpy import interp
import pandas as pd
import numpy as np
from gtda.diagrams import Scaler,  BettiCurve, PersistenceLandscape, Silhouette
from gtda.homology import VietorisRipsPersistence
from gtda.pipeline import make_pipeline
from gtda.time_series import TakensEmbedding
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import warnings


def get_random_list(start, stop, n):
    arr = list(range(start, stop + 1))
    shuffle_n(arr, n)
    return arr[-n:]


def shuffle_n(arr, n):
    random.seed(time.time())
    for i in range(len(arr) - 1, len(arr) - n - 1, -1):
        j = random.randint(0, i)
        arr[i], arr[j] = arr[j], arr[i]


def create_window(act, window_length, dataframe):
    indices = list(dataframe[dataframe.Action == act].index)
    groups = []
    temp = []
    group_count = 0
    for i in range(len(indices)):
        if i == len(indices) - 1:
            temp.append(indices[i])
            groups.append(temp)
            temp = []
            break
        temp.append(indices[i])
        if indices[i] + 1 != indices[i + 1]:
            group_count += 1
            groups.append(temp)
            temp = []

    fs = 64

    final_dataframe = pd.DataFrame()
    for i in groups:
        required = math.floor(len(i) / (window_length * fs))

        req_index = i[0:(required * window_length * fs)]

        final_dataframe = pd.concat([final_dataframe, dataframe.iloc[req_index, :]], axis=0)
    return final_dataframe


def sbj_df(df, sbj='S01'):
    DF0 = create_window(0, 2, df)
    DF0 = DF0[DF0['name'] == sbj]
    DF1 = create_window(1, 2, df)
    DF1 = DF1[DF1['name'] == sbj]
    DF2 = create_window(2, 2, df)
    DF2 = DF2[DF2['name'] == sbj]
    return DF0, DF1, DF2


def gtda(dataframe, w=128):
    all_data = []
    dataframe = dataframe.drop(columns=['time', 'Action', 'name'])
    for i in range(0, len(dataframe), w):
        data = dataframe.iloc[i:i + w]
        data = data.to_numpy().transpose()
        if data.shape[1] == w:
            all_data.append(data)
    all_data = np.array(all_data)
    steps = [TakensEmbedding(time_delay=5, dimension=3),
             VietorisRipsPersistence(),
             Scaler()
             ]
    tda_pipe = make_pipeline(*steps)
    diagrams = tda_pipe.fit_transform(all_data)
    BC = BettiCurve(n_bins=50).fit_transform(diagrams)
    PL = PersistenceLandscape(n_bins=50).fit_transform(diagrams)
    SL = Silhouette(n_bins=50).fit_transform(diagrams)

    return np.mean(BC, axis=1), np.sum(PL, axis=1), np.mean(SL, axis=1)


def training(x, y, sbj_name):
    y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
    model = RandomForestClassifier(random_state=1)
    model.fit(x_train, y_train)
    y_pre = model.predict(x_test)
    print(classification_report(y_test, y_pre, digits=4))
    pre_y = model.predict_proba(x_test)[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, pre_y)
    roc_auc = metrics.auc(fpr, tpr)
    print('AUC:', roc_auc)
    print('\n')

    cv = KFold(n_splits=5, shuffle=True, random_state=None)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    for train_index, test_index in cv.split(x):
        X_train, X_test = x[train_index], x[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        build_model = model
        build_model.fit(X_train, Y_train.astype("int"))
        Y_pre = build_model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(Y_test, Y_pre)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=.6, label='ROC fold %d(AUC=%0.2f)' % (i, roc_auc))
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC=%0.2f)' % mean_auc, lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    print('mean auc', mean_auc)
    plt.fill_between(mean_tpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(sbj_name + 'FOG_ROC')
    plt.legend(loc='lower right')
    plt.show()
    return


def training_plot(x, y, m):
    y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
    model = RandomForestClassifier(random_state=1)
    model.fit(x_train, y_train)
    y_pre = model.predict(x_test)
    print(classification_report(y_test, y_pre, digits=4))
    pre_y = model.predict_proba(x_test)[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, pre_y)
    roc_auc = metrics.auc(fpr, tpr)
    print('AUC:', roc_auc)
    print('\n')
    cv = KFold(n_splits=5, shuffle=True, random_state=None)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    for train_index, test_index in cv.split(x):
        X_train, X_test = x[train_index], x[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        build_model = model
        build_model.fit(X_train, Y_train.astype("int"))
        Y_pre = build_model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(Y_test, Y_pre)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        i += 1
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print('mean auc', mean_auc)
    plt.plot(fpr, tpr, lw=1, alpha=.6, label='FOG AUC of %s = %0.3f' % (m, mean_auc))


def clean_nan(x):
    x = np.nan_to_num(x, posinf=0, neginf=0)
    return x


warnings.filterwarnings("ignore")
df = pd.read_csv('TGNDA_for_FOG2.csv')
subject_list = ['S01', 'S02', 'S03', 'S05', 'S06', 'S07', 'S08', 'S09']
whole_x = []
whole_y = []
whole_bc_x = []
whole_bc_y = []
whole_pl_x = []
whole_pl_y = []
whole_sl_x = []
whole_sl_y = []

for sbj in subject_list:
    DF0, DF1, DF2 = sbj_df(df, sbj=sbj)
    NOR_DF = pd.concat([DF0, DF2])
    nor_bc, nor_pl, nor_sl = gtda(NOR_DF)
    fog_bc, fog_pl, fog_sl = gtda(DF1)
    print(len(fog_bc))
    fog = [fog_bc, fog_pl, fog_sl]
    normal = [nor_bc, nor_pl, nor_sl]
    idx = get_random_list(0, len(fog_bc) - 1, int(len(fog_bc) * 1))
    for length, nor_feature in enumerate(normal):
        fog_feature = fog[length]
        nor_feature = nor_feature[idx]

        X = np.concatenate([nor_feature, fog_feature], axis=0)
        Y = []
        Y += len(nor_feature) * [0]
        Y += len(fog_feature) * [1]
        if length == 0:
            whole_bc_x.append(X)
            whole_bc_y += Y
        elif length == 1:
            whole_pl_x.append(X)
            whole_pl_y += Y
        elif length == 2:
            whole_sl_x.append(X)
            whole_sl_y += Y
        print(sbj, '第{}个特征'.format(length + 1))
        training(X, Y, sbj)

    all_fog = np.concatenate(fog, axis=1)
    all_nor = np.concatenate(normal, axis=1)
    all_nor = all_nor[idx]
    all_X = np.concatenate([all_nor, all_fog], axis=0)
    all_Y = []
    all_Y += len(all_nor) * [0]
    all_Y += len(all_fog) * [1]
    print(sbj, '特征融合')
    training(all_X, all_Y, sbj)
    whole_x.append(all_X)
    whole_y += all_Y

whole_bc_x = np.concatenate(whole_bc_x, axis=0)
whole_pl_x = np.concatenate(whole_pl_x, axis=0)
whole_sl_x = np.concatenate(whole_sl_x, axis=0)
whole_x = np.concatenate(whole_x, axis=0)
whole = 'All Person'
plt.figure()
print('all person BC')
training_plot(whole_bc_x, whole_bc_y, 'BC')
print('all person PL')
training_plot(whole_pl_x, whole_pl_y, 'PL')
print('all person SL')
training_plot(whole_sl_x, whole_sl_y, 'SL')
print('all person whole')
training_plot(whole_x, whole_y, 'Fusion')
plt.title(whole + ' FOG ROC')
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
