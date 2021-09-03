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
from gtda.time_series import  TakensEmbedding
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from poincare_plot import extract_poincare_feat
from recurrence import recurrence_plot
import vectapen as vea
import vecsampen as ves
import nolds
from XEntropy import fuzzy_entropy
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
    # window_length = window_length*fs

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


def get_other(dataframe, w=128):
    poincare_temp_data = []
    recurrence_temp_data = []
    apen_temp_data = []
    samp_temp_data = []
    fuzzy_temp_data = []
    lyap_temp_data = []
    dataframe = dataframe.drop(columns=['time', 'Action', 'name'])
    for i in range(0, len(dataframe), w):
        data = dataframe.iloc[i:i + w]
        if data.shape[0] == w:
            for j in range(0, data.shape[1]):
                temp = data.iloc[:, [j]]
                temp = temp.to_numpy()
                both_input = temp.reshape(temp.shape[0] * temp.shape[1])
                poincare_result = extract_poincare_feat(both_input)
                recurrence_result = recurrence_plot(both_input)
                apen_result = vea.apen(both_input, 2, 0.2 * np.std(both_input))
                samp_result = ves.sampen(both_input, 2, 0.2 * np.std(both_input))
                fuzzy_result = fuzzy_entropy(both_input, 2, 0.2 * np.std(both_input))
                lyp_result = nolds.lyap_r(both_input, emb_dim=10, lag=None, min_tsep=None, tau=1, min_neighbors=20,
                                          trajectory_len=20, fit='RANSAC', debug_plot=False, debug_data=False,
                                          plot_file=None, fit_offset=0)

                poincare_temp_data.append(poincare_result)
                recurrence_temp_data.append(recurrence_result)
                apen_temp_data.append(apen_result)
                samp_temp_data.append(samp_result)
                fuzzy_temp_data.append(fuzzy_result)
                lyap_temp_data.append(lyp_result)

    poincare_data = np.array(poincare_temp_data)
    recurrence_data = np.array(recurrence_temp_data)
    apen_data = np.array(apen_temp_data)
    samp_data = np.array(samp_temp_data)
    fuzzy_data = np.array(fuzzy_temp_data)
    lyp_data = np.array(lyap_temp_data)

    row = int(len(dataframe) / w)
    poincare_data = poincare_data.reshape(row, 9)
    recurrence_data = recurrence_data.reshape(row, 9)
    apen_data = apen_data.reshape(row, 9)
    samp_data = samp_data.reshape(row, 9)
    fuzzy_data = fuzzy_data.reshape(row, 9)
    lyp_data = lyp_data.reshape(row, 9)

    return poincare_data, recurrence_data, apen_data, samp_data, fuzzy_data, lyp_data


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
        # score = model.score(X_test, Y_test.astype("int"))
        Y_pre = build_model.predict_proba(X_test)[:, 1]
        # Y_pre = build_model.predict(X_test)
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
whole_gtd_x = []
whole_gtd_y = []
whole_apen_x = []
whole_apen_y = []
whole_sampen_x = []
whole_sampen_y = []
whole_fuzzy_x = []
whole_fuzzy_y = []
whole_pc_x = []
whole_pc_y = []
whole_rc_x = []
whole_rc_y = []
whole_lyap_x = []
whole_lyap_y = []

for sbj in subject_list:
    DF0, DF1, DF2 = sbj_df(df, sbj=sbj)
    NOR_DF = pd.concat([DF0, DF2])
    nor_bc, nor_pl, nor_sl = gtda(NOR_DF)
    fog_bc, fog_pl, fog_sl = gtda(DF1)

    nor_pc, nor_rc, nor_apen, nor_samp, nor_fuzzy, nor_lyp = get_other(NOR_DF)
    nor_rc = clean_nan(nor_rc)
    nor_pc = clean_nan(nor_pc)
    nor_apen = clean_nan(nor_apen)
    nor_samp = clean_nan(nor_samp)
    nor_fuzzy = clean_nan(nor_fuzzy)
    nor_lyp = clean_nan(nor_lyp)

    fog_pc, fog_rc, fog_apen, fog_samp, fog_fuzzy, fog_lyp = get_other(DF1)
    fog_rc = clean_nan(fog_rc)
    fog_pc = clean_nan(fog_pc)
    fog_apen = clean_nan(fog_apen)
    fog_samp = clean_nan(fog_samp)
    fog_fuzzy = clean_nan(fog_fuzzy)
    fog_lyp = clean_nan(fog_lyp)

    print(len(fog_pc))
    gtd_fog = np.concatenate((fog_bc, fog_pl, fog_sl), axis=1)
    gtd_normal = np.concatenate((nor_bc, nor_pl, nor_sl), axis=1)
    fog = [fog_pc, fog_rc, fog_apen, fog_samp, fog_fuzzy, fog_lyp, gtd_fog]
    normal = [nor_pc, nor_rc, nor_apen, nor_samp, nor_fuzzy, nor_lyp, gtd_normal]

    idx = get_random_list(0, len(fog_pc) - 1, int(len(fog_pc) * 1))
    for length, nor_feature in enumerate(normal):
        fog_feature = fog[length]
        nor_feature = nor_feature[idx]

        X = np.concatenate([nor_feature, fog_feature], axis=0)
        Y = []
        Y += len(nor_feature) * [0]
        Y += len(fog_feature) * [1]
        if length == 0:
            whole_pc_x.append(X)
            whole_pc_y += Y
        elif length == 1:
            whole_rc_x.append(X)
            whole_rc_y += Y
        elif length == 2:
            whole_apen_x.append(X)
            whole_apen_y += Y
        elif length == 3:
            whole_sampen_x.append(X)
            whole_sampen_y += Y
        elif length == 4:
            whole_fuzzy_x.append(X)
            whole_fuzzy_y += Y
        elif length == 5:
            whole_lyap_x.append(X)
            whole_lyap_y += Y
        elif length == 6:
            whole_gtd_x.append(X)
            whole_gtd_y += Y
        print(sbj, '第{}个特征'.format(length + 1))
        training(X, Y, sbj)

whole_pc_x = np.concatenate(whole_pc_x, axis=0)
whole_rc_x = np.concatenate(whole_rc_x, axis=0)
whole_apen_x = np.concatenate(whole_apen_x, axis=0)
whole_sampen_x = np.concatenate(whole_sampen_x, axis=0)
whole_fuzzy_x = np.concatenate(whole_fuzzy_x, axis=0)
whole_lyap_x = np.concatenate(whole_lyap_x, axis=0)
whole_gtd_x = np.concatenate(whole_gtd_x, axis=0)

whole = 'All Person'
plt.figure()
print('all person PC')
training_plot(whole_pc_x, whole_pc_y, 'Poincare Plot')
print('all person RC')
training_plot(whole_rc_x, whole_rc_y, 'Recurrence Plot')
print('all person Apen')
training_plot(whole_apen_x, whole_apen_y, 'Approximate Entropy')
print('all person Sampen')
training_plot(whole_sampen_x, whole_sampen_y, 'Sample Entropy')
print('all person Fuzzy')
training_plot(whole_fuzzy_x, whole_fuzzy_y, 'Fuzzy Entropy')
print('all person Lyap')
training_plot(whole_apen_x, whole_apen_y, 'Lyapunov Exponent')
print('all person gtd')
training_plot(whole_gtd_x, whole_gtd_y, 'TGNDA')
plt.title(whole + ' FOG ROC')
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()