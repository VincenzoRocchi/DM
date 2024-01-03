import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV, cross_val_score

CAT = [
    'modality',
    'vocal_channel',
    'emotion',
    'emotional_intensity',
    'statement',
    'repetition',
    'actor',
    'sex',
    'filename'
]


def decision_tree_class(x_train, y_train, x_val, y_val):

    param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 4, 6, 8, 10],
    'min_samples_split': [2, 4, 6, 8, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5]
}
    
    model = DecisionTreeClassifier(random_state = 42)
    grid_model = GridSearchCV(estimator = model,
                          param_grid = param_grid,
                          scoring = 'accuracy',
                          cv = 5, 
                          n_jobs = 4)


    grid_model.fit(x_train, y_train)

    print(grid_model.best_params_)
    print(grid_model.best_score_)

    y_pred = grid_model.predict(x_val)

    print('Accuracy %s' % accuracy_score(y_val, y_pred))
    print('F1-score %s' % f1_score(y_val, y_pred, average=None))
    print(classification_report(y_val, y_pred, digits=3))

    return accuracy_score(y_val, y_pred)


def knn_class(x_train, y_train, x_val, y_val):

    param_grid = {
    'n_neighbors': list(range(1, int(math.sqrt(len(y_train)) + 1))),
    'weights': ['uniform', 'distance'],
    'p': [1, 2],
    }

    
    model = KNeighborsClassifier()
    grid_model = GridSearchCV(estimator = model,
                          param_grid = param_grid,
                          scoring = 'accuracy',
                          cv = 5,
                          n_jobs = 4)


    grid_model.fit(x_train, y_train)

    print(grid_model.best_params_)
    print(grid_model.best_score_)

    y_pred = grid_model.predict(x_val)

    print('Accuracy %s' % accuracy_score(y_val, y_pred))
    print('F1-score %s' % f1_score(y_val, y_pred, average=None))
    print(classification_report(y_val, y_pred, digits=3))

    return accuracy_score(y_val, y_pred)

def copy_df_and_drop(df, col):
    df_copy = df.copy()
    df_copy = df_copy.drop(col, axis = 1)
    return df_copy


def print_lof(df, nn, size = 3):
    lof = LocalOutlierFactor(n_neighbors = nn)
    lof.fit_predict(df)
    lof_scores = lof.negative_outlier_factor_

    # create a pca and plot the data in 2d and then 3d
    pca = PCA(n_components = 3)
    df_pca = pd.DataFrame(pca.fit_transform(df), columns=['PC' + str(i) for i in range(1, 4)])
    # plot the data
    plt.scatter(df_pca['PC1'], df_pca['PC2'], s = lof_scores * - size, c = lof_scores)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar()
    plt.show()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(df_pca['PC1'], df_pca['PC2'], df_pca['PC3'], s = lof_scores * - size, c= lof_scores)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(np.sort(lof_scores))
    plt.xlabel('Point')
    plt.ylabel('LOF Score')
    plt.show()
    plt.close()

    return lof_scores

def get_lof_score(df, nn):
    lof = LocalOutlierFactor(n_neighbors = nn)
    lof.fit_predict(df)
    lof_scores = lof.negative_outlier_factor_
    return lof_scores

def get_max_lof(df, nn_min, nn_max):
    l = []
    out = []
    for i in range(nn_min, nn_max):
        l.append(get_lof_score(df, i))
    # append to out the max lof score for each point
    for i in range(len(l[0])):
        out.append(min([j[i] for j in l]))
    return out

def color_emotions(emo):
    colors = []
    for i in emo:
        if i == 'neutral':
            colors.append('red')
        elif i == 'calm':
            colors.append('blue')
        elif i == 'happy':
            colors.append('green')
        elif i == 'sad':
            colors.append('orange')
        elif i == 'angry':
            colors.append('purple')
        elif i == 'fearful':
            colors.append('black')
        elif i == 'disgust':
            colors.append('yellow')
        elif i == 'surprised':
            colors.append('gray')
    
    return colors

def find_corr(corr, threshold):
    col_corr = [] # Set of all the names of correlated columns
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            if abs(corr.iloc[i, j]) > threshold and j != i:
                col_corr.append(corr.columns[i])
    # find most common value in col_corr
    if len(col_corr) == 0:
        return False
    return col_corr


def find_corr_good(corr, threshold):
    out = [[] for i in range(len(corr.columns))] # Set of all the names of correlated columns
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            if abs(corr.iloc[i, j]) > threshold and j != i:
                out[i].append(corr.columns[j])
    # find most common value in col_corr
    return out


def highes_corr(cor_list, col_names):
    #find sublist with highest len
    max_len = 0
    max_index = 0
    for i in range(len(cor_list)):
        if len(cor_list[i]) > max_len:
            max_len = len(cor_list[i])
            max_index = i
    if max_len == 0:
        return False, False
    return col_names[max_index], max_index

def modify_list(col_name, cor_list, max_index):
    for i in range(len(cor_list)):
        if col_name in cor_list[i]:
            cor_list[i].remove(col_name)
    cor_list[max_index] = []
    return cor_list

def find_out(corr_list, col_names):
    out = []
    while True:
        col_name, max_index = highes_corr(corr_list, col_names)
        if col_name == False:
            break
        corr_list = modify_list(col_name, corr_list, max_index)
        out.append(col_name)
    return out

def print_corr_col(corr):
    corr_col = find_corr(corr, 0.9)
    counter = Counter(corr_col)
    most_common_value = counter.most_common(1)[0][0]
    print(most_common_value)
    print(counter[most_common_value])
    npcor = np.array(corr_col)
    # print unique values in npcor with counts
    count = np.unique(npcor, return_counts = True)
    for i in range(len(count[0])):
        print(count[0][i], count[1][i])


def find_unique(df):
    unique = []
    for i in df.columns:
        if len(df[i].unique()) == 1:
            unique.append(i)
    return unique

# drop columns that have a single value

