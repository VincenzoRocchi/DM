import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
# import minmaxscaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


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

