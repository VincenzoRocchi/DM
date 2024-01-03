#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:28:43 2022

@author: vincenzorocchi
"""

import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

#importing the dataset
df = pd.read_csv("/Users/vincenzorocchi/Desktop/DataMining_project_RAVDESS/Ravdess/ravdess_features.csv")

print(df.shape, #checking shape,values and info
      df,"\n")

df.info()

#%%

print(pd.unique(df["modality"])) #checking for unique values

df_reduced = df.drop(axis = 1, columns=("modality")) #eliminating the modality column - unnecesary

#Creating a df_reduced dataset to work with it when needed

#%%

print(df.dtypes) #checking datatypes for each attribute

(df.isnull().any()) #checking for missing values
print(df.isnull().sum()) #printing the totals of the missing values

#filling missing values by probalby substituting them correlation based (if found meaningful one)

#%%---------
#CORRELATION
#-----------

print(df.corr())

#%%

#checking the unique values for each object value to substitute them

print(pd.unique(df["emotion"])) #too many unique values for the emotion column, solving it later
print(pd.unique(df["vocal_channel"]))
print(pd.unique(df["emotional_intensity"]))
print(pd.unique(df["repetition"]))
print(pd.unique(df["statement"]))
print(pd.unique(df["sex"]))

#defining the maps to convert the categorical variables in numerical ones

vocal_map = {"speech": 0, "song": 1}
emotional_int_map = {"normal": 0, "strong" :1}
repetition_map = {"1st": 0, "2nd" :1}
sex_map = {"F": 0, "M" :1}
statement_map = {"Dogs are sitting by the door" :0, "Kids are talking by the door":1}

#%%
#using the dummies method to convert the emotion attribute

emotion_dummies = pd.get_dummies(df.emotion)
df_reduced = pd.concat([df, emotion_dummies], axis = "columns")

print(df_reduced)

#%%
#replacing the maps and creating the corr. graph

df_reduced.replace({"vocal_channel": vocal_map,
            "emotional_intensity": emotional_int_map,
            "repetition": repetition_map,
            "sex": sex_map,
            "statement": statement_map
           }).corr(method="pearson").style.background_gradient(cmap='coolwarm', 
            vmin=-1, vmax=1, axis=None).format(precision=2)

#%%
#dropping the uncorrelated 

df_reduced = (df.drop(columns=("frame_rate")) )
df_reduced = (df.drop(columns=("sample_width")) )
df_reduced = (df.drop(columns=("stft_max")) )

df_reduced.corr(method="pearson").style.background_gradient(cmap='coolwarm',
                                    vmin=-1, vmax=1, axis=None).format(precision=3)

#%%

df_reduced.loc[1:1000, ["channels", "frame_width"]]