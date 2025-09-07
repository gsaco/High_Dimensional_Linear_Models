import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os


os.chdir(r"C:\Users\VICTOR\Documents\GitHub\High_Dimensional_Linear_Models")

df = pd.read_csv(r"Python\Input\apartments.csv")
df.head()

#a1
df["area2"] = df["area"]**2

#a2
h_dummies = ["hasparkingspace", "hasbalcony", "haselevator", "hassecurity", "hasstorageroom"]
for col in h_dummies:
    df[col] = df[col].map({"yes": 1, "no": 0})

#a3
df["area_numero"] = df["area"].astype(int) % 10
for i in range(10):
    df[f"end_{i}"] = (df["area_numero"] == i).astype(int)

#b1
categorical_cols = ['month', 'type', 'rooms', 'ownership', 'buildingmaterial']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
end_dummies = [f'end_{i}' for i in range(9)]
x_variables = ['area', 'area2', 'distance_to_school', 'distance_to_clinic', 'distance_to_postoffice']+h_dummies+end_dummies





