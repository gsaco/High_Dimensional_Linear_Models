import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.chdir(r"C:\Users\VICTOR\Documents\GitHub\High_Dimensional_Linear_Models")

df = pd.read_csv(r"Python\Input\apartments.csv")
df.head()

#a1
df["area2"] = df["area"]**2

#a2
cols_dummy = ["hasparkingspace", "hasbalcony", "haselevator", "hassecurity", "hasstorageroom"]
for col in cols_dummy:
    df[col] = df[col].map({"yes": 1, "no": 0})

#a3
df["area_numero"] = df["area"].astype(int) % 10
for i in range(10):
    df[f"end_{i}"] = (df["area_numero"] == i).astype(int)

#b1


