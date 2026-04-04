import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

df = pd.read_csv('diabetes_dataset.csv')

cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]  # we have done data cleaning here, pregnancies can be zero its alr
df[cols] = df[cols].replace(0, np.nan)  #here we have replaced all values in mentioned features which cant have 0
df.fillna(df.mean(), inplace=True) #replaced features with mean

x1=df["Pregnancies"].to_numpy()
x2= df['Glucose'].to_numpy()
x3= df['BloodPressure'].to_numpy()
x4= df['SkinThickness'].to_numpy()
x5= df['Insulin'].to_numpy()
x6= df['BMI'].to_numpy()
x7= df['Age'].to_numpy()
x8= df['DiabetesPedigreeFunction'].to_numpy()

y=df["Outcome"].to_numpy()

X = np.column_stack((x1, x2, x3, x4, x5, x6, x7, x8))

mu = X.mean(axis=0)   #this is feature scaling for better use
sd = X.std(axis=0)
X = (X - mu) / sd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train(X, y, a, it):
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    cl = []
    for i in range(it):
        z = np.dot(X, w) + b
        p = sigmoid(z)
        dz = p - y
        dw = (1/m) * np.dot(X.T, dz)
        db = (1/m) * np.sum(dz)
        w -= a * dw
        b -= a * db
        j = - (1/m) * np.sum(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
        cl.append(j)
    return cl

als = [0.1, 0.01, 0.001, 0.0001]
its = 1000

for a in als:
    hist = train(X, y, a, its)
    plt.plot(range(its), hist, label=f'a={a}')

plt.xlabel('Iterations')
plt.ylabel('Cost (J)')
plt.title('Alpha Tuning for Pima Dataset')
plt.legend()
plt.savefig('alpha_tuning.png')

