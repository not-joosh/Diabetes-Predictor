# Diabetes-Predictor
Diabetes Prediction Model | Machine Learning, ipynb, pandas, numpy 

---

##### Importing Dependencies
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
```

---

##### Data Collection and Analysis: PIMA Diabetes Dataset
```python
# Loading the diabetes dataset into a pandas DataFrame
diabetes_dataset = pd.read_csv('/content/diabetes.csv')
```

---

##### Printing the first 5 rows of the dataset
```python
diabetes_dataset.head()
```
**OUTPUT:**
  |   | Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI  | DiabetesPedigreeFunction | Age | Outcome |
  |---|-------------|---------|---------------|---------------|---------|------|-------------------------|-----|---------|
  | 0 | 6           | 148     | 72            | 35            | 0       | 33.6 | 0.627                   | 50  | 1       |
  | 1 | 1           | 85      | 66            | 29            | 0       | 26.6 | 0.351                   | 31  | 0       |
  | 2 | 8           | 183     | 64            | 0             | 0       | 23.3 | 0.672                   | 32  | 1       |
  | 3 | 1           | 89      | 66            | 23            | 94      | 28.1 | 0.167                   | 21  | 0       |
  | 4 | 0           | 137     | 40            | 35            | 168     | 43.1 | 2.288                   | 33  | 1       |

---

```python
# number of rows and columns in this dataset
diabetes_dataset.shape
```

---

```python
# getting the statistical measures of the data
diabetes_dataset.describe()
```
**OUTPUT**:
  |   | Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI  | DiabetesPedigreeFunction | Age | Outcome |
  |---|-------------|---------|---------------|---------------|---------|------|-------------------------|-----|---------|
  | 0 | 6           | 148     | 72            | 35            | 0       | 33.6 | 0.627                   | 50  | 1       |
  | 1 | 1           | 85      | 66            | 29            | 0       | 26.6 | 0.351                   | 31  | 0       |
  | 2 | 8           | 183     | 64            | 0             | 0       | 23.3 | 0.672                   | 32  | 1       |
  | 3 | 1           | 89      | 66            | 23            | 94      | 28.1 | 0.167                   | 21  | 0       |
  | 4 | 0           | 137     | 40            | 35            | 168     | 43.1 | 2.288                   | 33  | 1       |
---

**OUTPUT:**
(768, 9)

---

```python
# getting the statistical measures of the data
diabetes_dataset.describe()
```

**OUTPUT:**
|   | Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI  | DiabetesPedigreeFunction | Age | Outcome |
|---|-------------|---------|---------------|---------------|---------|------|-------------------------|-----|---------|
| 0 | 6           | 148     | 72            | 35            | 0       | 33.6 | 0.627                   | 50  | 1       |
| 1 | 1           | 85      | 66            | 29            | 0       | 26.6 | 0.351                   | 31  | 0       |
| 2 | 8           | 183     | 64            | 0             | 0       | 23.3 | 0.672                   | 32  | 1       |
| 3 | 1           | 89      | 66            | 23            | 94      | 28.1 | 0.167                   | 21  | 0       |
| 4 | 0           | 137     | 40            | 35            | 168     | 43.1 | 2.288                   | 33  | 1       |

---
```python
# Taking Outcome values, checking how many 0's and 1's
diabetes_dataset['Outcome'].value_counts()
```
**OUTPUT:**
0    500
1    268
Name: Outcome, dtype: int64

---
##### 0 ---> Non Diabetic
##### 1 ---> Diabetic
---
```python
# Getting the mean for these both of these values...
diabetes_dataset.groupby('Outcome').mean()
```
**OUTPUT:**
| Outcome | Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI     | DiabetesPedigreeFunction | Age      |
|---------|-------------|---------|---------------|---------------|---------|---------|-------------------------|----------|
| 0       | 3.298       | 109.980 | 68.184        | 19.664        | 68.792  | 30.3042 | 0.429734                | 31.190   |
| 1       | 4.865672    | 141.257 | 70.824627     | 22.164179     | 100.336 | 35.1425 | 0.5505                  | 37.067164|

---
```python
# Seperating Data on labels... Spitting. Axis = 1 if wee are dropping colmn, axis = 0 if we drop a row
X = diabetes_dataset.drop(columns = 'Outcome', axis = 1)
Y = diabetes_dataset['Outcome']
print(X)
```
**OUTPUT:**
|   | Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI  | DiabetesPedigreeFunction | Age |
|---|-------------|---------|---------------|---------------|---------|------|-------------------------|-----|
| 0 | 6           | 148     | 72            | 35            | 0       | 33.6 | 0.627                   | 50  |
| 1 | 1           | 85      | 66            | 29            | 0       | 26.6 | 0.351                   | 31  |
| 2 | 8           | 183     | 64            | 0             | 0       | 23.3 | 0.672                   | 32  |
| 3 | 1           | 89      | 66            | 23            | 94      | 28.1 | 0.167                   | 21  |
| 4 | 0           | 137     | 40            | 35            | 168     | 43.1 | 2.288                   | 33  |
|...| ...         | ...     | ...           | ...           | ...     | ...  | ...                     | ... |
|763| 10          | 101     | 76            | 48            | 180     | 32.9 | 0.171                   | 63  |
|764| 2           | 122     | 70            | 27            | 0       | 36.8 | 0.34                    | 27  |
|765| 5           | 121     | 72            | 23            | 112     | 26.2 | 0.245                   | 30  |
|766| 1           | 126     | 60            | 0             | 0       | 30.1 | 0.349                   | 47  |
|767| 1           | 93      | 70            | 31            | 0       | 30.4 | 0.315                   | 23  |

[768 rows x 8 columns]

---
```python
print(Y)
```
**OUTPUT:**
```c
  0      1
  1      0
  2      1
  3      0
  4      1
        ..
  763    0
  764    0
  765    0
  766    1
  767    0
  Name: Outcome, Length: 768, dtype: int64
```
---
```python
# Data Standardization | This is to establish a specific range for the ml model to make better predictions
scaler = StandardScaler()
scaler.fit(X) # Passing in our data to transform
standardizedData = scaler.transform(X)
# scaler.fit_transform(X) <-- Single Step.
print(standardizedData)
```
**OUTPUT:**
```cpp
  [[ 0.63994726  0.84832379  0.14964075 ...  0.20401277  0.46849198
     1.4259954 ]
   [-0.84488505 -1.12339636 -0.16054575 ... -0.68442195 -0.36506078
    -0.19067191]
   [ 1.23388019  1.94372388 -0.26394125 ... -1.10325546  0.60439732
    -0.10558415]
   ...
   [ 0.3429808   0.00330087  0.14964075 ... -0.73518964 -0.68519336
    -0.27575966]
   [-0.84488505  0.1597866  -0.47073225 ... -0.24020459 -0.37110101
     1.17073215]
   [-0.84488505 -0.8730192   0.04624525 ... -0.20212881 -0.47378505
    -0.87137393]]
```
##### Now, our model can make better predictions because the values are in similar range. It is between 0 and 1 now.

---
```python
X = standardizedData
print(X)
print(Y)
```
**OUTPUT:**
```cpp
  [[ 0.63994726  0.84832379  0.14964075 ...  0.20401277  0.46849198
     1.4259954 ]
   [-0.84488505 -1.12339636 -0.16054575 ... -0.68442195 -0.36506078
    -0.19067191]
   [ 1.23388019  1.94372388 -0.26394125 ... -1.10325546  0.60439732
    -0.10558415]
   ...
   [ 0.3429808   0.00330087  0.14964075 ... -0.73518964 -0.68519336
    -0.27575966]
   [-0.84488505  0.1597866  -0.47073225 ... -0.24020459 -0.37110101
     1.17073215]
   [-0.84488505 -0.8730192   0.04624525 ... -0.20212881 -0.47378505
    -0.87137393]]
  0      1
  1      0
  2      1
  3      0
  4      1
        ..
  763    0
  764    0
  765    0
  766    1
  767    0
  Name: Outcome, Length: 768, dtype: int64
```









