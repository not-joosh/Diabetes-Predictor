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












