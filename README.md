
# Heart Disease Prediction using Decision Tree and Random Forest

##  Project Overview
This project applies **Decision Tree** and **Random Forest** machine learning algorithms to predict whether a patient is likely to have heart disease.  
The dataset contains patient health metrics such as age, cholesterol, resting blood pressure, maximum heart rate, etc.  
We visualize the decision tree and analyze feature importance for the random forest model.

---

##  Dataset
The dataset `heart.csv` should be placed in the **project root folder**.  
It contains features such as age, cholesterol, resting blood pressure, maximum heart rate, etc., and the target variable:

- `0` → No Heart Disease  
- `1` → Heart Disease  

Example dataset source: [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

---

## 🛠 Requirements

Make sure you have the following installed:

- Python 3.8+
- pip (Python package manager)
- Graphviz (for Decision Tree visualization)
- Required Python libraries:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn graphviz
````



## Graphviz Installation

**For Windows:**

1. Download Graphviz installer from: [https://graphviz.org/download/](https://graphviz.org/download/)
2. Install it.
3. Add Graphviz `bin` folder path to your **System Environment Variables → Path**
   Example: `C:\Program Files\Graphviz\bin`
4. Restart your terminal or IDE.

---

## Steps to Run the Project

### 1️Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import graphviz
```

### 2️⃣ Load Dataset

```python
df = pd.read_csv("heart.csv")
```

### 3️⃣ Prepare Data

```python
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 4️⃣ Decision Tree Model

```python
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
```

### 5️⃣ Visualize Decision Tree

```python
dot_data = export_graphviz(
    dt_model, out_file=None,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True, rounded=True, special_characters=True
)
graph = graphviz.Source(dot_data)
graph.render("decision_tree")
```

### 6️⃣ Limited Depth Decision Tree

```python
dt_model_limited = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model_limited.fit(X_train, y_train)
y_pred_limited = dt_model_limited.predict(X_test)
print("Limited Depth Decision Tree Accuracy:", accuracy_score(y_test, y_pred_limited))
```

### 7️⃣ Random Forest Model

```python
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
```

### 8️⃣ Feature Importance

```python
importances = rf_model.feature_importances_
feat_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print(feat_importance_df)
```

### 9️⃣ Plot Feature Importance

```python
plt.figure(figsize=(8,5))
sns.barplot(x='Importance', y='Feature', data=feat_importance_df)
plt.title("Feature Importance (Random Forest)")
plt.show()
```

### 🔟 Cross-Validation

```python
scores = cross_val_score(rf_model, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Average CV Score:", scores.mean())
```

---

## 📊 Example Output

**Decision Tree Accuracy:**

```
0.85
```

**Random Forest Accuracy:**

```
0.90
```

**Feature Importance Plot:**
A bar chart showing which health features are most influential in prediction.

---



---


```
