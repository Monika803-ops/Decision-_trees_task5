
# Heart Disease Prediction using Decision Tree and Random Forest

##  Project Overview
This project uses **Decision Tree** and **Random Forest** classifiers to predict whether a patient is likely to have heart disease based on given features.  
We also visualize the decision tree and show feature importance for the random forest model.

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

---
Here’s the **step-by-step process** to perform this task from start to end, exactly based on your provided code:

---

### **Step 1: Install Required Libraries**

Make sure the following Python libraries are installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn graphviz
```

Also install **Graphviz software** on your system (not just the Python package) because it's needed to render the decision tree.

**For Windows:**

1. Download Graphviz installer from: [https://graphviz.org/download/](https://graphviz.org/download/)
2. Install it.
3. Add Graphviz `bin` folder path to your **System Environment Variables → Path** (e.g., `C:\Program Files\Graphviz\bin`).
4. Restart your terminal or IDE.

---

### **Step 2: Import Libraries**

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

---

### **Step 3: Load Dataset**

```python
df = pd.read_csv("heart.csv")
```

---

### **Step 4: Separate Features and Target**

```python
X = df.drop("target", axis=1)
y = df["target"]
```

---

### **Step 5: Split Data into Train & Test Sets**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

### **Step 6: Train Decision Tree Model**

```python
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
```

---

### **Step 7: Visualize Decision Tree**

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

This saves a `decision_tree.pdf` or `.png` (depending on your Graphviz setup).

---

### **Step 8: Train Limited Depth Decision Tree**

```python
dt_model_limited = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model_limited.fit(X_train, y_train)
y_pred_limited = dt_model_limited.predict(X_test)
print("Limited Depth Decision Tree Accuracy:", accuracy_score(y_test, y_pred_limited))
```

---

### **Step 9: Train Random Forest Model**

```python
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
```

---

### **Step 10: Feature Importance (Random Forest)**

```python
importances = rf_model.feature_importances_
feat_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print(feat_importance_df)
```

---

### **Step 11: Plot Feature Importance**

```python
plt.figure(figsize=(8,5))
sns.barplot(x='Importance', y='Feature', data=feat_importance_df)
plt.title("Feature Importance (Random Forest)")
plt.show()
```

---

### **Step 12: Cross-Validation**

```python
scores = cross_val_score(rf_model, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Average CV Score:", scores.mean())
```

---

If you want, I can also give you a **single ready-to-run .py file** that includes these steps in order. That way you just run it once and get all results.



