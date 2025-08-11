
## **Task 5: Heart Disease Prediction Using Decision Trees & Ensemble Learning**

### **1. Objective**

The goal was to:

* Understand **Decision Trees**.
* Explore **Ensemble Learning** methods like **Random Forest**.
* Learn how to calculate **Feature Importance**.
* Evaluate models using **cross-validation**.
* Visualize results with graphs.

---

## **2. Dataset Setup**

* We used the **Heart Disease dataset** from Kaggle:
  **Kaggle Dataset Link:** [Heart Disease UCI Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
* **Placement:** The CSV file `heart.csv` was placed in the **root folder** of the project.
* **Dataset Info:**

  * Features: `age`, `chol`, `trestbps`, `thalach`, `cp`, `ca`, `thal`, `sex`, etc.
  * Target:

    * `0` → No Heart Disease
    * `1` → Heart Disease

---

## **3. Environment Setup**

To run the task successfully, we needed to install the following:

### **a. Python Libraries**

We installed all necessary libraries using pip:

```bash
pip install pandas numpy matplotlib scikit-learn graphviz
```

**Why each library was needed:**

* **pandas** → For reading and handling CSV data.
* **numpy** → For numerical calculations.
* **matplotlib** → For plotting feature importance and graphs.
* **scikit-learn** → For Decision Tree, Random Forest, and evaluation metrics.
* **graphviz** → For visualizing Decision Trees as diagrams.

---

## **4. Graphviz Installation**

Even if we install the Python `graphviz` package, the **Graphviz software** itself must be installed on the system:

### **Windows Installation Steps:**

1. Download Graphviz from:
   [https://graphviz.org/download/](https://graphviz.org/download/)
2. Install it and note the installation path (usually: `C:\Program Files\Graphviz\bin`).
3. Add that path to **System Environment Variables** under `PATH`.
4. Verify installation by running in **Command Prompt**:

   ```bash
   dot -V
   ```

   This should show the version of Graphviz installed.

---

## **5. Data Preparation**

* Loaded the dataset into a DataFrame.
* Separated **features** (X) and **target** (y).
* Split the dataset into **training** and **testing** sets to evaluate performance.

---

## **6. Model Training**

We trained three models:

1. **Full Decision Tree** (no depth limit) → Very accurate but can overfit.
2. **Limited Depth Decision Tree** (e.g., `max_depth=3`) → Simpler, less likely to overfit.
3. **Random Forest** → An ensemble of decision trees for better performance.

---

## **7. Feature Importance**

* Calculated **feature importance** from the Random Forest model.
* Ranked which features influenced predictions the most.
* Example: `cp`, `ca`, `thalach`, `oldpeak`, and `thal` were highly important.

---

## **8. Model Evaluation**

* Checked **accuracy** on the test data.
* Performed **cross-validation** to get a more reliable performance score.
* Example output:

  ```
  Decision Tree Accuracy: 0.985
  Limited Depth Tree Accuracy: 0.800
  Random Forest Accuracy: 0.985
  Average CV Score: 0.997
  ```

---

## **9. Graph Output**



---



