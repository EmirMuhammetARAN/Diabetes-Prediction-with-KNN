# Diabetes Prediction using K-Nearest Neighbors (KNN)

This project uses the **K-Nearest Neighbors (KNN)** algorithm to predict whether a person is diabetic or not, based on diagnostic measurements from the popular Pima Indians Diabetes Dataset.

---

## Dataset

The dataset used is the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), which includes health-related data such as glucose level, BMI, age, insulin, and more.

**Total Samples:** 768  
**Classes:**  
- 0 → Non-diabetic  
- 1 → Diabetic

---

## Data Cleaning & Preprocessing

Some features in the dataset contain zero values that are not physically meaningful. These were replaced using:

- Mean for `Glucose`, `BloodPressure`
- Median for `SkinThickness`, `Insulin`, `BMI`

After cleaning:
- **Features were standardized** using `StandardScaler`
- **Stratified train-test split** was performed (67% train, 33% test)

---

## Model: K-Nearest Neighbors

- Optimal `k` value was chosen as **10** by visualizing training and testing scores.
- Model was trained using `KNeighborsClassifier(n_neighbors=10)`.

---

## Evaluation

```python
Accuracy Score: 0.78
[[145  16]
 [ 36  63]]
              precision    recall  f1-score   support

           0       0.80      0.90      0.85       161
           1       0.80      0.64      0.71        99

    accuracy                           0.80       260
   macro avg       0.80      0.77      0.78       260
weighted avg       0.80      0.80      0.79       260
