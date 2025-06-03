# Elevate_labs_task_6


---

````markdown
# KNN Classification - Heart Disease Prediction

This project demonstrates how to use the **K-Nearest Neighbors (KNN)** algorithm for classifying heart disease presence based on patient data. It includes data preprocessing, model training, evaluation, and visualization of decision boundaries.

---

## 📁 Dataset

- **Source**: `heart.csv`
- **Target**: `target` (0 = No Disease, 1 = Disease)
- **Features**: Age, Cholesterol, Chest Pain Type, Max Heart Rate, etc.

---

## 🚀 Features

1. **Data Normalization** using `StandardScaler`
2. **Dimensionality Reduction** with PCA for visualization
3. **KNN Classification** with varying values of `k`
4. **Accuracy Evaluation** and **Confusion Matrix**
5. **Decision Boundary Visualization** for each `k`

---

## 🧪 Dependencies

- Python 3.7+
- pandas
- numpy
- matplotlib
- scikit-learn

Install all dependencies using:

```bash
pip install -r requirements.txt
````

---

## 🧠 How It Works

```bash
# Run the main script
python main.py
```

The script will:

* Normalize features
* Split the data
* Train `KNeighborsClassifier` with multiple values of `k`
* Show decision boundaries (2D PCA-reduced)
* Display confusion matrix for the best model

---

## 🔍 Sample Output

* Accuracy for different `k` values plotted
* Decision boundary plots
* Best model's confusion matrix

---

## 📊 Visualization

* PCA is used to reduce data to 2D for plotting decision boundaries
* Confusion matrix is shown for best `k` value based on accuracy

---

## 📁 File Structure

```
Intern_Task_6/
│
├── heart.csv                # Dataset file
├── main.py                  # Main Python script
├── README.md                # Project documentation
└── requirements.txt         # Dependencies
```

---

## 📌 Notes

* Make sure `heart.csv` is placed in the same directory or update the path in `main.py`.
* PCA is only used for **visualization**, not for model training.

---

## ✅ Example Results

```
Best K = 5, Accuracy = 0.87
Confusion Matrix:
[[23  3]
 [ 2 33]]
```

---
