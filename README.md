


#  Crop Disease Prediction and Recommendation using Machine Learning

This project compares a range of machine learning algorithms for two major agricultural tasks:

- **Crop Recommendation** — based on soil and environmental features.
- **Plant Disease Prediction** — using deep features extracted from leaf images with ResNet101.

---

## 📁 Project Structure




```
Crop-Disease-ML/
├── catboost_info
│   ├── catboost_training.json
│   ├── learn
│   │   └── events.out.tfevents
│   ├── learn_error.tsv
│   ├── test
│   │   └── events.out.tfevents
│   ├── test_error.tsv
│   ├── time_left.tsv
│   └── tmp
├── code
│   ├── model
│   │   ├── code
│   │   │   ├── _eval.py
│   │   │   ├── adaboost.py
│   │   │   ├── bagging.py
│   │   │   ├── cat_boost_classifier.py
│   │   │   ├── decision_trees.py
│   │   │   ├── deeplearning.py
│   │   │   ├── gradiant_boosting.py
│   │   │   ├── knn.py
│   │   │   ├── logistic_regression.py
│   │   │   ├── naive_bayes.py
│   │   │   ├── neuralnet.py
│   │   │   ├── perceptron.py
│   │   │   ├── random_forest.py
│   │   │   ├── svm.py
│   │   │   └── xg_boost_classifier.py
│   │   └── trained_models
│   │       ├── adaboost.pkl
│   │       ├── bagging.pkl
│   │       ├── catboost_model.pkl
│   │       ├── decision_tree.pkl
│   │       ├── deep_learning_model.h5
│   │       ├── gradient_boosting.pkl
│   │       ├── knn.pkl
│   │       ├── logistic_regression.pkl
│   │       ├── naive_bayes.pkl
│   │       ├── neural_net.pkl
│   │       ├── perceptron.pkl
│   │       ├── random_forest.pkl
│   │       ├── svm.pkl
│   │       └── xgboost_model.pkl
│   ├── pre
│   │   └── pre.py
│   └── results
│       ├── catboost.png
│       └── xgboost.png
└── dataset
    └── features_with_labels.csv

```

---

## 🧪 ML Models Implemented

> Total: 13 classifiers

- Logistic Regression, Decision Tree, Random Forest, Naive Bayes
- Perceptron, K-Nearest Neighbors, SVM, Bagging, AdaBoost
- Gradient Boosting, XGBoost, CatBoost
- Deep Neural Network

---

## 🔍 Datasets Used

- **Crop Recommendation:** Soil and climate features (N, P, K, temp, humidity, pH, rainfall)
- **Disease Prediction:** Plant leaf images converted to feature vectors using **ResNet101**

---

## 🏆 Results Summary

| Task               | Best Model | Accuracy |
|--------------------|------------|----------|
| Crop Recommendation | Bagging    | 97.27%   |
| Disease Prediction  | XGBoost    | 91.4%    |

---

## ⚙️ Getting Started

### 1. Clone the Repository
```
git clone https://github.com/SwethaAyyappan/Crop-Disease-ML-Comparison.git
cd Crop-Disease-ML-Comparison
```


### 2. Install Dependencies
```
pip install -r requirements.txt

```


Preprocessing (ResNet Feature Extraction)
Before training disease classifiers:
```
python code/pre/pre.py

```
This script extracts deep features using ResNet101 and stores them in features.csv.

Training a Classifier
Example (Random Forest):
```
python code/model/code/random_forest.py

```
Each model has its own script inside code/model/code/.






