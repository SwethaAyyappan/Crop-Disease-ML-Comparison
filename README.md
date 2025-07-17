


#  Crop Disease Prediction and Recommendation using Machine Learning

This project compares a range of machine learning algorithms for two major agricultural tasks:

- **Crop Recommendation** — based on soil and environmental features.
- **Plant Disease Prediction** — using deep features extracted from leaf images with ResNet101.

---

## 📁 Project Structure

Crop-Disease-ML/
├── code/
│ ├── model/
│ │ ├── code/ # ML models (SVM, RF, XGBoost, etc.)
│ │ └── trained_models/ # Saved .pkl and .h5 models
│ ├── pre/ # ResNet101 feature extractor (pre.py)
│ └── results/ # Accuracy plots
├── dataset/ # Input CSV and image data folders
├── catboost_info/ # Logs from CatBoost training
├── ml_revised.pdf # Final project report
├── README.md
└── requirements.txt



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






