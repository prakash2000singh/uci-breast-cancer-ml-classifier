# Breast Cancer Diagnostic Classification & Streamlit Deployment

## a. Problem Statement
Build, evaluate, and deploy an end-to-end supervised Machine Learning classification system to predict whether a breast mass is **Malignant (Class 0)** or **Benign (Class 1)** using digitized Fine Needle Aspirate (FNA) cell nuclei measurements fetched directly from the **UCI Machine Learning Repository**. Five distinct classification models (**Logistic Regression**, **Decision Tree**, **K-Nearest Neighbor**, **Gaussian Naive Bayes**, and **Support Vector Machine**) are trained on standardized morphological features, evaluated across six performance metrics (**Accuracy**, **ROC-AUC**, **Precision**, **Recall**, **F1-Score**, and **Matthews Correlation Coefficient**), serialized as `.joblib` artifacts, and deployed as an interactive web application on **Streamlit Community Cloud**.

---

## b. Dataset Description
* **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset (WDBC)
* **Public Repository Source:** UCI Machine Learning Repository
  * **Dataset Landing Page:** [https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
  * **Direct UCI Data File URL:** [https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data)
* **Classification Task:** Binary Classification (`M -> 0 = Malignant`, `B -> 1 = Benign`; non-predictive `id` column removed)
* **Total Instances:** **569** rows (Exceeds the assignment minimum requirement of 500 instances)
  * **Class 0 (Malignant):** 212 instances (37.26%)
  * **Class 1 (Benign):** 357 instances (62.74%)
* **Total Features:** **30** continuous numerical input features (Exceeds the assignment minimum requirement of 12 features). The features quantify the **mean**, **standard error (error)**, and **worst (largest)** values of 10 cell nucleus characteristics:
  1. Radius (mean of distances from center to points on the perimeter)
  2. Texture (standard deviation of gray-scale values)
  3. Perimeter
  4. Area
  5. Smoothness (local variation in radius lengths)
  6. Compactness ($\text{perimeter}^2 / \text{area} - 1.0$)
  7. Concavity (severity of concave portions of the contour)
  8. Concave points (number of concave portions of the contour)
  9. Symmetry
  10. Fractal dimension ("coastline approximation" - 1)
* **Train / Test Split:** Stratified 80% / 20% split (`random_state = 5782`), yielding **455 training instances** and **114 held-out test instances** exported to `test_data.csv`.
* **Feature Preprocessing:** Standardized using `StandardScaler` fit strictly on the training set (`X_train`) and saved as `model/scaler.joblib` to prevent data leakage during test inference.

---

## c. Github Repository & Live Deployment Links
* **GitHub Repository Link:** (https://github.com/prakash2000singh/uci-breast-cancer-ml-classifier)
* **Live Streamlit App Link:** (https://uci-breast-cancer-classifier-aiml.streamlit.app/)

---

## d. Models Used: Evaluation Metrics Comparison Table

All 5 classification models were evaluated on the held-out UCI test dataset (`test_data.csv`, $N = 114$ instances):

| ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.9825 | 0.998  |      0.973  |   1      | 0.9863 | 0.9626 |
| Decision Tree          |     0.9035 | 0.8722 |      0.8961 |   0.9583 | 0.9262 | 0.7911 |
| KNN                    |     0.9737 | 0.9828 |      0.96   |   1      | 0.9796 | 0.9442 |
| Naive Bayes            |     0.9386 | 0.9911 |      0.9452 |   0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine |     0.9912 | 0.9987 |      0.9863 |   1      | 0.9931 | 0.9812 |

---

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved **Accuracy = 0.9825**, **AUC = 0.9980**, **Precision = 0.9730**, **Recall = 1.0000**, **F1 = 0.9863**, and **MCC = 0.9626**. Because standardized FNA morphological features exhibit strong linear separability in high-dimensional space, L2-regularized Logistic Regression establishes a highly reliable linear decision boundary with balanced precision and recall. |
| **Decision Tree** | Achieved **Accuracy = 0.9035**, **AUC = 0.8722**, **Precision = 0.8961**, **Recall = 0.9583**, **F1 = 0.9262**, and **MCC = 0.7911**. Axis-aligned recursive splits (`max_depth=5`) struggle to approximate diagonal decision boundaries across collinear geometric features (such as radius, perimeter, and area), resulting in higher variance on unseen test samples compared to linear/kernel classifiers. |
| **KNN** | Achieved **Accuracy = 0.9737**, **AUC = 0.9828**, **Precision = 0.9600**, **Recall = 1.0000**, **F1 = 0.9796**, and **MCC = 0.9442** using `n_neighbors=7` on standardized features. Feature scaling prevents large-magnitude attributes (such as area) from dominating Euclidean distance calculations, allowing local neighborhood voting to achieve strong sensitivity. |
| **Naive Bayes** | Achieved **Accuracy = 0.9386**, **AUC = 0.9911**, **Precision = 0.9452**, **Recall = 0.9583**, **F1 = 0.9517**, and **MCC = 0.8676**. While Gaussian Naive Bayes delivers fast probabilistic predictions and a solid ROC-AUC (0.9911), its strict conditional independence assumption is partially violated by correlated nuclei measurements (radius, perimeter, area, concavity), which limits its MCC compared to discriminative classifiers. |
| **Support Vector Machine** | Achieved **Accuracy = 0.9912**, **AUC = 0.9987**, **Precision = 0.9863**, **Recall = 1.0000**, **F1 = 0.9931**, and **MCC = 0.9812**. The RBF kernel effectively captures non-linear feature interactions in a higher-dimensional space while maximum-margin regularization prevents overfitting on borderline diagnostic cases. |
| **Overall, Winner for your dataset?** | **Support Vector Machine** is the overall best-performing model for this dataset (closely followed by **Logistic Regression**), achieving top composite scores across the test set (**Accuracy = 0.9912**, **AUC = 0.9987**, **Precision = 0.9863**, **Recall = 1.0000**, **F1 = 0.9931**, and **MCC = 0.9812**). Conversely, **Decision Tree** recorded the lowest overall MCC (0.7911). |
