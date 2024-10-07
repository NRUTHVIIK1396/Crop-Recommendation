# Crop Recommendation Predictive Analysis

## 1. Executive Summary

This report presents the findings of a predictive analysis conducted to recommend suitable crops for cultivation based on soil and climate conditions. Using machine learning algorithms, developed a predictive model that analyzes key environmental factors such as soil nutrients, pH, rainfall, temperature, and humidity to suggest optimal crops. Using clustering techniques, found various groups of crops that can be grown in similar soil and weather conditions. This approach helps farmers make informed decisions, thereby improving agricultural productivity and reducing resource waste.

## 2. Introduction

### 2.1 Background

Agriculture is a critical sector in many economies, and the decision of which crop to grow is often influenced by a variety of environmental factors. Inaccurate crop selection can lead to poor yields and inefficient use of resources. To address this, this project aims to leverage machine learning to create a recommendation system that identifies the most suitable crops based on soil and climate conditions.

### 2.2 Objective

- The primary objective of this project is to develop a machine learning model that can recommend crops based on key environmental variables. Specifically, the aim is to improve accuracy and provide actionable insights that can be used by farmers to optimize crop selection.
- Suggest crops that can be grown together to apply crop rotation technique.

## 3. Data Description

The dataset used in this analysis contains the following key features:

- **Nitrogen (N)**: Nitrogen content in the soil (kg/ha).
- **Phosphorus (P)**: Phosphorus content in the soil (kg/ha).
- **Potassium (K)**: Potassium content in the soil (kg/ha).
- **Temperature (°C)**: The average temperature.
- **Humidity (%)**: The average humidity level.
- **pH**: Soil pH value (acidity/alkalinity).
- **Rainfall (mm)**: Total rainfall.
- **Crop Name**: The target variable indicating the crop recommended for the given soil and climate conditions.

This dataset is sourced from Kaggle. <a>Link</a> to dataset.

## 3. Methodology

### 3.1 Machine Learning Models

Several supervised learning classifiers were used to predict the most suitable crop:

1. **Logistic Regression**: A linear model to predict the crop class based on the probability of the features.
2. **Decision Tree Classifier**: A tree-based model used to capture non-linear relationships between the features.
3. **Support Vector Machine (SVM)**: A model that aims to maximize the margin between different crop classes.
4. **K-Nearest Neighbors (KNN)**: A distance-based model that classifies crops based on the nearest neighbors in the feature space.
5. **Stacking Classifier**: An ensemble machine learning technique that combines multiple models (or base learners) to improve predictive performance.

### 3.2 Hyperparameter Tuning

To enhance the performance of the models, **GridSearchCV** was used for hyperparameter tuning. The models were trained and validated using cross-validation techniques to prevent overfitting.

### 3.3 Evaluation Metrics

The models were evaluated based on the following metrics:
- **Accuracy**: Percentage of correct predictions.
- **Precision**: The ratio of true positive crops to all predicted crops.
- **Recall**: The ratio of true positive crops to all actual crops.
- **F1 Score**: The harmonic mean of precision and recall, providing a balanced measure of the model's performance.

## 4. Results

### 4.1 Model Performance

The performance of each model was measured across the dataset:

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| KNN                 | 0.9712   | 0.9705    | 0.97   | 0.9713   |
| Logistic Regression | 0.9712   | 0.9715    | 0.97   | 0.9708   |
| Decision Tree       | 0.9864   | 0.9863    | 0.9863 | 0.9863   |
| SVM                 | 0.9803   | 0.9793    | 0.9793 | 0.9803   |

To improve model accuracy, stacking classifier with various meta models were analyzed and the results are:

| Model                    | Accuracy | Precision | Recall | F1 Score |
|--------------------------|----------|-----------|--------|----------|
| SC-Logistic Regression   | 0.9833   | 0.9824    | 0.9824 | 0.9833   |
| SC-Decision Tree         | 0.9756   | 0.9753    | 0.9753 | 0.9759   |
| SC-SVM                   | 0.8121   | 0.7974    | 0.7974 | 0.7908   |
| SC-KNN                   | 0.9363   | 0.9350    | 0.9350 | 0.9354   |
| SC-RandomForest          | 0.9848   | 0.9842    | 0.9842 | 0.9847   |

### 4.2 Cross Validation

| Model               | Mean Accuracy | Standard Deviation |
|---------------------|---------------|--------------------|
| Decision Tree       | 0.9859        | 0.0063             | 
| SC-Random Forest    | 0.9900        | 0.0068             | 

### 5.2 Feature Importance

The most important features that contributed to crop prediction were:
1. **Humidity**
2. **Rainfall**
3. **Nitrogen content**

These features play a significant role in determining which crop is suitable for a particular region.

### 5.3 Clustering Analysis

A KMeans clustering analysis was also performed on the most important features (Humidity and Rainfall) to group similar crops. The elbow method was used to determine the optimal number of clusters (k), which resulted in 6 clusters.

### 5.4 Crop Grouping by Clusters

Crops were grouped into clusters based on their similarity in feature space. The following are the top 3 crops in each cluster:

| Cluster | Crops                                           |
|---------|-------------------------------------------------|
| 0       | Rice, Jute, Coconut, Papaya                     |
| 1       | Coffee, Pigeonpeas                              |
| 2       | Chickpea, Kidneybeans                           |
| 3       | Mungbean, Grapes, Watermelon, Muskmelon, Cotton |
| 4       | Pomegranate, Banana, Apple, Orange              |
| 5       | Maize, Mothbeans, Blackgram, Lentil Mango       |

## 6. Conclusion

### 6.1 Summary of Findings

The **Stacking classifier model with base models KNN, Logistic Regression, SVM, Decision Tree and meta model as RandomForest** outperformed other models, achieving an accuracy of 99%. The most important features contributing to the crop recommendation were humidity and rainfall, as well as nitrogen content in soil. Clustering analysis revealed distinct groups of crops that share similar soil and climatic requirements.

### 6.2 Implications

The predictive model developed in this project can be used by farmers and agricultural experts to make data-driven decisions about crop selection. This can lead to optimized resource usage, improved crop yields, and reduced crop failure rates. Such tools can have a significant impact on agricultural efficiency, particularly in regions where climate variability poses a challenge.

### 6.3 Future Work

Future improvements to this project could include:
- Incorporating additional features like soil moisture and more granular climate data.
- Expanding the dataset to include more crops and geographic regions.
- Using advanced ensemble methods, such as boosting or stacking, to improve accuracy further.

## 7. Appendix

### 7.1 Code Repository

The code and data used in this project can be found [here](#).

### 7.2 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Agricultural Data Resource](#) (Include dataset links or references)

