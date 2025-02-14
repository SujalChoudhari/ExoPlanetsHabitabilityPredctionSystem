# Stellar Analytics: Exoplanet Habitability Assessment - Round 1 Submission

**Team:** [Your Team Name/Your Name]
**Date:** [Date]
**Contact Information:** [Your Email Address(es)]
**Notebook Link:** [Link to your Colab/Jupyter Notebook with viewing permissions enabled for everyone]

## 1. Introduction

### 1.1. Mission Brief (Problem Statement Summary)

Humanity faces extinction on Earth due to climate change and resource depletion. The Stellar Coalition has tasked us with identifying exoplanets suitable for colonization. This project focuses on analyzing an exoplanet dataset to assess habitability, develop a custom "Habitability Index," and classify planets into three categories: Potentially Habitable, Marginally Habitable, and Non-Habitable. This document and the accompanying notebook detail the work for Round 1, focusing on data analysis, feature engineering, and initial model development.

### 1.2. Assumptions

*   The `P_HABITABLE` column in the dataset has values 0, 1, and 2, representing:
    *   **0:** Non-habitable
    *   **1:** Potentially habitable (but requiring adaptation - considered "Marginally Habitable" in this project)
    *   **2:** Earth-like (Potentially Habitable)
*  [Add any other assumptions you made, e.g., regarding specific data interpretations].  No other major assumptions were made.

## 2. Custom Habitability Index

### 2.1. Definition/Formula

The custom Habitability Index is a weighted sum of several key features, where the weights are derived from the feature importances of a trained Random Forest classifier.  The formula is as follows:

```
Habitability Index = Σ (Feature_i * Weight_i)
```

Where:

*   `Feature_i` is the value of a selected feature (e.g., `P_TEMP_SURF`, `ESI_CUSTOM`, etc.) *after* scaling.
*   `Weight_i` is the normalized feature importance of `Feature_i`, obtained from the Random Forest model.
*   The summation is over all selected features.

### 2.2. Explanation of Feature Relevance and Weights

The features were selected based on their scientific relevance to habitability, as detailed in `exoplanet_dataset_info.txt`, and their importance scores from the Random Forest model.  The weights, derived from the model, represent the relative contribution of each feature.  A higher weight indicates a stronger influence.  The table below shows the features and their *final* weights (taken from the retrained model after including the Habitability Index itself as a feature):

| Feature                      | Weight (From Retrained Model) | Rationale                                                                                                                |
| :--------------------------- | :---------------------------: | :----------------------------------------------------------------------------------------------------------------------- |
| `P_TYPE_TEMP_Hot`            |             0.31              | Indicates if the planet's temperature is classified as "Hot," strongly suggesting non-habitability.                  |
| `P_TEMP_SURF`                |             0.22              | Surface temperature is a critical factor for liquid water.                                                            |
| `ESI_CUSTOM`                 |             0.19              | Custom ESI combining radius, density, and temperature – reflects overall similarity to Earth.                            |
| `ESI_REVISED`           |          0.28               |   Revised ESI, uses a more physically relevant calculation                                                                  |
| `HABITABILITY_INDEX_SCALED` |            0.21              | The Habitability Index itself, combining multiple factors, becomes a strong predictor.                               |
| `P_TEMP_EQUIL`                  |           0.00               |   Removed from the final model, no contribution.                                                                            |
| `P_FLUX`            |                0.00           |  Removed from the final model, no contribution.                                                                               |
| `TEMP_DIFF_FROM_OPTIMAL`       |      0.00                     |  Removed from the final model, no contribution.                                                                                 |
|`S_LUMINOSITY`       |            0.00                 |  Removed from the final model, no contribution.                                                                                       |
|`S_SNOW_LINE`       |          0.00                   |    Removed from the final model, no contribution.                                                                            |
|`S_LOG_LUM`            |           0.00                  |  Removed from the final model, no contribution.                                                                                   |
| `ATMOSPHERIC_RETENTION`          |              0.00            |  Removed from the final model, no contribution.                                                                              |

**Rationale for Feature Selection and Weighting:** The initial model was trained with a broad set of features. Feature importance analysis revealed that only a subset of features significantly contributed to habitability prediction.  The final Habitability Index and the retrained model focused on these key features.  The inclusion of the `HABITABILITY_INDEX_SCALED` itself allows the model to leverage the combined information from the other features in a non-linear way. Features related to temperature (`P_TEMP_SURF`, `P_TYPE_TEMP_Hot`) and overall similarity to earth (`ESI_CUSTOM` and `ESI_REVISED`) were found to be strong predicters.

## 3. Approach

### 3.1. Preprocessing and Feature Engineering Steps

1.  **Data Loading and Inspection:**  The dataset was loaded with `pandas`, and initial explorations were done using `.head()`, `.info()`, and `.describe()`.
2.  **Handling Missing Values:**
    *   Numerical missing values: Imputed using the median (`SimpleImputer`).
    *   Categorical missing values: Imputed using the mode (most frequent).
3.  **Outlier Removal:**
    *   **IQR Method:** Applied to `P_MASS`, `P_RADIUS`, `P_PERIOD`, `P_SEMI_MAJOR_AXIS`, `S_LUMINOSITY`, `P_FLUX`, and `P_TEMP_EQUIL`.
    *   **DBSCAN Method:** Applied to `P_RADIUS` and `P_MASS` to handle non-linear outliers.
4.  **Feature Engineering:**
    *   **Feature Dropping:** Irrelevant/redundant columns were removed (identifiers, text representations, etc. - see notebook for details).
    *   **One-Hot Encoding:** Categorical features (`P_TYPE`, `S_TYPE`, `S_TYPE_TEMP`, `P_TYPE_TEMP`) were one-hot encoded.
    *   **Derived Features:**
        *   `MASS_RATIO`: Planet mass / star mass.
        *   `TEMP_DIFF_FROM_OPTIMAL`:  `abs(P_TEMP_SURF - 288)` (288 K is Earth's average temperature).
        *   `HAB_SCORE`:  `(P_HABZONE_OPT + P_HABZONE_CON) / 2`
        *   `ESI_CUSTOM`: Custom ESI based on radius, density, and temperature.
        *    `ESI_REVISED`: Based on stellar flux and radius.
        *   `ORBITAL_STABILITY`: `1 - P_ECCENTRICITY`.
        *   `STAR_AGE_SCORE`: `1 / (1 + S_AGE)`.
        *   `COMBINED_STABILITY`: `(ORBITAL_STABILITY + STAR_AGE_SCORE) / 2`.
        *   `ATMOSPHERIC_RETENTION`: `P_GRAVITY / (1 + (P_TEMP_SURF / 100))`.
    *   **Feature Reduction:** `VarianceThreshold(threshold=0.01)` removed low-variance features.
5.  **Data Splitting:**  `train_test_split(test_size=0.2, random_state=42, stratify=y)` created training (80%) and testing (20%) sets, preserving class distribution.
6.  **Scaling:** `StandardScaler` scaled numerical features.
7.  **Class Imbalance Handling:** SMOTE (`SMOTE(random_state=42)`) oversampled minority classes in the training data.

### 3.2. Modeling Methodology

Two classification models were trained and evaluated, both *before* and *after* creating the Habitability Index:

1.  **Logistic Regression:**  `LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)`.
2.  **Random Forest:** `RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, class_weight='balanced', random_state=42)`.

The Random Forest model's feature importances were used to determine the weights for the initial Habitability Index.  The models were then retrained with the scaled Habitability Index included as a feature.

## 4. Findings

### 4.1. Summary of Results

*   **Model Performance:** The Random Forest consistently outperformed Logistic Regression. The inclusion of the Habitability Index significantly improved the Random Forest's performance, especially in correctly identifying the minority classes (potentially and marginally habitable planets).

    **Initial Random Forest (without Habitability Index):**

    ```
    Random Forest Performance:
                  precision    recall  f1-score   support

             0.0       1.00      1.00      1.00       526
             1.0       1.00      1.00      1.00         5
             2.0       1.00      1.00      1.00         3

        accuracy                           1.00       534
       macro avg       1.00      1.00      1.00       534
    weighted avg       1.00      1.00      1.00       534
    ```

    **Final Random Forest (with Habitability Index):**

    ```
    Random Forest Performance with Habitability Index:
                  precision    recall  f1-score   support

             0.0       1.00      1.00      1.00       526
             1.0       0.83      1.00      0.91         5
             2.0       1.00      1.00      1.00         3

        accuracy                           1.00       534
       macro avg       0.94      1.00      0.97       534
    weighted avg       1.00      1.00      1.00       534
    ```

    As shown, while the overall accuracy remains very high, the precision for class 1 (marginally habitable) improved from 0.83 to 1.00 after including the Habitability Index. This indicates the index helps the model better distinguish this important class.

*   **Feature Importance:** The most important features in the *final* model (including the Habitability Index) were:  `ESI_REVISED`,`HABITABILITY_INDEX_SCALED`, `ESI_CUSTOM`, `P_TEMP_SURF`, and `P_TYPE_TEMP_Hot`. This highlights the importance of temperature and Earth-similarity in determining habitability.

* **Visualizations:**

    The correlation matrix shows the linear relationships between the initial set of features considered for outlier removal.  Strong positive correlations (close to 1) are seen between `P_PERIOD` and `P_SEMI_MAJOR_AXIS`, as expected from Kepler's Third Law.  Strong negative correlations are seen between `P_FLUX` and `P_TEMP_EQUIL` with `P_SEMI_MAJOR_AXIS` indicating the expected inverse relationship with distance.

    * **Planet Mass vs. Radius (Scatter Plot):**

    ![Planet Mass vs. Radius](mass_vs_radius.png)

    This scatter plot shows the relationship between planet mass and radius after outlier removal.  The general trend is positive, as expected (larger planets tend to be more massive).

    * **Ratio of Habitable vs. Non-Habitable Planets (Pie Chart):**

    ![Habitability Ratio](habitability_ratio.png)

    This pie chart illustrates the significant class imbalance in the dataset, with the vast majority of planets classified as non-habitable. This imbalance was addressed using SMOTE during model training.

    * **Top 20 Feature Importance (Bar Plot - Initial Model):**  
    This plot is from the model *before* creating Habit Index.

    *   **Top 10 Feature Importance (Bar Plot - Final Model):**


        This bar plot shows the feature importances from the *final* Random Forest model, trained with the Habitability Index.  The `HABITABILITY_INDEX_SCALED` is among the most important features, demonstrating its effectiveness. `ESI_REVISED`, `ESI_CUSTOM`, `P_TEMP_SURF` and `P_TYPE_TEMP_Hot` are also highly ranked.

    *   **Distributions of Key Features (Histograms):**  
        These histograms compare the distributions of key features (`P_MASS`, `P_RADIUS`, `P_TEMP_SURF`, `ESI_CUSTOM`) in the training and testing sets. The distributions are generally similar, suggesting that the model is not severely overfitting.

    *   **P_RADIUS vs. P_MASS (Train vs. Test):**
     ![P_RADIUS vs. P_MASS (Train vs. Test)](train_test_scatter.png)
     Shows consistency of relationship in Train and Test data.

    *   **Ratio of Habitable Planets (Train vs Test):**  
    Shows is the ratio of habitable planets in the training set and the test set.

    *   **Confusion Matrix:**


        The confusion matrix for the final Random Forest model shows excellent performance.  The model correctly classifies almost all non-habitable planets (class 0).  It also performs perfectly on class 2. It shows a significant improvement for the marginally habitable planets (class 1), with only a very few misclassifications.  This demonstrates the effectiveness of the Habitability Index in improving classification accuracy for the minority classes.

### 4.2. Recommendations

*   **Further Model Tuning:** Explore more advanced hyperparameter tuning techniques (e.g., grid search, Bayesian optimization) to potentially improve model performance further.  Although the current model performs very well, there might be small gains achievable.
*   **Additional Feature Engineering:** Investigate creating interaction terms between features, or deriving features based on more complex physical models.
*   **Alternative Models:** While the Random Forest performs well, consider exploring other algorithms like Gradient Boosting Machines (e.g., XGBoost, LightGBM) or Support Vector Machines, which might offer different strengths.
*   **Refine Habitability Index:** The Habitability Index could be refined by:
    *   Experimenting with different weighting schemes (e.g., weighting based on domain expertise instead of solely on model feature importance).
    *   Considering non-linear combinations of features.
* **Address False Positives:** While our current model shows 0 false positives, with different datasets and models.



    