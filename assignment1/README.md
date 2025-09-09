Student Performance Prediction Report

--------------------------------------------------
1. Problem Identification
--------------------------------------------------
The problem addressed is predicting student performance in terms of both final exam scores and pass/fail outcomes. By analyzing demographic, behavioral, and academic factors, we aim to build machine learning models that can identify students at risk of failing and estimate their exam scores.

- Regression target variable: Final_Exam_Score (numeric 0–100).
- Classification target variable: Pass_Fail (categorical: Pass/Fail).
- Independent features: Gender, Study Hours per Week, Attendance Rate, Past Exam Scores,
  Parental Education Level, Internet Access at Home, Extracurricular Activities.

--------------------------------------------------
2. Data Collection
--------------------------------------------------
- Dataset: Student performance dataset with 708 rows and 10 columns.
- Data quality: No missing values. Features include both numerical and categorical data, requiring preprocessing.
- Preprocessing steps:
  * Dropped Student_ID (not predictive).
  * Encoded categorical variables with one-hot encoding. labels or categories
  * Split dataset into training (80%) and testing (20%), stratified on Pass/Fail.

--------------------------------------------------
3. Model Development
--------------------------------------------------
Trained two regression models and two classification models:

- Regression (predict Final Exam Score)
  * Linear Regression
  * Random Forest Regressor

- Classification (predict Pass/Fail)
  * Logistic Regression
  * Random Forest Classifier

All models were implemented in scikit-learn using pipelines.

--------------------------------------------------
4. Results
--------------------------------------------------

Regression Performance (Final Exam Score)
-----------------------------------------
| Model                  | MSE   | RMSE  | R²   |
|-------------------------|-------|-------|------|
| Linear Regression       | 19.823| 4.452 | 0.514|
| Random Forest Regressor | 14.207| 3.769 | 0.652|

Some definitions:
MSE (Mean Squared Error) → the average of the squared differences between predicted and actual scores. Smaller = better.
RMSE (Root Mean Squared Error) → the square root of MSE, same units as exam scores. This tells you how far off the model is on average.
R² (R-squared) → explains how much of the variation in exam scores is captured by the model. R² = 0.65 means the model explains 65% of the score variation.


Interpretation: Random Forest Regressor achieved higher R² and lower RMSE than Linear Regression, meaning it captured more variance and predicted exam scores more accurately.

Classification Performance (Pass/Fail)
--------------------------------------
| Model                  | Accuracy | Precision | Recall | F1   | ROC AUC |
|-------------------------|----------|-----------|--------|------|---------|
| Logistic Regression     | 0.768    | 0.750     | 0.803  | 0.776| 0.870   |
| Random Forest Classifier| 0.880    | 0.821     | 0.972  | 0.890| 0.980   |

Interpretation: Random Forest Classifier outperformed Logistic Regression, with higher accuracy, recall, F1, and ROC AUC. The ROC AUC of 0.98 indicates excellent discriminative ability.

Some definitions:
Accuracy → overall % of correct predictions. (88% means 88 out of 100 students correctly classified.)
Precision → when the model predicts “Pass,” how often it’s correct. (High precision = fewer false alarms.)
Recall → of all the actual “Pass” students, how many were correctly identified. (High recall = the model misses fewer passing students.)
F1 Score → the balance of precision and recall. Useful when both false positives and false negatives matter.
ROC AUC (Area Under the ROC Curve) → measures how well the model separates Pass vs Fail. AUC = 1 is perfect; AUC = 0.5 is guessing.

Visual Results:
- Confusion Matrix: Showed strong prediction ability, especially for the Random Forest model, with very few misclassifications.
- ROC Curve: Random Forest Classifier achieved near-perfect separation between classes.

--------------------------------------------------
5. Interpretation & Conclusion
--------------------------------------------------
- Regression: Random Forest Regressor is more effective for predicting numeric exam scores compared to Linear Regression.
- Classification: Random Forest Classifier is the most suitable for predicting pass/fail, outperforming Logistic Regression across all metrics.
- Overall: Classification is useful for flagging at-risk students, while regression provides insight into expected exam performance levels.