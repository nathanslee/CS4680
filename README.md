
Problem: Predict whether students will pass or fail based on study habits, attendance, and background factors.

Target variables:
Regression → Final_Exam_Score (numeric, 0–100).
Classification → Pass_Fail (categorical: Pass/Fail).

Features (independent variables):
Gender
Study_Hours_per_Week
Attendance_Rate
Past_Exam_Scores
Parental_Education_Level
Internet_Access_at_Home
Extracurricular_Activities

Data Collection

Dataset size: 708 students.
Features: Includes both numeric (study hours, attendance, scores) and categorical (gender, parental education, internet access, extracurriculars).

We will train two types of models:

Regression (predict final exam score):
Linear Regression
Random Forest Regressor

Classification (predict pass/fail):
Logistic Regression
Random Forest Classifier