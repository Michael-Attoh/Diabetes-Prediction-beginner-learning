# **Diabetic Patient Prediction (Comprehensive Learning Tool for Beginners)**

This project is designed to help beginners learn how to build and automate a machine learning pipeline for predicting diabetic patients. It walks through every step of the process, including:
- Data Preprocessing: Cleaning and preparing the data for model training.
- Model Training and Evaluation: Using logistic regression to predict diabetic patients and evaluate the model's performance.
- Automation: Learn how to automate the entire workflow, from data loading to model evaluation.
- Scheduling: Implement triggers to automatically run the pipeline at specified intervals.
- Visualization: Explore metrics and insights from the model, such as accuracy, confusion matrix, and feature importance, through visualizations.
  
This repository is ideal for beginners looking to understand the full lifecycle of a machine learning project, including automation and visualization. Feel free to explore, modify, and learn as you build your own end-to-end solutions!
--
## **Steps in the Project Workflow:**

1. Import the Diabetes Data: Load the diabetes dataset into the environment.
2. Check for Null Values: Inspect the dataset to determine if there are any missing (null) values.
3. Check Dataset Dimensions: Verify how many rows and columns the dataset contains.
4. Check Column Data Types: Review the data types of each column in the dataset.
5. Generate Descriptive Statistics: Generate summary statistics for each column, including mean, median, min, max, and standard deviation.
6. Check Data Imbalance: Analyze the balance between diabetic and non-diabetic patients in the target variable.
7. Explore Each Variable: Identify the distribution, shape, and outliers of each variable.
8. Create Scatterplots for Each Variable: Visualize the relationships between different variables using scatterplots.
9. Generate Correlation for Each Variable:Compute the correlation between all variables.
10. Visualize the Correlation: Create a heatmap or other visual representation to display the correlation between variables.
11. Analyze the Target Variable: Compare how each feature differs for diabetic and non-diabetic patients.
12. Create X and Y Variables for Machine Learning: Define the feature set (X) and target variable (Y) for training the model.
13.Split the Data into Train and Test Sets: Divide the dataset into training and testing subsets.
14. Run Machine Learning Models and Evaluate: Implement the following models and calculate accuracy, classification report, and confusion matrix:
- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- XGBoost
15. Identify Key Variables: Find the variables that have the most significant impact on the model's predictions.
16. Test the Model with External Values: Validate the model using external test data to evaluate its generalization performance.

## Import libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

## Load data 
```python
data = pd.read_csv('data_for_ML/diabetes.csv')
```
## view the first few columns of the data
<img width="945" alt="datahead" src="https://github.com/user-attachments/assets/601e401c-4aaf-462b-b0e6-341415d2f181">

## Check for null values
Identify missing values and take appropriate action, such as removing or imputing them.

<img width="259" alt="isnull" src="https://github.com/user-attachments/assets/0a03c2fa-f57e-4c7c-a0a0-33eb1b257d6e">


## Check the summary statistics
```python
data.describe()
```
<img width="1111" alt="datadescribe" src="https://github.com/user-attachments/assets/fa57e07c-b625-4095-95a6-41fee3b6142e">


## Since it's a classification problem, check the class distribution of the target variable
```python
data['Outcome'].value_counts()
```
Then the percentage distribution of each class in the target variable:
```python
data['Outcome'].value_counts(normalize=True)*100
```
<img width="441" alt="valuecount" src="https://github.com/user-attachments/assets/9ed394ec-82ff-4b9f-b2a8-3e7aeae2b93c">

To address the class imbalance in our target variable, we can set the model’s class_weight parameter to 'balanced' to manage this imbalance more effectively.
```python
from sklearn.linear_model import LogisticRegression    
from sklearn.ensemble import RandomForestClassifier

logreg = LogisticRegression(class_weight='balanced', max_iter=1000)
RFclassifier = RandomForestClassifier(class_weight='balanced', random_state=42)
```
## Let's do some EDA
1. for Pregnancies
```python
plt.figure(2)
plt.subplot(121)   #121 - 1 row 2 plots 1st diagram
sns.distplot(data['Pregnancies'])
plt.subplot(122)
data['Pregnancies'].plot.box(figsize=(15,5))
```
<img width="1313" alt="snspregnancy" src="https://github.com/user-attachments/assets/cc476e21-22a9-4bcd-8520-eeb9c97637d6">

2. Visuals for each pair of features in the dataset
```python
sns.pairplot(data)
```
![pairplot](https://github.com/user-attachments/assets/3c10f8a6-d04e-4314-80e5-9c186f568152)

3. Correlation metric 
```python
data.corr()
```

4. Heat map
```python
ax = plt.subplots(figsize=(12,8))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, cmap="coolwarm")
```
<img width="1201" alt="heatmap" src="https://github.com/user-attachments/assets/54a17acf-cd1d-48b8-8bc0-6c101f570e39">

5. compare the average values of features across different classes
```python
data.groupby('Outcome').mean()
```
<img width="1028" alt="groupby" src="https://github.com/user-attachments/assets/836a9ac7-3aae-49db-a848-77ffd94b1b3e">

--
## Feature Separation 
```python
X = data.drop(columns='Outcome',axis=1)
Y = data['Outcome']
```
## Model
```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=.2,random_state=123)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(class_weight='balanced', max_iter=1000) #initiate model

logreg.fit(X_train, Y_train) #train model

logreg_predict = logreg.predict(X_test)  #test model
```
Check model performance
```python
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, logreg_predict)
```
<img width="354" alt="logregscore" src="https://github.com/user-attachments/assets/4c6ef93c-4738-4a73-9f0d-9e6048597774">

### Digression
Classification Model Perfomance Test
- Accuracy = Correct prediction /total prediciton
- Precision = TP/(TP + FP)
- Recall(sensitivity) = TP/(TP + FN)
- F1 Score = balances both precision and recall to give a single measure of performance.

##  Back to project
We can now test our mnodel **logreg** on any random data 
```python
test_data = (8,183,64,0,0,23.3,.672,32)   # assume this is a new data
test_data_np = np.asarray(test_data)

test_data_rs = test_data_np.reshape(1,-1)  #reshape new data

# Test the model to determine the prediction it will give us
test_predict = logreg.predict(test_data_rs)
test_predict  
```
<img width="164" alt="logreg new data score" src="https://github.com/user-attachments/assets/269521a3-640a-4634-a229-c83834d807c9">

## Confusion Matrix
```python
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(Y_test, logreg_predict))
print(classification_report(Y_test, logreg_predict))
```
<img width="511" alt="logreg confusion matrix" src="https://github.com/user-attachments/assets/ca8ba795-4705-46ed-a175-255858b202d8">


## End of simple Task.

# Proceeding further

## Other Models
```python
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Create an SVM model with balanced class weights
svm_model = SVC(class_weight='balanced', kernel='linear')

from sklearn.ensemble import RandomForestClassifier
rfc_model = RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42)

#train model
rfc_model.fit(X_train, Y_train)   

#test mdoel
rfc_predict = rfc_model.predict(X_test)

#check model's performance
accuracy_score(Y_test, rfc_predict)
```
<img width="410" alt="rfscore" src="https://github.com/user-attachments/assets/664ce1b0-31dd-4ab1-9c8e-017b5581d90b">

## Check Important features detectected by the mdoel
```python
rfc_model.feature_importances_
pd.Series(rfc_model.feature_importances_,index=X.columns).plot(kind='barh')
```
<img width="835" alt="rf keyimportant features" src="https://github.com/user-attachments/assets/de3e848b-d88c-4e7a-a8e1-4de047dca494">

# Automation
Instead of running each line of code, we can automate the whole process.
1. Automation with function
```python
# Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib

# Step 1: Data Loading
def load_data(file_path):
    return pd.read_csv(file_path)

# Step 2: Data Preprocessing
def preprocess_data(df):
    # Handle missing values if any
    df = df.fillna(method='ffill')  # Forward fill as an example
    
    # Feature and target separation
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # One-Hot Encoding for categorical features if necessary
    # Example: X = pd.get_dummies(X)
    
    return X, y

# Step 3: Model Training
def train_model(X_train, y_train):
    # Standardize features
    scaler = StandardScaler()
    
    # Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    
    # Create a pipeline
    pipeline = Pipeline([
        ('scaler', scaler),
        ('model', model)
    ])
    
    # Hyperparameter tuning with GridSearchCV
    param_grid = {'model__C': [0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    return best_model

# Step 4: Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Step 5: Save the Model
def save_model(model, file_path):
    joblib.dump(model, file_path)

# Step 6: Load and Predict (For future use)
def load_and_predict(file_path, model_path):
    # Load new data and model
    new_data = pd.read_csv(file_path)
    model = joblib.load(model_path)
    
    # Preprocess and predict
    X_new, _ = preprocess_data(new_data)
    y_pred = model.predict(X_new)
    return y_pred

# Full Pipeline Execution
if __name__ == "__main__":
    # File paths
    data_file_path = 'data_for_ML/diabetes.csv'
    model_file_path = 'diabetic_model.pkl'
    
    # Load data
    df = load_data(data_file_path)
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, model_file_path)
    
    print("Pipeline completed successfully.")

```

2. Automation with Class
```python
# Automated Pipeline with CLASS

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib

class DataPipeline:
    def __init__(self, data_file_path, model_file_path, predictions_file_path=None):
        """Initialize the pipeline with file paths."""
        self.data_file_path = data_file_path
        self.model_file_path = model_file_path
        self.predictions_file_path = predictions_file_path
    
    def load_data(self):
        """Load data from a CSV file."""
        self.df = pd.read_csv(self.data_file_path)
    
    def preprocess_data(self):
        """Preprocess the data."""
        # Handle missing values
        self.df = self.df.fillna(self.df.median())  # median fill
        
        # Feature and target separation
        self.X = self.df.drop('Outcome', axis=1)
        self.y = self.df['Outcome']
        
        # One-Hot Encoding for categorical features if necessary
        # Example: self.X = pd.get_dummies(self.X)
    
    def split_data(self):
        """Split the data into training and testing sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
    
    def train_model(self):
        """Train a Logistic Regression model with hyperparameter tuning."""
        # Standardize features
        scaler = StandardScaler()
        
        # Logistic Regression model
        model = LogisticRegression(max_iter=1000)
        
        # Create a pipeline
        pipeline = Pipeline([
            ('scaler', scaler),
            ('model', model)
        ])
        
        # Hyperparameter tuning with GridSearchCV
        param_grid = {'model__C': [0.01, 0.1, 1, 10, 100]}
        grid_search = GridSearchCV(pipeline, param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train)
        
        # Best model
        self.best_model = grid_search.best_estimator_
    
    def evaluate_model(self):
        """Evaluate the model and print the results."""
        y_pred = self.best_model.predict(self.X_test)
        
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
    
    def save_model(self):
        """Save the trained model to a file."""
        joblib.dump(self.best_model, self.model_file_path)
    
    def load_and_predict(self, new_data_file_path):
        """Load new data, preprocess it, and make predictions using the saved model."""
        # Load new data
        new_data = pd.read_csv(new_data_file_path)
        
        # Preprocess new data
        X_new = new_data.drop('Outcome', axis=1)
        X_new = X_new.fillna(X_new.median())  # Ensure the new data is handled similarly
        
        # Predict
        y_pred = self.best_model.predict(X_new)
        
        # Export predictions if file path is provided
        if self.predictions_file_path:
            results = pd.DataFrame({'Predictions': y_pred})
            results.to_csv(self.predictions_file_path, index=False)
        
        return y_pred
    
    def cross_validate(self, folds=5):
        """Perform cross-validation and return average accuracy."""
        cv_scores = cross_val_score(self.best_model, self.X, self.y, cv=folds, scoring='accuracy')
        return cv_scores.mean()

# Full Pipeline Execution
if __name__ == "__main__":
    # File paths
    data_file_path = 'data_for_ML/diabetes.csv'
    model_file_path = 'diabetic_model3.pkl'
    predictions_file_path = 'model_predictions.csv'  # For Power BI

    # Initialize and run the pipeline
    pipeline = DataPipeline(data_file_path, model_file_path, predictions_file_path)
    pipeline.load_data()
    pipeline.preprocess_data()
    pipeline.split_data()
    pipeline.train_model()
    pipeline.evaluate_model()
    pipeline.save_model()
    
    # Perform cross-validation
    average_accuracy = pipeline.cross_validate()
    print(f"\nAverage Cross-Validation Accuracy: {average_accuracy:.4f}")
    
    print("Pipeline completed successfully.")

```

# Trigger options to run the Automation Code

To **Trigger** this automation, you can choose from several methods depending on your setup and requirements. Here are a few options:

1. **Scheduled Task (Windows) or Cron Job (Linux/Mac):**
- Windows: Use Task Scheduler to run your script at specified intervals.

    Open Task Scheduler, create a new task, set the trigger (e.g., daily, weekly), and specify the action to run your Python script.

- Linux/Mac: Use cron jobs to schedule tasks.

    Edit the crontab file with crontab -e and add a line to schedule your script, e.g., 0 2 * * * /usr/bin/python3 /path/to/your/script.py to run it daily at 2 AM.

2. **Python Scheduler:**

Use a Python scheduling library like schedule or APScheduler to manage the execution of your script within Python itself.

```python
!pip install schedule

import schedule
import time

def job():
    # Code to run your pipeline
    pipeline = DataPipeline(data_file_path, model_file_path, predictions_file_path)
    pipeline.load_data()
    pipeline.preprocess_data()
    pipeline.split_data()
    pipeline.train_model()
    pipeline.evaluate_model()
    pipeline.save_model()
    average_accuracy = pipeline.cross_validate()
    print(f"\nAverage Cross-Validation Accuracy: {average_accuracy:.4f}")

schedule.every().day.at("02:00").do(job)  # Schedule to run daily at 2 AM

while True:
    schedule.run_pending()
    time.sleep(1)

```

3. **Cloud Services:**

- **AWS Lambda/Azure Functions/Google Cloud Functions:**

  Deploy your script as a serverless function and use cloud scheduling services (e.g., AWS CloudWatch Events, Azure Scheduler) to trigger the function.

- **Google Cloud Scheduler:**
  
  If using Google Cloud, set up a Cloud Scheduler job to trigger a Cloud Function or a Compute Engine instance running your script.

4. **CI/CD Pipelines:**

- Integrate your script into a Continuous Integration/Continuous Deployment (CI/CD) pipeline (e.g., using GitHub Actions, GitLab CI/CD) to trigger the automation as part of your deployment process.

5. **Manual Trigger:**

- Run the script manually by executing it through your terminal or command line interface: python your_script.py.


# Connect to PowerBI/Tableau for Visualization
## Metrics and insights to effectively communicate the performance and outcomes of the model

1. **Model Performance Metrics
Confusion Matrix:**
- A heatmap showing true vs. predicted classifications to visualize model performance.
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['True_0', 'True_1'], columns=['Pred_0', 'Pred_1'])
cm_df.to_csv('confusion_matrix.csv', index=True)
```
- Classification Report Metrics: Precision, recall, F1-score, and accuracy for each class. You can use bar charts or tables for this.
```python
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('classification_report.csv', index=True)
```

2. **Prediction Results**
- Actual vs. Predicted Values: Scatter plots or line charts comparing the actual target values with the predicted values.
- Prediction Distributions: Histograms or density plots showing the distribution of predicted probabilities or classes.
```python
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
predictions_df.to_csv('model_predictions.csv', index=False)
```

3. **Feature Importance**
- Feature Importance Plot: Bar charts displaying the importance of each feature in the model, which helps to understand which variables are most influential.
```python
if hasattr(model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    })
    feature_importance.to_csv('feature_importance.csv', index=False)
```
4. **Cross-Validation Scores**
- Cross-Validation Results: Box plots or line charts showing the performance across different folds during cross-validation to assess model stability and variance.
```python
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
cv_scores_df = pd.DataFrame(cv_scores, columns=['CV_Accuracy'])
cv_scores_df.to_csv('cv_scores.csv', index=False)
```
5. **Class Distribution**
- Target Variable Distribution: Pie charts or bar charts showing the distribution of the target variable classes, which helps to visualize class imbalance.
```python
target_distribution = y_train.value_counts(normalize=True)
target_distribution_df = pd.DataFrame(target_distribution).reset_index()
target_distribution_df.columns = ['Class', 'Proportion']
target_distribution_df.to_csv('target_distribution.csv', index=False)
```
6. **Error Analysis**
- Error Distribution: Histograms or density plots of prediction errors (e.g., residuals) to identify patterns or anomalies.
```python
errors_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Error': y_test - y_pred
})
errors_df.to_csv('prediction_errors.csv', index=False)

```
7. **Model Training Progress**
- Learning Curves: Line charts showing the training and validation performance over epochs (if applicable) to visualize overfitting or underfitting.
8. **Custom Metrics**
- Custom Business Metrics: Any other metrics relevant to your business or application that could provide additional insights from the model’s predictions.


# Finally

## This is the whole project automated
### Including all the above codes in the automation pipeline 

- Confusion Matrix: Generated and saved as a CSV and a heatmap image.
- Classification Report: Saved as a CSV file.
- Prediction Results: Saved as a CSV file.
- Feature Importance: Saved as a CSV file (only if the model supports it).
- Cross-Validation Scores: Saved as a CSV file.
- Class Distribution: Saved as a CSV file.
- Error Analysis: Saved as a CSV file.

**This code is now set up to automatically generate and save all the relevant metrics and insights that you might want to visualize in Power BI/Tableau.**

1. Automated Pipeline with CLASS
```python
# Automated Pipeline with CLASS

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
import joblib

class DataPipeline:
    def __init__(self, data_file_path, model_file_path, predictions_file_path=None, preprocessed_data_path=None):
        """Initialize the pipeline with file paths."""
        self.data_file_path = data_file_path
        self.model_file_path = model_file_path
        self.predictions_file_path = predictions_file_path
        self.preprocessed_data_path = preprocessed_data_path
    
    def load_data(self):
        """Load data from a CSV file."""
        self.df = pd.read_csv(self.data_file_path)
    
    def preprocess_data(self):
        """Preprocess the data."""
        # Handle missing values
        self.df = self.df.fillna(self.df.median())  # Median fill for missing values
        
        # Feature and target separation
        self.X = self.df.drop('Outcome', axis=1)
        self.y = self.df['Outcome']
        
        # One-Hot Encoding for categorical features if necessary
        # Example: self.X = pd.get_dummies(self.X)
        
        # Save preprocessed data if save_path is provided
        if self.preprocessed_data_path:
            preprocessed_df = pd.concat([self.X, self.y], axis=1)
            preprocessed_df.to_csv(self.preprocessed_data_path, index=False)
    
    def split_data(self):
        """Split the data into training and testing sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
    
    def train_model(self):
        """Train a Logistic Regression model with hyperparameter tuning."""
        # Standardize features
        scaler = StandardScaler()
        
        # Logistic Regression model
        model = LogisticRegression(max_iter=1000)
        
        # Create a pipeline
        pipeline = Pipeline([
            ('scaler', scaler),
            ('model', model)
        ])
        
        # Hyperparameter tuning with GridSearchCV
        param_grid = {'model__C': [0.01, 0.1, 1, 10, 100]}
        grid_search = GridSearchCV(pipeline, param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train)
        
        # Best model
        self.best_model = grid_search.best_estimator_
    
    def evaluate_model(self):
        """Evaluate the model and save the metrics."""
        y_pred = self.best_model.predict(self.X_test)
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=['True_0', 'True_1'], columns=['Pred_0', 'Pred_1'])
        cm_df.to_csv('confusion_matrix.csv', index=True)
        plt.figure(figsize=(6,6))
        plt.title("Confusion Matrix")
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.savefig('confusion_matrix_heatmap.png')
        
        # Classification Report
        report = classification_report(self.y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv('classification_report.csv', index=True)
        
        # Prediction Results
        predictions_df = pd.DataFrame({'Actual': self.y_test, 'Predicted': y_pred})
        predictions_df.to_csv('model_predictions.csv', index=False)
        
        # Feature Importance (if applicable)
        if hasattr(self.best_model.named_steps['model'], 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': self.X_train.columns,
                'Importance': self.best_model.named_steps['model'].feature_importances_
            })
            feature_importance.to_csv('feature_importance.csv', index=False)
        
        # Cross-Validation Scores
        cv_scores = cross_val_score(self.best_model, self.X, self.y, cv=5, scoring='accuracy')
        cv_scores_df = pd.DataFrame(cv_scores, columns=['CV_Accuracy'])
        cv_scores_df.to_csv('cv_scores.csv', index=False)
        
        # Class Distribution
        target_distribution = self.y_train.value_counts(normalize=True)
        target_distribution_df = pd.DataFrame(target_distribution).reset_index()
        target_distribution_df.columns = ['Class', 'Proportion']
        target_distribution_df.to_csv('target_distribution.csv', index=False)
        
        # Error Analysis
        errors_df = pd.DataFrame({'Actual': self.y_test, 'Predicted': y_pred, 'Error': self.y_test - y_pred})
        errors_df.to_csv('prediction_errors.csv', index=False)
    
    def save_model(self):
        """Save the trained model to a file."""
        joblib.dump(self.best_model, self.model_file_path)
    
    def load_and_predict(self, new_data_file_path):
        """Load new data, preprocess it, and make predictions using the saved model."""
        # Load new data
        new_data = pd.read_csv(new_data_file_path)
        
        # Preprocess new data
        X_new = new_data.drop('Outcome', axis=1)
        X_new = X_new.fillna(X_new.median())  # Ensure the new data is handled similarly
        
        # Predict
        y_pred = self.best_model.predict(X_new)
        
        # Export predictions if file path is provided
        if self.predictions_file_path:
            results = pd.DataFrame({'Predictions': y_pred})
            results.to_csv(self.predictions_file_path, index=False)
        
        return y_pred
    
    def cross_validate(self, folds=5):
        """Perform cross-validation and return average accuracy."""
        cv_scores = cross_val_score(self.best_model, self.X, self.y, cv=folds, scoring='accuracy')
        return cv_scores.mean()

# Full Pipeline Execution
if __name__ == "__main__":
    # File paths
    data_file_path = 'data_for_ML/diabetes.csv'
    model_file_path = 'diabetic_model3.pkl'
    predictions_file_path = 'model_predictions.csv'  # For Power BI
    preprocessed_data_path = 'preprocessed_data.csv'  # Path to save preprocessed data

    # Initialize and run the pipeline
    pipeline = DataPipeline(data_file_path, model_file_path, predictions_file_path, preprocessed_data_path)
    pipeline.load_data()
    pipeline.preprocess_data()
    pipeline.split_data()
    pipeline.train_model()
    pipeline.evaluate_model()
    pipeline.save_model()
    
    # Perform cross-validation
    average_accuracy = pipeline.cross_validate()
    print(f"\nAverage Cross-Validation Accuracy: {average_accuracy:.4f}")
    
    print("Pipeline completed successfully.")
```

2. Or Automated Pipeline with function
```python
# Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
import joblib

# Step 1: Data Loading
def load_data(file_path):
    return pd.read_csv(file_path)

# Step 2: Data Preprocessing
def preprocess_data(df, save_path=None):
    # Handle missing values if any
    df = df.fillna(method='ffill')  # Forward fill as an example
    
    # Feature and target separation
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # One-Hot Encoding for categorical features if necessary
    # Example: X = pd.get_dummies(X)
    
    # Save preprocessed data if save_path is provided
    if save_path:
        preprocessed_df = pd.concat([X, y], axis=1)
        preprocessed_df.to_csv(save_path, index=False)
    
    return X, y

# Step 3: Model Training
def train_model(X_train, y_train):
    # Standardize features
    scaler = StandardScaler()
    
    # Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    
    # Create a pipeline
    pipeline = Pipeline([
        ('scaler', scaler),
        ('model', model)
    ])
    
    # Hyperparameter tuning with GridSearchCV
    param_grid = {'model__C': [0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    return best_model

# Step 4: Model Evaluation
def evaluate_model(model, X_test, y_test, X, y):
    y_pred = model.predict(X_test)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=['True_0', 'True_1'], columns=['Pred_0', 'Pred_1'])
    cm_df.to_csv('confusion_matrix.csv', index=True)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig('confusion_matrix_heatmap.png')
    
    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('classification_report.csv', index=True)
    
    # Prediction Results
    predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    predictions_df.to_csv('model_predictions.csv', index=False)
    
    # Feature Importance (if applicable)
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': model.named_steps['model'].feature_importances_
        })
        feature_importance.to_csv('feature_importance.csv', index=False)
    
    # Cross-Validation Scores
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_scores_df = pd.DataFrame(cv_scores, columns=['CV_Accuracy'])
    cv_scores_df.to_csv('cv_scores.csv', index=False)
    
    # Print the average cross-validation accuracy
    average_accuracy = cv_scores.mean()
    print(f"\nAverage Cross-Validation Accuracy: {average_accuracy:.4f}")
    
    # Class Distribution
    target_distribution = y_test.value_counts(normalize=True)
    target_distribution_df = pd.DataFrame(target_distribution).reset_index()
    target_distribution_df.columns = ['Class', 'Proportion']
    target_distribution_df.to_csv('target_distribution.csv', index=False)
    
    # Error Analysis
    errors_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Error': y_test - y_pred})
    errors_df.to_csv('prediction_errors.csv', index=False)

# Step 5: Save the Model
def save_model(model, file_path):
    joblib.dump(model, file_path)

# Step 6: Load and Predict (For future use)
def load_and_predict(file_path, model_path):
    # Load new data and model
    new_data = pd.read_csv(file_path)
    model = joblib.load(model_path)
    
    # Preprocess and predict
    X_new, _ = preprocess_data(new_data)
    y_pred = model.predict(X_new)
    return y_pred

# Full Pipeline Execution
if __name__ == "__main__":
    # File paths
    data_file_path = 'data_for_ML/diabetes.csv'
    model_file_path = 'diabetic_model.pkl'
    preprocessed_data_path = 'preprocessed_data.csv'  # Path to save preprocessed data
    
    # Load data
    df = load_data(data_file_path)
    
    # Preprocess data
    X, y = preprocess_data(df, save_path=preprocessed_data_path)
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test, X, y)
    
    # Save model
    save_model(model, model_file_path)
    
    print("Pipeline completed successfully.")
```





























