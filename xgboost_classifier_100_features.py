import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from xgboost import XGBClassifier

data = pd.read_csv('~/Desktop/project_data_new/embedding_768_TCGA_COAD.csv')
data.index = data['PatientID']
# drop last 2 columns
data = data.drop(data.columns[-2:], axis=1)
data_target = pd.read_csv('~/Desktop/project_data_new/target_768_avg_expanded.csv')
data_target.index = data_target['Unnamed: 0']
data_target = data_target.drop(['Unnamed: 0'], axis = 1)
# only keep the columns with category in the name
data_target = data_target.loc[:, data_target.columns.str.contains('category')]
data = data[data.index.isin(data_target.index)]

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score
from scipy.stats import randint, uniform
from sklearn.preprocessing import LabelEncoder

# Initialize the XGBoost Classifier
xgb_clf = xgb.XGBClassifier(objective='multi:softmax', random_state=42)

# Extract the feature values from data
X = data.values

# Set the chunk size (number of target variables per part)
chunk_size = 40

# Loop through target columns in chunks
for chunk_start in range(200, 241, chunk_size):
    # Get the current chunk of target columns
    chunk_end = min(chunk_start + chunk_size, 241)
    target_chunk = data_target.columns[chunk_start:chunk_end]
    
    # Initialize list to store results for the current chunk
    chunk_results = []

    # Open a file to write classification reports for the current chunk
    report_file_path = f"/home/qiuaodon/Desktop/CRC_image/Best_features_XGboost_results/Classification_Reports_100features_XGB_part_{chunk_start}_{chunk_end}.txt"
    with open(report_file_path, "w") as report_file:
        # Loop through each target column in the current chunk
        for target_col in target_chunk:
            # Extract the target column for the current category
            Y = data_target[target_col].values

            # Encode the labels if necessary
            label_encoder = LabelEncoder()
            Y = label_encoder.fit_transform(Y)
            
            # Select top 100 features using RFE
            xgb = XGBClassifier()
            xgb.fit(X, Y)

            # Get feature importances and select the top 100 indices
            importances = xgb.feature_importances_
            top_100_indices = np.argsort(importances)[-100:]

            # Select top 100 features
            X_selected = X[:, top_100_indices]


            # Split the data into training and testing sets
            X_train, X_test, Y_train, Y_test = train_test_split(
                X_selected, Y, test_size=0.2, random_state=42, stratify=Y
            )
            
            # Define the hyperparameter distribution for RandomizedSearchCV
            param_dist = {
                # Most important hyperparameters
                'max_depth': randint(6, 13),
                'min_child_weight': randint(1, 10),
                'subsample': uniform(0.9, 0.1),          # Higher is usually better
                'colsample_bytree': uniform(0.9, 0.1),   # Range from 0.5 to 1.0
                'learning_rate': uniform(0.01, 0.05),    # Starting from 0.01
                # Regularization hyperparameters
                'n_estimators': randint(1000, 1500),
                'gamma': uniform(2, 3),
                # 'reg_alpha': uniform(0, 1),
                # 'reg_lambda': uniform(1, 5),
            }
            
            # Perform Randomized Search with cross-validation
            random_search = RandomizedSearchCV(
                estimator=xgb_clf,
                param_distributions=param_dist,
                n_iter=160, 
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
            random_search.fit(X_train, Y_train)
            
            # Get the best model from Randomized Search
            best_xgb = random_search.best_estimator_
            best_params = random_search.best_params_
            
            # Fit the model on the training data for evaluation
            best_xgb.fit(X_train, Y_train)
            
            # Evaluate on the training set
            Y_train_pred = best_xgb.predict(X_train)
            train_report_dict = classification_report(Y_train, Y_train_pred, output_dict=True)
            train_report = classification_report(Y_train, Y_train_pred)
            
            # Evaluate on the test set
            Y_test_pred = best_xgb.predict(X_test)
            test_report_dict = classification_report(Y_test, Y_test_pred, output_dict=True)
            test_report = classification_report(Y_test, Y_test_pred)
            
            # Write classification reports to the file
            report_file.write(f"Classification Report for {target_col} (Train Set):\n")
            report_file.write(train_report)
            report_file.write("\n")
            report_file.write(f"Classification Report for {target_col} (Test Set):\n")
            report_file.write(test_report)
            report_file.write("\n\n" + "="*80 + "\n\n")
            
            # Calculate and store test set metrics
            test_precision = precision_score(Y_test, Y_test_pred, average='weighted')
            test_recall = recall_score(Y_test, Y_test_pred, average='weighted')
            test_accuracy = accuracy_score(Y_test, Y_test_pred)
            
            # Extract test set class-specific metrics for class '1'
            test_class_1_metrics = test_report_dict.get('2', {"precision": None, "recall": None, "f1-score": None})
            
            # Calculate and store training set metrics
            train_precision = precision_score(Y_train, Y_train_pred, average='weighted')
            train_recall = recall_score(Y_train, Y_train_pred, average='weighted')
            train_accuracy = accuracy_score(Y_train, Y_train_pred)
            
            # Extract training set class-specific metrics for class '1'
            train_class_1_metrics = train_report_dict.get('2', {"precision": None, "recall": None, "f1-score": None})
            
            # Append results to chunk_results
            chunk_results.append({
                "Target Variable": target_col,
                # Test set metrics
                "Test Precision": test_precision,
                "Test Recall": test_recall,
                "Test Accuracy": test_accuracy,
                "Test Class 1 Precision": test_class_1_metrics["precision"],
                "Test Class 1 Recall": test_class_1_metrics["recall"],
                "Test Class 1 F1-Score": test_class_1_metrics["f1-score"],
                # Training set metrics
                "Train Accuracy": train_accuracy,
                "Train Class 1 F1-Score": train_class_1_metrics["f1-score"],
                # Best hyperparameters
                "Best Hyperparameters": best_params
            })
    
    # Save the results for the current chunk to an Excel file
    results_df = pd.DataFrame(chunk_results)
    results_df.to_excel(
        f"/home/qiuaodon/Desktop/CRC_image/Best_features_XGboost_results/Precision_Recall_Accuracy_100features_XGB_part_{chunk_start}_{chunk_end}.xlsx",
        index=False
    )
