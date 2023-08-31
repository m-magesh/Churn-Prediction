import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

#Load and preprocess the dataset
data = pd.read_csv("churn_dataset.csv")
import numpy as np

#Perform data preprocessing steps as described above
pd.options.display.max_columns = None
data['gender'] = np.where(data['gender']=='Male', 1, 0)
data['Partner'] = np.where(data['Partner']=='Yes', 1, 0)
data['Dependents'] = np.where(data['Dependents']=='Yes', 1, 0)
data['PhoneService'] = np.where(data['PhoneService']=='Yes', 1, 0)
data['OnlineSecurity'] = np.where(data['OnlineSecurity']=='Yes', 1, 0)
data['OnlineBackup'] = np.where(data['OnlineBackup']=='Yes', 1, 0)
data['DeviceProtection'] = np.where(data['DeviceProtection']=='Yes', 1, 0)
data['TechSupport'] = np.where(data['TechSupport']=='Yes', 1, 0)
data['StreamingTV'] = np.where(data['StreamingTV']=='Yes', 1, 0)
data['StreamingMovies'] = np.where(data['StreamingMovies']=='Yes', 1, 0)
data['PaperlessBilling'] = np.where(data['PaperlessBilling']=='Yes', 1, 0)
data['Churn'] = np.where(data['Churn']=='Yes', 1, 0)
numeric_var = {'MultipleLines': {'No phone service':0, 'No':1, 'Yes':2}}
data = data.replace(numeric_var)
numeric_var = {'InternetService': {'DSL':1, 'No':0, 'Fiber optic':2}}
data = data.replace(numeric_var)
numeric_var = {'Contract': {'Month-to-month':0, 'One year':1, 'Two year':2}}
data = data.replace(numeric_var)
numeric_var = {'PaymentMethod': {'Electronic check':0, 'Mailed check':1, 'Bank transfer (automatic)':2, 'Credit card (automatic)': 3}}
data = data.replace(numeric_var)
data['TotalCharges'] = np.where(data['TotalCharges'] == ' ', 0, data['TotalCharges'])
data = data.astype({'TotalCharges': 'float64'})
X = data.drop(["Churn","customerID"], axis=1)
y = data["Churn"]

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Initialize and train the XGBoost model
model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
model.fit(X_train, y_train)

#Make predictions on the test set
y_pred = model.predict(X_test)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"ROC-AUC: {roc_auc}")
print("Classification Report:")
print(classification_rep