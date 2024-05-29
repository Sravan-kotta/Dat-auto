import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import joblib 
import pickle


def train_model(file_path, model_type, test_size=0.2, random_state=42):

    
    data = pd.read_csv(file_path,encoding='ISO-8859-1')
    
    
    X = data.iloc[:, :-1]   #for columns 
    y = data.iloc[:, -1]
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    
    if model_type == 'SVM':
        model = SVC()
    elif model_type == 'LogisticRegression':
        model = LogisticRegression()
    elif model_type == 'DecisionTree':
        model = DecisionTreeClassifier()
    elif model_type == 'NaiveBayes':
        model = GaussianNB()
    elif model_type == 'LinearRegression':
        model = LinearRegression()
    else:
        raise ValueError("Invalid model type specified. Choose from 'SVM', 'LogisticRegression', 'DecisionTree', 'NaiveBayes', 'LinearRegression'.")
    
    model.fit(X_train, y_train)
    

    model_filename = f"{model_type}_model.pkl"
    pickle.dump(model, model_filename)
    print(f"Model training complete. The trained model has been saved as '{model_filename}'.")

file_path = input("Enter file path")
model = input('''Enter the model in which you want to train your data
                    SVM for Support vector machine
                    LogisticRgression 
                    DecisionTree
                    NaiveBayes
                    LinearRegression''')
train_model(file_path, model)
  
