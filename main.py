import json
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import simpledialog

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, roc_auc_score, \
    roc_curve, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from statsmodels.api import qqplot
from xgboost import XGBClassifier, XGBRegressor

# Create the main window
root = tk.Tk()
root.title("Expert System")
root.geometry("250x250")

Encode = LabelEncoder()
# Function to handle the "Open" button
def onOpen():
    '''
    Ask the user to Select a CSV file, Drop unwanted Columns, And select Target
    :return:
    feature: Pandas DataFrame, Features to Train on
    target: Pandas Series, Target to predict
    '''
    try:
        global fileName
        fileName = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
        if fileName:
            dataDF = pd.read_csv(fileName)
            # Ask the user to select columns to drop
            colsToDrop = simpledialog.askstring('Input', 'Enter the columns to drop (separated by commas):', parent=root)
            if colsToDrop:
                # Split the input string into a list of column names
                colsToDrop = colsToDrop.split(',')
                # Drop the selected columns from the DataFrame
                dataDF = dataDF.drop(colsToDrop, axis=1)

            outcomeColumn = simpledialog.askstring('Input', 'Enter the outcome column:', parent=root)
            if outcomeColumn:
                features = dataDF.drop(outcomeColumn, axis=1)
                target = dataDF[outcomeColumn]

        return (features,target)
    except:
        print('Something Went Wrong')


def PreprocessData(Reg=True):
    """
    Preprocess The Data
    :return:
    xTrain: Pandas DataFrame, training data
    xTest: Pandas DataFrame, test data
    yTrain: Pandas Series, training label
    yTest: Pandas Series, test label
    """
    features, target = onOpen()  # Get features and target

    if Reg:
        # split Data
        xTrain, xTest, yTrain, yTest = train_test_split(features, target, shuffle=True)

    else:
        #encode and split data
        target=Encode.fit_transform(target)
        xTrain, xTest, yTrain, yTest = train_test_split(features, target, shuffle=True,stratify=target)

    # split data based on dtype and impute
    types = set([str(x) for x in xTrain.dtypes])
    if 'datetime64[ns]' in types:
        xTrainDate = xTrain.select_dtypes(include='datetime64[ns]').fillna(method='ffill')
        xTestDate = xTest.select_dtypes(include='datetime64[ns]').fillna(method='ffill')

        xTrain.drop(xTrainDate.columns, axis=1,inplace=True)
        xTrain = pd.concat([xTrain, xTrainDate], axis=1)
        xTest.drop(xTestDate.columns, axis=1,inplace=True)
        xTest = pd.concat([xTest, xTestDate], axis=1)

    if 'int64' in types or 'float64' in types:
        # Select columns where dtype is numerical
        numCols = xTrain.select_dtypes(include=['int64', 'float64']).columns

        numericTransformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())])


    if 'object' in types:
        # Select column where dtype is Object (string)
        catCols = xTrain.select_dtypes(include=['object']).columns
        categoricalTransformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    #Build Preprocessor
    if 'categoricalTransformer' in locals() and 'numericTransformer' in locals():
        preprocessor = ColumnTransformer(
            transformers=[('num', numericTransformer, numCols),('cat', categoricalTransformer, catCols)])

    elif 'categoricalTransformer' in locals():
        preprocessor = ColumnTransformer(
            transformers=[('cat', categoricalTransformer, catCols)])

    else:
        preprocessor = ColumnTransformer(
            transformers=[('num', numericTransformer, numCols)])

    return (xTrain, xTest, yTrain, yTest, preprocessor,features)


def BuildRegressor():
    '''
    Build and Train Optimal Regression Model
    '''

    #Get preprocessed Data
    xTrain, xTest, yTrain, yTest, preprocessor, features = PreprocessData()

    params={
        'xgb__n_estimators': [100, 500, 1000, 1500],
        'xgb__learning_rate': [0.01, 0.05, 0.1],
        'xgb__max_depth': [3, 4, 5, 6],
        'xgb__colsample_bytree': [0.3, 0.5, 0.7],
        'xgb__gamma': [0, 0.1, 0.2],
        'lasso__alpha': [x / 10 for x in range(1, 10, 1)],
        'lasso__fit_intercept': [True, False],
        'lasso__precompute': [True, False],
        'lasso__max_iter': [1000, 1100, 1200, 1300, 1400],
        'lasso__tol': [0.0001, 0.001, 0.01],
        'lasso__selection': ['cyclic', 'random'],
        'knr__n_neighbors': [3, 5, 7, 9, 11],
        'knr__weights': ['uniform', 'distance'],
        'knr__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'knr__leaf_size': [10, 20, 30, 40, 50],
        'knr__p': [1, 2]
    }

    #Pipeline
    regressors=[('xgb',XGBRegressor()),('lasso',Lasso()),('knr',KNeighborsRegressor())]
    vote=VotingRegressor(estimators=regressors)
    RandomCV=RandomizedSearchCV(vote, params, cv=5, n_jobs=-1)

    PipeReg=Pipeline([('preprocessor', preprocessor),('model',RandomCV)])
    PipeReg.fit(xTrain,yTrain)

    #Remove old buttons
    RegressionButton.pack_forget()
    ClassificationButton.pack_forget()
    loadModelButton.pack_forget()

    def PredVAct():
        #Plot Predictions vs Actual
        pred=PipeReg.predict(xTest)
        plt.scatter(yTest, pred, color='red', label='Predicted')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted')
        diagonal = np.linspace(min(yTest.min(), pred.min()), max(yTest.max(), pred.max()), 100)
        plt.plot(diagonal, diagonal, 'k--',label='Perfect Model')
        plt.text(0.95, 0.01, 'R^2 Score: {:.2f}'.format(PipeReg.score(xTest,yTest)),
                 horizontalalignment='right', verticalalignment='bottom', transform=plt.gca().transAxes)
        plt.legend(loc='upper left')
        plt.show()

    PredVActButton = tk.Button(root, text='Prediction Vs Actual Graph', command=PredVAct)
    PredVActButton.pack()

    def QQPlot():
        #Plot QQPlot
        resid=yTest-(PipeReg.predict(xTest))
        qqplot(resid,fit=True,line='45')
        plt.title('QQ Plot')
        plt.show()

    qqButton = tk.Button(root, text='QQ Plot', command=QQPlot)
    qqButton.pack()

    def FeatureImp():
        #Plot Feature importance
        featureNames=PipeReg.named_steps['preprocessor'].get_feature_names_out()
        plt.figure(figsize=(12, 6))
        plt.barh(featureNames, RandomCV.best_estimator_.estimators_[0].feature_importances_)
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.title('Feature Importance')
        plt.show()

    featButton = tk.Button(root, text='Feature Importance Plot', command=FeatureImp)
    featButton.pack()

    def onPredict():
        #Predict new data
        inputData = {}
        for column in features.columns:
            value = simpledialog.askstring('Input', f'Enter value for {column}:', parent=root)
            inputData[column] = value
        inputData = pd.DataFrame([inputData])

        for n,type in enumerate([str(x) for x in features.dtypes]):
            try:
                inputData.iloc[:,n]=inputData.iloc[:,n].astype(type)
            except:
                print(f"Failed at col {inputData.iloc[:,n].column}")
                continue

        prediction = PipeReg.predict(inputData)
        tk.messagebox.showinfo('Prediction', f'Prediction: {prediction[0]}')

    predictButton = tk.Button(root, text='Predict', command=onPredict)
    predictButton.pack()

    def onSaveClassifier():
        #save Classifier
        dataclf = {'clf': PipeReg, 'feature_names': features,'cat': False}
        joblib.dump(dataclf, f'{fileName.split("/")[-1]}-Regression.joblib')
        print(f'Classifier saved to {fileName.split("/")[-1]}-Regression.joblib')

    saveClassifierButton = tk.Button(root, text='Save Classifier', command=onSaveClassifier)
    saveClassifierButton.pack()

    def onSaveDetails():
        #Save Metrics about models
        details = {
            'R^2': PipeReg.score(xTest, yTest),
            'MSE':  mean_squared_error(yTest, PipeReg.predict(xTest)),
            'RMSE': np.sqrt(mean_squared_error(yTest, PipeReg.predict(xTest))),
            'MAE':  mean_absolute_error(yTest, PipeReg.predict(xTest))
        }
        with open(f'{fileName.split("/")[-1]}-Regression.txt', 'w') as file:
            file.write(json.dumps(details))
        print(f'Details saved to {fileName.split("/")[-1]}-Regression.txt')

    saveDetailsButton = tk.Button(root, text='Save Details', command=onSaveDetails)
    saveDetailsButton.pack()

def BuildClassifier():
    '''
    Build optimal Classifier
    '''
    #Preprocessed Data
    xTrain, xTest, yTrain, yTest, preprocessor, features=PreprocessData(Reg=False)

    params={
        'xgb__booster': ["gbtree", "dart"],
        'xgb__tree_method':["auto", "exact", "approx", "hist"],
        'xgb__learning_rate': [0.01, 0.05, 0.1],
        'xgb__max_depth': [3, 4, 5, 6],
        'xgb__colsample_bytree': [0.3, 0.5, 0.7],
        'xgb__gamma': [0, 0.1, 0.2],
        'mnb__alpha': [0.1, 0.5, 1.0, 2.0],
        'mnb__fit_prior': [True, False],
        'knn__n_neighbors': [3, 5, 7, 9, 11],
        'knn__weights': ['uniform', 'distance'],
        'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'knn__leaf_size': [10, 20, 30, 40, 50],
        'knn__p': [1, 2]

    }

    #Pipeline
    classifiers = [('xgb', XGBClassifier()), ('mnb', MultinomialNB()), ('knn', KNeighborsClassifier())]
    vote = VotingClassifier(estimators=classifiers,voting='soft')
    RandomCV = RandomizedSearchCV(vote, params, cv=5, n_jobs=-1)

    PipeClas = Pipeline([('preprocessor', preprocessor), ('model', RandomCV)])
    PipeClas.fit(xTrain, yTrain)
    print(PipeClas.score(xTest, yTest))

    RegressionButton.pack_forget()
    ClassificationButton.pack_forget()
    loadModelButton.pack_forget()

    def ConfMat():
        #Plot confusion matrix
        ax=plt.subplot()
        preds=PipeClas.predict(xTest)
        conf=confusion_matrix(yTest,preds)
        g=ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=Encode.inverse_transform(PipeClas.classes_))
        g.plot(ax=ax, colorbar=False)
        plt.title("Confusion Matrix")
        plt.show()

    ConfMatButton = tk.Button(root, text='Confusion Matrix', command=ConfMat)
    ConfMatButton.pack()

    if len(set(yTest)) == 2:
        #Check for binary classification to plot ROC curve
        def ROC():
            pred = PipeClas.predict_proba(xTest)[:, 1]
            fpr, tpr, thresholds = roc_curve(yTest, pred)

            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.plot(fpr, tpr, label='Model')

            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")

            plt.text(0.95, 0.01, 'ROC AUC: {:.2f}'.format(roc_auc_score(yTest,PipeClas.predict(xTest))),
                     horizontalalignment='right', verticalalignment='bottom', transform=plt.gca().transAxes)

            plt.legend(loc='upper left')
            plt.show()

        ROCButton = tk.Button(root, text='ROC Curve', command=ROC)
        ROCButton.pack()

    def FeatureImp():
        #Plot feature importance
        featureNames=PipeClas.named_steps['preprocessor'].get_feature_names_out()
        plt.figure(figsize=(12, 6))
        plt.barh(featureNames, RandomCV.best_estimator_.estimators_[0].feature_importances_)
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.title('Feature Importance')
        plt.show()

    featButton = tk.Button(root, text='Feature Importance Plot', command=FeatureImp)
    featButton.pack()

    def onPredict():
        #Predict new data
        inputData = {}
        for column in features.columns:
            value = simpledialog.askstring('Input', f'Enter value for {column}:', parent=root)
            inputData[column] = value
        inputData = pd.DataFrame([inputData])

        for n,type in enumerate([str(x) for x in features.dtypes]):
            try:
                inputData.iloc[:,n]=inputData.iloc[:,n].astype(type)
            except:
                print(f"Failed at col {inputData.iloc[:,n].column}")
                continue

        prediction = PipeClas.predict(inputData)
        tk.messagebox.showinfo('Prediction', f'Prediction: {Encode.inverse_transform(prediction)}')

    predictButton = tk.Button(root, text='Predict', command=onPredict)
    predictButton.pack()

    def onSaveClassifier():
        #Save classifier
        dataclf = {'clf': PipeClas, 'feature_names': features,'cat': True,'LabelEncoder':Encode}
        joblib.dump(dataclf, f'{fileName.split("/")[-1]}-Classification.joblib')
        print(f'Classifier saved to {fileName.split("/")[-1]}-Classification.joblib')

    saveClassifierButton = tk.Button(root, text='Save Classifier', command=onSaveClassifier)
    saveClassifierButton.pack()

    def onSaveDetails():
        #Save metrics about model
        pred=PipeClas.predict(xTest)
        details = {
            'Accuracy': PipeClas.score(xTest,yTest),
            'Percision': precision_score(yTest,pred),
            'Recall': recall_score(yTest,pred),
            'F1-Score': f1_score(yTest,pred),
            'ROC AUC Score': roc_auc_score(yTest,pred)
        }
        with open(f'{fileName.split("/")[-1]}-Classification.txt', 'w') as file:
            file.write(json.dumps(details))
        print(f'Details saved to {fileName.split("/")[-1]}-Classification.txt')

    saveDetailsButton = tk.Button(root, text='Save Details', command=onSaveDetails)
    saveDetailsButton.pack()

def loadModel():
    '''
    Load Pre-made model to predict
    '''
    fileName = filedialog.askopenfilename(filetypes=[('Joblib Files', '*.joblib')])
    if fileName:
        data = joblib.load(fileName)
        clf = data['clf']
        featureNames = data['feature_names']
        cat=data['cat']
        print('Classifier loaded from', fileName)

        inputData = {}
        for column in featureNames.columns:
            value = simpledialog.askstring('Input', f'Enter value for {column}:', parent=root)
            inputData[column] = value
        inputData = pd.DataFrame([inputData])

        for n, type in enumerate([str(x) for x in featureNames.dtypes]):
            try:
                inputData.iloc[:, n] = inputData.iloc[:, n].astype(type)
            except:
                print(f"Failed at col {inputData.iloc[:, n].column}")
                continue
        if cat:
            encode=data['LabelEncoder']
            prediction = clf.predict(inputData)
            tk.messagebox.showinfo('Prediction', f'Prediction: {encode.inverse_transform(prediction)}')
        else:
            prediction = clf.predict(inputData)
            tk.messagebox.showinfo('Prediction', f'Prediction: {prediction[0]}')

# Create the "Regression" button
RegressionButton = tk.Button(root, text='Train Regressor', command=BuildRegressor)
RegressionButton.pack()

ClassificationButton = tk.Button(root, text='Train Classifier', command=BuildClassifier)
ClassificationButton.pack()

# Create the "Load Model" button
loadModelButton = tk.Button(root, text='Predict From Pre-made Model', command=loadModel)
loadModelButton.pack()

# Run the main loop
root.mainloop()
