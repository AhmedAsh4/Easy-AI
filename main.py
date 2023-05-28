import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import simpledialog
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image, ImageTk
import io
import pydotplus
import xml.etree.ElementTree as ET
import joblib
import matplotlib.pyplot as plt

# Create the main window
root = tk.Tk()
root.title("Expert System")
root.geometry("250x250")


# Function to handle the "Open" button
def onOpen():
    fileName = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
    if fileName:
        datadf = pd.read_csv(fileName)
        outcomeColumn = simpledialog.askstring('Input', 'Enter the outcome column:', parent=root)
        if outcomeColumn:
            features = datadf.drop(outcomeColumn, axis=1)
            target = datadf[outcomeColumn]
            # Delete OpenCSV and Load Classifier Buttons
            openButton.pack_forget()
            loadClassifierButton.pack_forget()
            # Create a StringVar to hold the value of the selected option
            classifierVar = tk.StringVar(value='dtc')

            # Create a label for the radio button
            classifierLabel = tk.Label(root, text='Choose a classifier:')
            classifierLabel.pack()

            # Create a radio button for each option
            dtcRadio = tk.Radiobutton(root, text='Decision Tree', variable=classifierVar, value='dtc')
            dtcRadio.pack()
            nbRadio = tk.Radiobutton(root, text='Naive Bayes', variable=classifierVar, value='nb')
            nbRadio.pack()
            knnRadio = tk.Radiobutton(root, text='KNN', variable=classifierVar, value='knn')
            knnRadio.pack()

            # Wait for the user to choose an option
            root.wait_variable(classifierVar)

            # Remove the radio button and label from the GUI
            classifierLabel.pack_forget()
            dtcRadio.pack_forget()
            nbRadio.pack_forget()
            knnRadio.pack_forget()

            # Get the value of the selected option
            classifier = classifierVar.get()

            # Train the selected classifier
            if classifier == 'dtc':
                clf = DecisionTreeClassifier()
                clf.fit(features, target)
                print('Decision Tree trained!')

                # Visualize the decision tree
                dot_data = export_graphviz(clf, out_file=None, feature_names=list(features.columns),
                                           class_names=[str(x) for x in clf.classes_], filled=True)
                graph = pydotplus.graph_from_dot_data(dot_data)
                image_data = graph.create_png()
                image = Image.open(io.BytesIO(image_data))
                photo = ImageTk.PhotoImage(image)
                image_label.image = photo
                print("Done")

                # function to handle the "Display Tree" button
                def DisplayTree():
                    if image_label.image:
                        image = Image.open(io.BytesIO(image_data))
                        image.show()

                displayTreeButton = tk.Button(root, text='Display Tree', command=DisplayTree)
                displayTreeButton.pack()

                # Function to handle the "Save Rules" button
                def onSaveRules():
                    if clf:
                        rules = export_graphviz(clf, out_file=None, feature_names=list(features.columns),
                                                class_names=[str(x) for x in clf.classes_], filled=True)
                        root = ET.Element('rules')
                        root.text = rules
                        tree = ET.ElementTree(root)
                        tree.write(f'{fileName.split("/")[-1]}-rulesDT.xml')
                        print(f'Rules saved to {fileName.split("/")[-1]}-rulesDT.xml')

                # Create the "Save Rules" button
                saveRulesButton = tk.Button(root, text='Save Rules', command=onSaveRules)
                saveRulesButton.pack()

                # Function to handle the "Predict" button
                def onPredict():
                    if clf:
                        inputData = {}
                        for column in features.columns:
                            value = simpledialog.askstring('Input', f'Enter value for {column}:', parent=root)
                            inputData[column] = value
                        inputDf = pd.DataFrame([inputData])
                        prediction = clf.predict(inputDf)
                        tk.messagebox.showinfo('Prediction', f'Prediction: {prediction[0]}')

                # Create the "Predict" button
                predictButton = tk.Button(root, text='Predict', command=onPredict)
                predictButton.pack()

                # Function to handle the "Save Classifier" button
                def onSaveClassifier():
                    if clf:
                        dataclf = {'clf': clf, 'feature_names': features}
                        joblib.dump(dataclf, f'{fileName.split("/")[-1]}-classifierDT.joblib')
                        print(f'Classifier saved to {fileName.split("/")[-1]}-classifierDT.joblib')

                # Create the "Save Classifier" button
                saveClassifierButton = tk.Button(root, text='Save Classifier', command=onSaveClassifier)
                saveClassifierButton.pack()

            elif classifier == 'nb':
                clf = GaussianNB()
                clf.fit(features, target)
                print('Naive Bayes trained!')

                def visualizeNb():
                    # Get the number of features
                    nFeatures = features.shape[1]
                    featureNames = datadf.columns

                    # Create a figure with one subplot for each feature
                    fig, axes = plt.subplots(nrows=1, ncols=nFeatures)

                    # Plot the distribution of each feature for each class
                    for i in range(nFeatures):
                        # Get the data for the current feature
                        data = features.iloc[:, i]

                        # Plot the distribution for each class
                        for j in range(clf.classes_.shape[0]):
                            # Get the data for the current class
                            classData = data[target == clf.classes_[j]]

                            # Plot a histogram of the data
                            axes[i].hist(classData, alpha=0.5, label=str(clf.classes_[j]))

                        # Set the title and labels
                        axes[i].set_title(featureNames[i])
                        axes[i].set_xlabel('Value')
                        axes[0].set_ylabel('Count')

                    # Add a legend
                    plt.legend()

                def displayNb():
                    visualizeNb()
                    plt.show()

                visualizeNbButton = tk.Button(root, text='Show Distribution Graph', command=displayNb)
                visualizeNbButton.pack()

                # Function to handle the "Save Rules" button
                def onSaveRules():
                    if clf:
                        # Create the root element
                        root = ET.Element('NaiveBayes')

                        # Create an element for the class priors
                        priorsElement = ET.SubElement(root, 'Priors')
                        for i, prior in enumerate(clf.class_prior_):
                            priorElement = ET.SubElement(priorsElement, 'Prior')
                            priorElement.set('class', str(clf.classes_[i]))
                            priorElement.set('value', str(prior))

                        # Create an element for the means
                        meansElement = ET.SubElement(root, 'Means')
                        for i in range(clf.theta_.shape[0]):
                            meanElement = ET.SubElement(meansElement, 'Mean')
                            meanElement.set('class', str(clf.classes_[i]))
                            for j in range(clf.theta_.shape[1]):
                                featureElement = ET.SubElement(meanElement, 'Feature')
                                featureElement.set('index', str(j))
                                featureElement.set('value', str(clf.theta_[i][j]))

                        # Create an element for the variances
                        variancesElement = ET.SubElement(root, 'Variances')
                        for i in range(clf.var_.shape[0]):
                            varianceElement = ET.SubElement(variancesElement, 'Variance')
                            varianceElement.set('class', str(clf.classes_[i]))
                            for j in range(clf.var_.shape[1]):
                                featureElement = ET.SubElement(varianceElement, 'Feature')
                                featureElement.set('index', str(j))
                                featureElement.set('value', str(clf.var_[i][j]))

                        # Write the XML data to a file
                        tree = ET.ElementTree(root)
                        tree.write(f'{fileName.split("/")[-1]}-ParametersNB.xml')
                        print(f'Naive Bayes parameters saved to {fileName.split("/")[-1]}-ParametersNB.xml')

                # Create the "Save Rules" button
                saveRulesButton = tk.Button(root, text='Save Parameters', command=onSaveRules)
                saveRulesButton.pack()

                # Function to handle the "Predict" button
                def onPredict():
                    if clf:
                        inputData = {}
                        for column in features.columns:
                            value = simpledialog.askstring('Input', f'Enter value for {column}:', parent=root)
                            inputData[column] = value
                        inputDf = pd.DataFrame([inputData])
                        prediction = clf.predict(inputDf)
                        tk.messagebox.showinfo('Prediction', f'Prediction: {prediction[0]}')

                # Create the "Predict" button
                predictButton = tk.Button(root, text='Predict', command=onPredict)
                predictButton.pack()

                # Function to handle the "Save Classifier" button
                def onSaveClassifier():
                    if clf:
                        dataclf = {'clf': clf, 'feature_names': features}
                        joblib.dump(dataclf, f'{fileName.split("/")[-1]}-classifierNB.joblib')
                        print(f'Classifier saved to {fileName.split("/")[-1]}-classifierNB.joblib')

                # Create the "Save Classifier" button
                saveClassifierButton = tk.Button(root, text='Save Classifier', command=onSaveClassifier)
                saveClassifierButton.pack()


            elif classifier == 'knn':
                clf = KNeighborsClassifier()
                clf.fit(features, target)
                print('KNN trained!')

                def visualizeKnn():
                    # Create a heatmap of the correlation matrix
                    corr = features.corr()
                    plt.imshow(corr, cmap='coolwarm')
                    plt.xticks(range(len(corr.columns)), corr.columns)
                    plt.yticks(range(len(corr.columns)), corr.columns)
                    plt.colorbar()
                    plt.title('Heatmap')

                def displayKNN():
                    visualizeKnn()
                    plt.show()

                visualizeKnnButton = tk.Button(root, text='Show Heatmap', command=displayKNN)
                visualizeKnnButton.pack()

                # Function to handle the "Save Rules" button
                def onSaveRules():
                    if clf:
                        # Create the root element
                        root = ET.Element('KNN')

                        # Create an element for k
                        kElement = ET.SubElement(root, 'k')
                        kElement.text = str(clf.n_neighbors)

                        # Create an element for the distance metric
                        metricElement = ET.SubElement(root, 'metric')
                        metricElement.text = clf.metric

                        # Create an element for the training instances
                        instancesElement = ET.SubElement(root, 'instances')
                        for i in range(clf._fit_X.shape[0]):
                            instanceElement = ET.SubElement(instancesElement, 'instance')
                            instanceElement.set('class', str(clf._y[i]))
                            for j in range(clf._fit_X.shape[1]):
                                featureElement = ET.SubElement(instanceElement, 'feature')
                                featureElement.set('index', str(j))
                                featureElement.set('value', str(clf._fit_X[i][j]))

                        # Write the XML data to a file
                        tree = ET.ElementTree(root)
                        tree.write(f'{fileName.split("/")[-1]}-ParametersKNN.xml')
                        print(f'KNN parameters saved to {fileName.split("/")[-1]}-ParametersKNN.xml')

                # Create the "Save Rules" button
                saveRulesButton = tk.Button(root, text='Save Rules', command=onSaveRules)
                saveRulesButton.pack()

                # Function to handle the "Predict" button
                def onPredict():
                    if clf:
                        inputData = {}
                        for column in features.columns:
                            value = simpledialog.askstring('Input', f'Enter value for {column}:', parent=root)
                            inputData[column] = value
                        inputDf = pd.DataFrame([inputData])
                        prediction = clf.predict(inputDf)
                        tk.messagebox.showinfo('Prediction', f'Prediction: {prediction[0]}')

                # Create the "Predict" button
                predictButton = tk.Button(root, text='Predict', command=onPredict)
                predictButton.pack()

                # Function to handle the "Save Classifier" button
                def onSaveClassifier():
                    if clf:
                        dataclf = {'clf': clf, 'feature_names': features}
                        joblib.dump(dataclf, f'{fileName.split("/")[-1]}-classifierKNN.joblib')
                        print(f'Classifier saved to {fileName.split("/")[-1]}-classifierKNN.joblib')

                # Create the "Save Classifier" button
                saveClassifierButton = tk.Button(root, text='Save Classifier', command=onSaveClassifier)
                saveClassifierButton.pack()


def onLoadClassifier():
    fileName = filedialog.askopenfilename(filetypes=[('Joblib Files', '*.joblib')])
    if fileName:
        data = joblib.load(fileName)
        clf = data['clf']
        featureNames = data['feature_names']
        print('Classifier loaded from', fileName)

        loadClassifierButton.pack_forget()
        openButton.pack_forget()

        def onPredict():
            if clf:
                inputData = {}
                for column in featureNames.columns:
                    value = simpledialog.askstring('Input', f'Enter value for {column}:', parent=root)
                    inputData[column] = value
                inputDf = pd.DataFrame([inputData])
                prediction = clf.predict(inputDf)
                tk.messagebox.showinfo('Prediction', f'Prediction: {prediction[0]}')

        predictButton = tk.Button(root, text='Predict', command=onPredict)
        predictButton.pack()


# Create the "Open" button
openButton = tk.Button(root, text='Open CSV', command=onOpen)
openButton.pack()

# Create the "Load Classifier" button
loadClassifierButton = tk.Button(root, text='Load Classifier', command=onLoadClassifier)
loadClassifierButton.pack()

# Create the image label
image_label = tk.Label(root)
image_label.pack()

# Run the main loop
root.mainloop()
