import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import simpledialog
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from PIL import Image, ImageTk
import io
import pydotplus
import xml.etree.ElementTree as ET
import joblib

# Create the main window
root = tk.Tk()
root.title("Decision Tree")
root.geometry("250x250")

#function to handle the "Open" button
def on_open():
    filename = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
    if filename:
        data = pd.read_csv(filename)
        outcome_column = simpledialog.askstring('Input', 'Enter the outcome column:', parent=root)
        if outcome_column:
            features = data.drop(outcome_column, axis=1)
            target = data[outcome_column]
            dtc = DecisionTreeClassifier()
            dtc.fit(features, target)
            print('Decision tree trained!')

            # Visualize the decision tree
            dot_data = export_graphviz(dtc, out_file=None, feature_names=list(features.columns),
                                       class_names=[str(x) for x in dtc.classes_], filled=True)
            graph = pydotplus.graph_from_dot_data(dot_data)
            image_data = graph.create_png()
            image = Image.open(io.BytesIO(image_data))
            photo = ImageTk.PhotoImage(image)
            image_label.image = photo
            print("Done")

            # function to handle the "View Full Size" button
            def on_view_full_size():
                if image_label.image:
                    image = Image.open(io.BytesIO(image_data))
                    image.show()

            # Create the "View Full Size" button
            view_full_size_button = tk.Button(root, text='View Decision Tree', command=on_view_full_size)
            view_full_size_button.pack()

            # function to handle the "Save Rules" button
            def on_save_rules():
                if dtc:
                    rules = export_graphviz(dtc, out_file=None, feature_names=list(features.columns),
                                            class_names=[str(x) for x in dtc.classes_], filled=True)
                    root = ET.Element('rules')
                    root.text = rules
                    tree = ET.ElementTree(root)
                    tree.write('rules.xml')
                    print('Rules saved to rules.xml')

            # Create the "Save Rules" button
            save_rules_button = tk.Button(root, text='Save Rules', command=on_save_rules)
            save_rules_button.pack()

            # function to handle the "Predict" button
            def on_predict():
                if dtc:
                    input_data = {}
                    for column in features.columns:
                        value = simpledialog.askstring('Input', f'Enter value for {column}:', parent=root)
                        input_data[column] = value
                    input_df = pd.DataFrame([input_data])
                    prediction = dtc.predict(input_df)
                    tk.messagebox.showinfo('Prediction', f'Prediction: {prediction[0]}')

            # Create the "Predict" button
            predict_button = tk.Button(root, text='Predict', command=on_predict)
            predict_button.pack()

            # function to handle the "Save Classifier" button
            def on_save_classifier():
                if dtc:
                    joblib.dump(dtc, 'classifier.joblib')
                    print('Classifier saved to classifier.joblib')

            # Create the "Save Classifier" button
            save_classifier_button = tk.Button(root, text='Save Classifier', command=on_save_classifier)
            save_classifier_button.pack()

def on_load_classifier():
    global dtc
    filename = filedialog.askopenfilename(filetypes=[('Joblib Files', '*.joblib')])
    if filename:
        dtc = joblib.load(filename)
        print('Classifier loaded from', filename)

# Create the "Open" button
open_button = tk.Button(root, text='Open CSV', command=on_open)
open_button.pack()

# Create the "Load Classifier" button
load_classifier_button = tk.Button(root, text='Load Classifier', command=on_load_classifier)
load_classifier_button.pack()

# Create the image label
image_label = tk.Label(root)
image_label.pack()

# Run the main loop
root.mainloop()
