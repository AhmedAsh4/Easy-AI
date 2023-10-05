# V2 Added
Version 2 now allows for Both Regression and Classification. <br>
Where both involve selecting and .CSV file, preprocessing it then passing it by a Voting ensemble. <br>
Models used for Regression:
- XGBoost Regressor
- Lasso Regressor
- KNeighbors Regressor
<br>
Models used for Classification: <br>
- XGBoost Classifier
- MultinomialNB
- KNeighbors Classifier

Both options allow for displaying performance graphs (PredictionsVsActual and QQplot for regression , Confusion matrix and ROC curve for classification), as well as predicting new data,saving the classifier as .joblib, and saving metrics in .txt file <br>

you can also load .joblib classifier to predict data without training again <br>
<br> <br>
# Expert System
This code started as a Project for AI312 (Expert Systems). <br>
The Program is inspired by WEKA 3 by the university of waikato. <br>
The program can accept any .CSV file that can be separated into features and label. 
The User can choose from 3 classification algorithms (Decision Tree/ Naive Bayes/ KNN).
Each Classification algorithm gets a different visualization, where the Decision Tree gets a Tree, the Naive Bayes get a Histogram of each feature, and the KNN gets a Heatmap.
The User can save details about each classsifier into an .XML file, 
The Program also allows the user to use the classifier to make new predictions. 
And finally the user can save the classifier into a .joblib file and load it again whenever they need
