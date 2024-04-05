import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


#Plot the graph for feature selection for decision tree and random forest
def plot_feature_importances_mydata(model):
    n_features = X_train_std.shape[1]
    plt.figure(figsize=(8,6))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    temp_data = data.drop(data.columns[-1], axis=1)

    plt.yticks(np.arange(n_features), list(temp_data))
    plt.title("Feature Selection")
    plt.xlabel("Variable importance")
    plt.ylabel("Independent Variable")
    plt.show()
# Reading the dataset voice.csv

if __name__ == '__main__':
    data = pd.read_csv("voiceDataSet.csv")
    data.head()

    # Check the distribution of male and female in each feature columns

    for col in data.columns:
        plt.hist(data.loc[data['label'] == 'female', col], label="female")
        plt.hist(data.loc[data['label'] == 'male', col], label="male")
        plt.title(col)
        plt.xlabel("Feature magnitude")
        plt.ylabel("Frequency")
        plt.legend(loc='upper right')
        plt.show()

    from sklearn.model_selection import train_test_split

    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y,
                         test_size=0.3,
                         random_state=0,
                         stratify=y)

    # Scaling the features

    from sklearn.preprocessing import StandardScaler

    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)

    # Train logistic regression model

    logit = LogisticRegression()
    logit.fit(X_train_std, y_train)

    print("Logistic Regression")
    print("Accuracy on training set: {:.3f}".format(logit.score(X_train_std, y_train)))
    print("Accuracy on test set: {:.3f}".format(logit.score(X_test_std, y_test)))

    y_pred_logit = logit.predict(X_test_std)
    print("Predicted value: ", y_pred_logit)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred_logit, average='micro')
    print("Precision, Recall and fscore:", precision, recall, fscore, )

    # Train Decision Tree mo
    # Train decision tree model

    tree = DecisionTreeClassifier(random_state=0, max_depth=4)
    tree.fit(X_train_std, y_train)

    print("Decision Tree")
    print("Accuracy on training set: {:.3f}".format(tree.score(X_train_std, y_train)))
    print("Accuracy on test set: {:.3f}".format(tree.score(X_test_std, y_test)))

    y_pred_tree = tree.predict(X_test_std)
    print("Predicted value: ", y_pred_tree)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred_tree, average='micro')
    print("Precision, Recall and fscore:", precision, recall, fscore, )

    # Train random forest model

    forest = RandomForestClassifier(n_estimators=5, random_state=0)
    forest.fit(X_train_std, y_train)

    print("Random Forest")
    print("Accuracy on training set: {:.3f}".format(forest.score(X_train_std, y_train)))
    print("Accuracy on test set: {:.3f}".format(forest.score(X_test_std, y_test)))

    y_pred_forest = forest.predict(X_test_std)
    print("Predicted value: ", y_pred_forest)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred_forest, average='micro')
    print("Precision, Recall and fscore:", precision, recall, fscore, )

    # Train support vector machine model

    svm = SVC()
    svm.fit(X_train_std, y_train)

    print("Support Vector Machine")
    print("Accuracy on training set: {:.3f}".format(svm.score(X_train_std, y_train)))
    print("Accuracy on test set: {:.3f}".format(svm.score(X_test_std, y_test)))

    y_pred_sm = svm.predict(X_test_std)
    print("Predicted value: ", y_pred_sm)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred_sm, average='micro')
    print("Precision, Recall and fscore:", precision, recall, fscore, )

    # Read the file which got generated using our voice samples and using code written in R.

    plot_feature_importances_mydata(forest)

    data_new = pd.read_csv("voiceSamples.csv")
    data_new.head()

    # Creating X and Y
    X1, y1 = data_new.iloc[:, :-1].values, data_new.iloc[:, -1].values
    y1

    # standardizing the features
    stdsc = StandardScaler()
    X1_std = stdsc.fit_transform(X1)

    # Predicting the target variable using Logistic, Decision Tree , Random Forest, SVM
    y1_pred_logit = logit.predict(X1_std)
    y1_pred_tree = tree.predict(X1_std)
    y1_pred_forest = forest.predict(X1_std)
    y1_pred_svm = svm.predict(X1_std)

    print("Logistic Regression: ", y1_pred_logit)
    print("TRee", y1_pred_tree)
    print("Random Forest: ", y1_pred_forest)
    print("SVM: ", y1_pred_svm)

