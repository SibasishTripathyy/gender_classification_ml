from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

# Initialising all the classifiers
classifier_tree = tree.DecisionTreeClassifier()
classifier_neighbor = KNeighborsClassifier()
classifier_svm = SVC()
classifier_NB = GaussianNB()

# Fitting all the classifiers to the dataset
classifier_tree.fit(X, y)
classifier_neighbor.fit(X, y)
classifier_svm.fit(X, y)
classifier_NB.fit(X, y)

# Predicting result of a specific data using all the classifiers
prediction_tree = classifier_tree.predict([[190, 70, 43]])
prediction_neighbor = classifier_neighbor.predict([[190, 70, 43]])
prediction_svm = classifier_svm.predict([[190, 70, 43]])
prediction_NB = classifier_NB.predict([[190, 70, 43]])


# Applying k-fold cross validation 
from sklearn.model_selection import cross_val_score
accuracy_tree = cross_val_score(estimator = classifier_tree, X = X, y = y)
accuracy_neighbor = cross_val_score(estimator = classifier_neighbor, X = X, y = y)
accuracy_svm = cross_val_score(estimator = classifier_svm, X = X, y = y)
accuracy_NB = cross_val_score(estimator = classifier_NB, X = X, y = y)

# Calculating the mean of the accuracies
mean_tree = accuracy_tree.mean()
mean_neighbor = accuracy_neighbor.mean()
mean_svm = accuracy_svm.mean()
mean_NB = accuracy_NB.mean()

# Finding the most accurate prediction
most_accurate = max(accuracy_tree.mean(), accuracy_neighbor.mean(), accuracy_svm.mean(), accuracy_NB.mean())

# Comparing and declaring the result
if most_accurate == mean_tree:
    print("Decision Tree Classification has the best result")
elif most_accurate == mean_neighbor:
    print("K-NN Classification has the best result")
elif most_accurate == mean_svm:
    print("SVM Classification has the best result")
elif most_accurate == mean_NB:
    print("Naive Bayes Classification has the best result")