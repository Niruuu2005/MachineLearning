from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

training_data = ["This is a cat", "The sky is blue", "Python is a programming language"]
training_labels = ["animal", "color", "programming"]

testing_data = ["java is a programming language"]

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(training_data)
X_test = vectorizer.transform(testing_data)

clf = MultinomialNB()
clf.fit(X_train, training_labels)

predicted_labels = clf.predict(X_test)

print("Predicted Label:", predicted_labels)

ground_truth_labels = [training_labels[0]]
accuracy = accuracy_score(ground_truth_labels, predicted_labels)
print("Accuracy of class being ",ground_truth_labels[0],":",accuracy*100)

ground_truth_labels = [training_labels[1]]
accuracy = accuracy_score(ground_truth_labels, predicted_labels)
print("Accuracy of class being ",ground_truth_labels[0],":",accuracy*100)

ground_truth_labels = [training_labels[2]]
accuracy = accuracy_score(ground_truth_labels, predicted_labels)
print("Accuracy of class being ",ground_truth_labels[0],":",accuracy*100)
