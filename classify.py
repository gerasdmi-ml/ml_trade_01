# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree
import matplotlib.pyplot as plt


col_names = ['returns','ret_2', 'ret_5', 'ret_10','ret_21','rsi','macd','atr','stoch','ultosc','result','returns_v','ret_2_v', 'ret_5_v', 'ret_10_v','ret_21_v','rsi_v','macd_v','macd_0','macd_1','macd_2']




feature_cols = ['returns','ret_2', 'ret_5', 'ret_10','ret_21','rsi','macd','atr','stoch','ultosc','returns_v','ret_2_v', 'ret_5_v', 'ret_10_v','ret_21_v','rsi_v','macd_v','macd_0','macd_1','macd_2']

#feature_cols = ['returns','ret_2', 'ret_5', 'ret_10','ret_21','rsi','macd','atr','stoch','ultosc']


pima = pd.read_csv("D:/ml_data/ml_input/ml_data_700.csv", sep=';', header=None, names=col_names)


positive = pima['result'].sum()
total = pima['result'].count()

print('base ratio=', round(positive /total,2))

X = pima[feature_cols] # Features
y = pima.result # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


'''
clf = DecisionTreeClassifier(criterion="entropy", max_depth=2)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_train)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
'''


clf = DecisionTreeClassifier(criterion="entropy", max_depth=2)
clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_train)
print("Accuracy train:",metrics.accuracy_score(y_train, y_pred))


y_pred = clf.predict(X_test)

#print (y_pred)
#print (y_test)

print("Accuracy test :",metrics.accuracy_score(y_test, y_pred))



text_representation = tree.export_text(clf)
print(text_representation)


fig = plt.figure(figsize=(5,5))
_ = tree.plot_tree(clf,
                   feature_names=col_names,
                   class_names='result',
                   filled=True)

plt.show()