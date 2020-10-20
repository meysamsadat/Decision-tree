import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv(r'C:\Users\meysam-sadat\Desktop\winequality-red.csv',sep=';')
df.dtypes
df['quality'] = df['quality'].astype(object)
df['quality'] = df['quality'].astype(str)
df.dtypes
x = df.drop('quality',axis=1)
y = df.quality
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

clf = DecisionTreeClassifier(criterion='gini',max_depth=5)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
accuracy_score(y_test,y_pred)
from sklearn import tree
#how to viualize your tree
tree.export_graphviz(clf,out_file='wine-class.dot',feature_names=x_train.columns,class_names=y.unique(),label='all',rounded=True,filled=True)

#go to https://dreampuf.github.io/GraphvizOnline/ and paste your dot file in order to visualize your tree online