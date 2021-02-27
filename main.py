from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
import pandas as pd
import numpy as np
import json
import os
import graphviz
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import tree

path = os.getcwd()
isAllFeatures = False
if isAllFeatures == True:
  outfile = "all_features"#feature_selection
else:
  outfile = "feature_selection"
#term-deposit-marketing-2020.csv

df = pd.read_csv(path+"\\"+"term-deposit-marketing-2020.csv")
labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
df['job'] = labelencoder.fit_transform(df['job'])
df['marital'] = labelencoder.fit_transform(df['marital'])
df['education'] = labelencoder.fit_transform(df['education'])
df['month'] = labelencoder.fit_transform(df['month'])
df['default'] = labelencoder.fit_transform(df['default'])
df['contact'] = labelencoder.fit_transform(df['contact'])
df['housing'] = labelencoder.fit_transform(df['housing'])
df['loan'] = labelencoder.fit_transform(df['loan'])
df['y'] = labelencoder.fit_transform(df['y'])

sns.heatmap(df.corr())
plt.savefig(path+"\\"+'correlation.png',dpi=400)

df = shuffle(df)
if isAllFeatures == True:
   X = df.drop(columns=['y'])
else:
   X = df[['duration']]#feature selection part
feature_names = X.columns
y = df['y'].values

X_folds = np.array_split(X, 5)
y_folds = np.array_split(y, 5)
scores = list()
prediction_list = np.zeros(y.shape)
target_list = np.zeros(y.shape)
models_list = []
for k in range(5):
  clf =  DecisionTreeClassifier(random_state=0,max_depth = 5,class_weight={0:1,1:10})
  X_train = list(X_folds)
  X_test = X_train.pop(k)
  X_train = np.concatenate(X_train)
  y_train = list(y_folds)
  y_test = y_train.pop(k)
  y_train = np.concatenate(y_train)

  scores.append(clf.fit(X_train, y_train).score(X_test, y_test))
  models_list.append(clf)
  prediction_list[k*y_test.shape[0]:(k+1)*y_test.shape[0]] = clf.predict(X_test)

for i, clf in enumerate(models_list):
  dot_data = tree.export_graphviz(clf, out_file=None,feature_names = feature_names,
                                  class_names = ["no","yes"],
                                  filled=True)
  graph = graphviz.Source(dot_data, format="png")
  graph.render(path+"\\"+outfile+"\\decision_tree_"+str(i))

df[prediction_list == 1].to_csv(path+"\\"+'segmentofcustomer.csv',index=False)

evaluation_dic = {}
conf_mat = confusion_matrix(y, prediction_list)
accuracy = (conf_mat[1,1] + conf_mat[0,0])/np.sum(conf_mat)
evaluation_dic['accuracy'] = accuracy

precision = conf_mat[1,1]/(conf_mat[0,1] + conf_mat[1,1])
evaluation_dic['precision'] = precision

recall = conf_mat[1,1]/(conf_mat[1,0] + conf_mat[1,1])
evaluation_dic['recall'] = recall
evaluation_dic

with open(path+"\\"+outfile+'_result.txt', 'w') as file:
     file.write(json.dumps(evaluation_dic)) # use `json.loads` to do the reverse