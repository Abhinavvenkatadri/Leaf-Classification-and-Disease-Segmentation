import numpy as np
import pandas as pd
import os
import string

dataset = pd.read_csv("Flavia_features_final.csv")


dataset.head(5)

breakpoints = [1001,1059,1060,1122,1552,1616,1123,1194,1195,1267,1268,1323,1324,1385,1386,1437,1497,1551,1438,1496,2001,2050,2051,2113,2114,2165,2166,2230,2231,2290,2291,2346,2347,2423,2424,2485,2486,2546,2547,2612,2616,2675,3001,3055,3056,3110,3111,3175,3176,3229,3230,3281,3282,3334,3335,3389,3390,3446,3447,3510,3511,3563,3566,3621]



maindir = r'C:\Users\Admin\Desktop\Leaf\Leaves\Leaves_Flavia'
ds_path = maindir + "\\Flavia"
img_files = os.listdir(ds_path)




target_list = []
for file in img_files:
    target_num = int(file.split(".")[0])
    flag = 0
    i = 0 
    for i in range(0,len(breakpoints),2):
        if((target_num >= breakpoints[i]) and (target_num <= breakpoints[i+1])):
            flag = 1
            break
    if(flag==1):
        target = int((i/2))
        target_list.append(target)

y = np.array(target_list)
y


X = dataset.iloc[:,1:]

X.head(5)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 142)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


from sklearn import svm
clf = svm.SVC(C= 10, kernel='rbf',gamma=0.1)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

from sklearn import metrics
metrics.accuracy_score(y_test, y_pred)
print(metrics.classification_report(y_test, y_pred))
cm=metrics.confusion_matrix(y_test, y_pred)

from sklearn.model_selection import GridSearchCV
parameters = [{'kernel': ['rbf'],
               'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
               'C': [1, 10, 100, 1000]},
              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
              {'kernel': ['poly'],'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5], 'C': [1, 10, 100, 1000]}
             ]
svm_clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=5)
svm_clf.fit(X_train, y_train)


svm_clf.best_params_
means = svm_clf.cv_results_['mean_test_score']
stds = svm_clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, svm_clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

y_pred_svm = svm_clf.predict(X_test)
metrics.accuracy_score(y_test, y_pred_svm)
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_svm)
print(metrics.classification_report(y_test, y_pred_svm))

print(metrics.confusion_matrix(y_test, y_pred_svm))


