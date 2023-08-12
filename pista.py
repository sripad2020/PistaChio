import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sn
import matplotlib.pyplot as plt
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from keras.models import Sequential
from keras.layers import Dense
import keras.activations,keras.losses,keras.metrics

data=pd.read_csv('pistachio.csv')
print(data.columns)
print(data.describe())
print(data.isna().sum())
print(data.info())
lab=LabelEncoder()

for i in data.select_dtypes(include='object').columns.values:
    data[i]=lab.fit_transform(data[i])

for i in data.columns.values:
    data['z-scores']=(data[i]-data[i].mean())/data[i].std()
    outliers=np.abs(data['z-scores'])
    print(f'The number of outliers in column for  {i}  ',outliers.sum())

for i in data.columns.values:
    upper=data[i].mean()+1.8*data[i].std()
    lower=data[i].mean()-1.8*data[i].std()
    data=data[(data[i] >lower)&(data[i]<upper)]

'''for i in data.columns.values:
    if len(data[i].value_counts())<=20:
        val=data[i].value_counts().values
        index=data[i].value_counts().index
        plt.pie(val,labels=index,autopct='%1.1f%%')
        plt.title(f'{i} column values')
        plt.legend()
        plt.show()

sn.pairplot(data)
plt.show()

# 2. Pair Plot
sn.pairplot(data, hue='HVACSystem')
plt.show()


correlation_matrix = data.corr()
sn.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()'''
'''plt.figure(figsize=(17, 6))
corr = data.corr(method='spearman')
my_m = np.triu(corr)
sn.heatmap(corr, mask=my_m, annot=True, cmap="Set2")
plt.show()

for i in data.select_dtypes(include='number').columns.values:
    for j in data.select_dtypes(include='number').columns.values:
        sn.histplot(data[i], label=f"{i}", color='red')
        sn.histplot(data[j], label=f"{j}", color="blue")
        plt.title(f"ITS {i} vs {j}")
        plt.legend()
        plt.show()'''''
for i in data.columns.values:
    sn.boxplot(data[i])
    plt.show()

x=data[['AREA', 'MINOR_AXIS','ROUNDNESS',
       'EQDIASQ', 'SOLIDITY', 'CONVEX_AREA', 'EXTENT', 'ASPECT_RATIO',
        'COMPACTNESS',
       'SHAPEFACTOR_3', 'SHAPEFACTOR_4']]
y=data['Class']
x_train,x_test,y_train,y_test=train_test_split(x,y)

lr = LogisticRegression(max_iter=200)
lr.fit(x_train, y_train)
print('The logistic regression: ', lr.score(x_test, y_test))

xgb = XGBClassifier()
xgb.fit(x_train, y_train)
print("the Xgb : ", xgb.score(x_test, y_test))

lgb = LGBMClassifier()
lgb.fit(x_train, y_train)
print('The LGB', lgb.score(x_test, y_test))

tree = DecisionTreeClassifier(criterion='gini', max_depth=1)
tree.fit(x_train, y_train)
print('Dtree ', tree.score(x_test,y_test))

rforest = RandomForestClassifier(criterion='gini')
rforest.fit(x_train, y_train)
print('The random forest: ', rforest.score(x_test, y_test))

adb = AdaBoostClassifier()
adb.fit(x_train, y_train)
print('the adb ', adb.score(x_test, y_test))

grb = GradientBoostingClassifier()
grb.fit(x_train, y_train)
print('Gradient boosting ', grb.score(x_test, y_test))

bag = BaggingClassifier()
bag.fit(x_train, y_train)
print('Bagging', bag.score(x_test, y_test))

X=data[['AREA', 'MINOR_AXIS','ROUNDNESS',
       'EQDIASQ', 'SOLIDITY', 'CONVEX_AREA', 'EXTENT', 'ASPECT_RATIO',
        'COMPACTNESS',
       'SHAPEFACTOR_3', 'SHAPEFACTOR_4']]
Y=pd.get_dummies(data['Class'])
x_trin,x_tst,y_trin,y_tst=train_test_split(X,Y)

models=Sequential()
models.add(Dense(units=X.shape[1],input_dim=X.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=X.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=X.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=X.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=X.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=Y.shape[1],activation=keras.activations.sigmoid))
models.compile(optimizer='adam',loss=keras.losses.binary_crossentropy,metrics='accuracy')
hist=models.fit(x_trin,y_trin,batch_size=20,epochs=350)

plt.plot(hist.history['accuracy'], label='training accuracy', marker='o', color='red')
plt.plot(hist.history['loss'], label='loss', marker='o', color='darkblue')
plt.title('Training Vs  Validation accuracy with adam optimizer')
plt.legend()
plt.show()


models1=Sequential()
models1.add(Dense(units=X.shape[1],input_dim=X.shape[1],activation=keras.activations.sigmoid))
models1.add(Dense(units=X.shape[1],activation=keras.activations.sigmoid))
models1.add(Dense(units=X.shape[1],activation=keras.activations.sigmoid))
models1.add(Dense(units=X.shape[1],activation=keras.activations.sigmoid))
models1.add(Dense(units=Y.shape[1],activation=keras.activations.sigmoid))
models1.compile(optimizer='rmsprop',loss=keras.losses.binary_crossentropy,metrics='accuracy')
histo=models1.fit(x_trin,y_trin,batch_size=20,epochs=350)

plt.plot(histo.history['accuracy'], label='training accuracy', marker='o', color='red')
plt.plot(histo.history['loss'], label='loss', marker='o', color='darkblue')
plt.title('Training Vs  Validation accuracy with adam rmsprop')
plt.legend()
plt.show()