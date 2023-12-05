
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from google.colab import drive
drive.mount('/content/drive')

data = pd.read_csv("/content/drive/MyDrive/GlaucomaDATA.csv")

data.info()

data



data.duplicated()

data.isnull().sum()

plt.plot(data['Class'])
plt.xlabel("at")
plt.ylabel("Levels")
plt.title("Line Plot")
plt.show()

#the data contains String values In class Column
#storing the data in a new variable to change the string values
newdata = data
#the line replaces string vlaues normal and glaucoma with 0 and 1 as type integer
newdata['Class'] = newdata['Class'].replace(r'normal','0').replace(r'glaucoma','1').astype(int)
newdata

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
data_len=data[newdata['Class']==1]['eat'].value_counts()
ax1.hist(data_len,color='red')
ax1.set_title('Having Glaucoma')
data_len=data[newdata['Class']==0]['eat'].value_counts()
ax2.hist(data_len,color='green')
ax2.set_title('Not Having Glaucoma')
fig.suptitle('Glaucoma results')
plt.show()

from sklearn import preprocessing
d = preprocessing.normalize(newdata.iloc[:,1:5], axis=0)
scaled_df = pd.DataFrame(d, columns=["Class", "at", "eat", "mv"])
scaled_df.head()

feature = ['at','eat']
train,test=train_test_split(newdata,test_size=0.3,random_state=0,stratify=data['Class'])
X=data[feature]
Y=data['Class']
train_X=train[train.columns[:-1]]
train_Y=train[train.columns[-1:]]
test_X=test[test.columns[:-1]]
test_Y=test[test.columns[-1:]]

model = LogisticRegression()
model.fit(train_X,train_Y)
prediction3=model.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction3,test_Y))
report = classification_report(test_Y, prediction3)
print("Classification Report:\n", report)

#confusion Matrix
cm = confusion_matrix(test_Y, prediction3)
cm
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 16})
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

model = LinearRegression()
model.fit(train_X, train_Y)
prediction = model.predict(test_X)
accuracy = accuracy_score(test_Y, prediction.round())
print('The accuracy of Linear Regression is:', accuracy)