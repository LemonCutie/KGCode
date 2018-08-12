import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import mode
import cmath
import time
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

def computeMean(dataset,sexname,column='Title'):
    sum=0;
    count=0;
    for i in range(len(dataset[column])):
        if dataset[column][i]==sexname:
            if not cmath.isnan(dataset['Age'][i]):
                sum+=float(dataset['Age'][i])
                count+=1
    aver=sum/count
    return aver

def isnan(num):
    return num != num
def numberlize(embarkedlist):
    templist=[]
    for index in range(len(embarkedlist)):
        str = embarkedlist[index]
        if str=='S':
            templist.append(1)
        elif str =='C':
            templist.append(2)
        elif str=='Q':
            templist.append(3)
        else:
            templist.append(0)
    return templist

enc=OneHotEncoder()
model=joblib.load('01GBDT.pkl')
print("开始测试")
#测试
testdata=pd.read_csv('test.csv',header=0)
#Passenger字段删掉
#没有Survived字段
#Pclass字段
# onehotpclass=enc.fit_transform(np.array(testdata['Pclass']).reshape(-1,1)).toarray()
labelpclass=LabelEncoder().fit_transform(np.array(testdata['Pclass']))
#Name，暂且认为姓名只用于补齐缺失年龄
#Sex字段
labelsex=LabelEncoder().fit_transform(np.array(testdata['Sex']))
# onehotsex=enc.fit_transform(labelsex.reshape(-1,1)).toarray()
#Age
testdata['Title']=testdata['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
mraver=computeMean(testdata,'Mr')
mrsaver=computeMean(testdata,'Mrs')

for index in range(len(testdata['Age'])):
    if cmath.isnan(testdata['Age'][index]):
        if testdata['Title'][index] == 'Miss':
            if testdata['SibSp'][index]>0 or testdata['Parch'][index]>0:
                testdata['Age'][index] = 10
            else:
                testdata['Age'][index] = 25
        elif testdata['Title'][index] == 'Master':
            if testdata['SibSp'][index] > 0 or testdata['Parch'][index] > 0:
                testdata['Age'][index] = 5
            else:
                testdata['Age'][index] = 15
        elif testdata['Title'][index] == 'Mr':
            testdata['Age'][index]=mraver
        elif testdata['Title'][index] == 'Mrs':
            testdata['Age'][index]=mrsaver
        else:
            testdata['Age'][index] = mrsaver
#SibSp不用改
#Parch不用改
#Ticket字段
labelticket= LabelEncoder().fit_transform(np.array(testdata['Ticket']))
# onehotticket=enc.fit_transform(labelticket.reshape(-1,1)).toarray()
#Fare字段有缺失值，用中位数填补
testdata['Fare'].fillna(mode(testdata['Fare']).mode[0],inplace=True)
#Cabin字段
onehotcabin=[]
allcabin=[]
cabinperlist=[]
#统计一共出现了几个cabin，然后为它们生成onehot
for index in range(len(testdata['Cabin'])):
    str=testdata['Cabin'][index]
    if isnan(str):
        cabinperlist.append(['Z'])#空值用Z代替
        testdata['Cabin'][index] = 'Z'
    else:
        iscabin=str.strip().split(' ')
        cabinperlist.append(iscabin)
        for i in range(len(iscabin)):
            if iscabin[i] not in allcabin:
                allcabin.append(iscabin[i])
allcabin.append('Z')
labelcabin=LabelEncoder().fit_transform(np.array(testdata['Cabin']))
# labelcabin=LabelEncoder().fit_transform(allcabin)
# onehotcodeofcabin=enc.fit_transform(labelcabin.reshape(-1,1)).toarray()
# #建立字典，每个船厢都有自己的onehot码
# onehotofcabindic={}
# for index in range(len(allcabin)):
#     onehotofcabindic[allcabin[index]]=np.array(onehotcodeofcabin[index])
# for index in range(len(cabinperlist)):
#     per = cabinperlist[index]
#     s=[]
#     for i in range(len(per)):#一个人对应好几个船厢，相加
#         s.append(onehotofcabindic[per[i]])
#     s=np.array(s)
#     onehotcabin.append(np.sum(s,axis=0))
#Embarked字段
#补齐上车站字段，mode是众数
testdata['Embarked']=numberlize(testdata['Embarked'])
testdata['Embarked'].fillna(mode(testdata['Embarked']).mode[0],inplace=True)
# onehotembardked=enc.fit_transform(np.array(testdata['Embarked']).reshape(-1,1)).toarray()

#拼接
# newtestdata=np.concatenate((onehotpclass,onehotsex,np.array(testdata['Age']).reshape(-1,1),np.array(testdata['SibSp']).reshape(-1,1),np.array(testdata['Parch']).reshape(-1,1),
#                        onehotticket,np.array(testdata['Fare']).reshape(-1,1),onehotcabin,onehotembardked),axis=1)
newtestdata=np.concatenate((np.array(labelpclass).reshape(-1,1),np.array(labelsex).reshape(-1,1),np.array(testdata['Age']).reshape(-1,1),np.array(testdata['SibSp']).reshape(-1,1)
                        ,np.array(testdata['Parch']).reshape(-1,1),np.array(labelticket).reshape(-1,1),np.array(testdata['Fare']).reshape(-1,1),np.array(labelcabin).reshape(-1,1)
                        ,np.array(testdata['Embarked']).reshape(-1,1)),axis=1)
# print(testdata.isnull().sum())
newtestdata=MinMaxScaler().fit_transform(newtestdata)
newtestdata=StandardScaler().fit_transform(newtestdata)
# poly = PolynomialFeatures(2)
# newtestdata=poly.fit_transform(newtestdata)
result=model.predict(newtestdata)
tt=pd.DataFrame({'PassengerId':testdata['PassengerId'],'Survived':result})
tt.to_csv('sub.csv',index=False)
print("测试完毕")