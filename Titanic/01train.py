import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from scipy.stats import mode
import cmath
import time
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import PolynomialFeatures

data=pd.read_csv('train.csv',header=0)
print("读取train文件完毕，开始预处理")
enc=OneHotEncoder()
#现有的判空方法都不支持字符串类型，所以手写一个
def isnan(num):
    return num != num
#计算各个列的缺失值个数
# print(data.isnull().sum())
#统计Cabin的缺失值字段是否应该填充
#如果缺失值比例很大，且是否为空与标记的关联性较大，说明应该将缺失值作为一个特征值，不予填充
# data['Cabin01']=data["Cabin"].apply(lambda x:1 if x is not np.nan else 0)
# pd.pivot_table(data,index=['Cabin01'],values=['Survived']).plot.bar(figsize=(8,5))
# plt.title('Survival Rate')
# plt.show()
# imp=Imputer(missing_values='NaN',strategy='most_frequent',axis=0) 这种好像只适用于数值类型，文本类型不行
# imp.fit_transform(data['Embarked'])

#Passenger字段删掉
#Survived字段是标签
#Pclass字段
# onehotpclass=enc.fit_transform(np.array(data['Pclass']).reshape(-1,1)).toarray()
labelpclass=LabelEncoder().fit_transform(np.array(data['Pclass']))
#Name，暂且认为姓名只用于补齐缺失年龄
#Sex字段
labelsex=LabelEncoder().fit_transform(np.array(data['Sex']))
# onehotsex=enc.fit_transform(labelsex.reshape(-1,1)).toarray()
#Age
data['Title']=data['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
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

mraver=computeMean(data,'Mr')
mrsaver=computeMean(data,'Mrs')

for index in range(len(data['Age'])):
    if cmath.isnan(data['Age'][index]):
        if data['Title'][index] == 'Miss':
            if data['SibSp'][index]>0 or data['Parch'][index]>0:
                data['Age'][index] = 10
            else:
                data['Age'][index] = 25
        elif data['Title'][index] == 'Master':
            if data['SibSp'][index] > 0 or data['Parch'][index] > 0:
                data['Age'][index] = 5
            else:
                data['Age'][index] = 15
        elif data['Title'][index] == 'Mr':
            data['Age'][index]=mraver
        elif data['Title'][index] == 'Mrs':
            data['Age'][index]=mrsaver
        else:
            data['Age'][index] = mrsaver
#SibSp不用改
#Parch不用改
#Ticket字段
labelticket= LabelEncoder().fit_transform(np.array(data['Ticket']))
# onehotticket=enc.fit_transform(labelticket.reshape(-1,1)).toarray()
#Fare不用改
#Cabin字段
onehotcabin=[]
allcabin=[]
cabinperlist=[]
#统计一共出现了几个cabin，然后为它们生成onehot
for index in range(len(data['Cabin'])):
    str=data['Cabin'][index]
    if isnan(str):
        cabinperlist.append(['Z'])#空值用Z代替
        data['Cabin'][index]='Z'
    else:
        iscabin=str.strip().split(' ')
        cabinperlist.append(iscabin)
        for i in range(len(iscabin)):
            if iscabin[i] not in allcabin:
                allcabin.append(iscabin[i])
allcabin.append('Z')
labelcabin=LabelEncoder().fit_transform(np.array(data['Cabin']))
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
#补齐上车站字段，mode是众数
data['Embarked']=numberlize(data['Embarked'])
# labelembarked=LabelEncoder().fit_transform(np.array(data['Embarked']))
data['Embarked'].fillna(mode(data['Embarked']).mode[0],inplace=True)
# onehotembardked=enc.fit_transform(np.array(data['Embarked']).reshape(-1,1)).toarray()

#拼接
# newdata=np.concatenate((onehotpclass,onehotsex,np.array(data['Age']).reshape(-1,1),np.array(data['SibSp']).reshape(-1,1),np.array(data['Parch']).reshape(-1,1),
#                        onehotticket,np.array(data['Fare']).reshape(-1,1),onehotcabin,onehotembardked),axis=1)
newdata=np.concatenate((np.array(labelpclass).reshape(-1,1),np.array(labelsex).reshape(-1,1),np.array(data['Age']).reshape(-1,1),np.array(data['SibSp']).reshape(-1,1)
                        ,np.array(data['Parch']).reshape(-1,1),np.array(labelticket).reshape(-1,1),np.array(data['Fare']).reshape(-1,1),np.array(labelcabin).reshape(-1,1)
                        ,np.array(data['Embarked']).reshape(-1,1)),axis=1)
# newdata=np.concatenate((np.array(data['Pclass']).reshape(-1,1),np.array(data['Sex']).reshape(-1,1),np.array(data['Age']).reshape(-1,1),np.array(data['SibSp']).reshape(-1,1)
#                         ,np.array(data['Parch']).reshape(-1,1),np.array(data['Ticket']).reshape(-1,1),np.array(data['Fare']).reshape(-1,1),np.array(data['Cabin']).reshape(-1,1)
#                         ,np.array(data['Embarked']).reshape(-1,1)),axis=1)直接上字符串字段的好像不行，SVM不能传入字符串类型
print("数据处理完毕，开始训练")
#将转换好的数据输出csv
# newti=pd.DataFrame(data=newdata)
# newti.to_csv('newti.csv',encoding='utf-8')
# print(data.isnull().sum())
# print("done")
start=time.time()
#GBDT
# param_grid={'n_estimators':[100,110,120,130,140,150,160],'learning_rate':[0.05,0.08,0.1,0.12],'max_depth':[3,4,5]}
# grid_search=GridSearchCV(GradientBoostingClassifier(),param_grid,cv=5)
#LR
# param_grid={'penalty':['l2','l1'],'solver':['saga','liblinear']}
# grid_search=GridSearchCV(LogisticRegression(),param_grid,cv=5)
#RF
# param_grid={'n_estimators':[10,12,15],'criterion':['gini','entropy'],'max_depth':[3,4,5]}
# grid_search=GridSearchCV(RandomForestClassifier(),param_grid,cv=5)
#SVM
param_grid={'C':[0.7,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.5],'gamma':[0.005,0.008,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.12,0.15]}
model=SVC(kernel='rbf',probability=False)
#XGBoost
# param_grid={'learning_rate':[0.05,0.08,0.1,0.11,0.12,0.15,0.18,0.2],'max_depth':[3,4,5,6],'n_estimators':[110,120,130,140,150,200]}
# model=XGBClassifier(objective='binary:logistic')
grid_search=GridSearchCV(model,param_grid,cv=5)
newdata=MinMaxScaler().fit_transform(newdata)
newdata=StandardScaler().fit_transform(newdata)
# poly = PolynomialFeatures(2)
# newdata=poly.fit_transform(newdata)
grid_search.fit(newdata,np.array(data['Survived']).reshape(-1,1).ravel())
end=time.time()
# print("训练完毕，用时"+str(end-start))
print("训练结果:")
print(grid_search.best_params_,grid_search.best_score_)

bestparams=grid_search.best_params_
# model=GradientBoostingClassifier(n_estimators=bestparams['n_estimators'],learning_rate=bestparams['learning_rate'],max_depth=bestparams['max_depth'])
# model=XGBClassifier(objective='binary:logistic',learning_rate=bestparams['learning_rate'],max_depth=bestparams['max_depth'],n_estimators=bestparams['n_estimators'])
model=SVC(kernel='rbf',C=bestparams['C'],gamma=bestparams['gamma'],probability=False)
model.fit(newdata,np.array(data['Survived']).reshape(-1,1).ravel())
savemodel=joblib.dump(model,'01GBDT.pkl')
print("模型已保存至本地：01GBDT.pkl")