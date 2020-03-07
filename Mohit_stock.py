#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[69]:


dataset=pd.read_csv(r'/Users/Harshita/info1.csv')
dataset2=pd.read_csv(r'/Users/Harshita/tcs.csv')


# In[70]:


dataset


# In[6]:


plt.figure(figsize=(15,5))
plt.title('Infosys One year Dataset')
plt.xlabel('Date')
plt.ylabel('Stock Open prices ')
plt.plot(dataset['Open'])
plt.xticks(np.arange(0,241,20),dataset['Date'][0:241:20])


# In[7]:


plt.figure(figsize=(15,5))
plt.title('TCS One year Dataset')
plt.xlabel('Date')
plt.ylabel('Stock Open prices ')
plt.plot(dataset2['Open'])
plt.xticks(np.arange(0,241,20),dataset['Date'][0:241:20])


# In[7]:


x=np.mean(dataset['Open'])
x


# In[8]:


y=np.std(dataset['Open'])
y


# In[9]:


cv=y/x
cv*100


# In[8]:


#coeffiecient of variation of Infosys dataset
cv1=np.std(dataset['Open'])/np.mean(dataset['Open'])*100
cv1


# In[9]:


#coeffiecient of variation of TCS dataset
cv2=np.std(dataset2['Open'])/np.mean(dataset2['Open'])*100
cv2


# In[10]:


df=pd.DataFrame({"HL":(dataset['High']+dataset['Low'])/2})
df


# In[11]:


cv1=np.std(df['HL'])/np.mean(df['HL'])
cv1*100


# In[12]:


plt.figure(figsize=(15,5))
plt.plot(df['HL'])
plt.xticks(np.arange(0,241,25),dataset['Date'][0:241:25])


# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error



# In[12]:


dataset.isnull().any()


# In[13]:


plt.figure(figsize=(5,5))#to set the size of figure
lag_plot(dataset['Open'],lag=5)
plt.title("Infosys stock data")


# In[14]:


a=int(len(dataset)*0.8)
a


# In[15]:


train_set,test_set=dataset[0:a],dataset[a:]
train_set


# In[16]:


plt.figure(figsize=(15,5))
plt.title("Infosys data ")
plt.xlabel("Date")
plt.ylabel("stock open data")
plt.plot(train_set['Open'],color='blue',label='Training data')
plt.plot(test_set['Open'],color='green',label='Testing data')
plt.legend()
plt.xticks(np.arange(0,241,25),dataset['Date'][0:241:25])


# In[17]:


def smape(y_true,y_pred):
    return np.mean((np.abs(y_true-y_pred)*200/(np.abs(y_true)+np.abs(y_pred))))


# In[18]:


train_val=train_set['Open'].values
test_val=test_set['Open'].values
history=[x for x in train_val]
print(type(history))#this is list of training data
prediction=list()
prediction
for t in range(len(test_val)):
    model=ARIMA(history,order=(3,1,0))
    model_fit=model.fit(disp=0)
    output=model_fit.forecast()
    yhat=output[0]
    prediction.append(yhat)
    obs=test_val[t]
    history.append(obs)
error=mean_squared_error(test_val,prediction)
print("Mean squared error : %0.3f",error)
error2=smape(test_val,prediction)
print("Symmetric mean absolute percentage error: %0.3f",error2)


# In[19]:


print('Testing Mean Squared Error: %.3f' % error)
print("Symmetric mean absolute percentage error: %0.3f" %error2)


# In[20]:


df1=pd.DataFrame({'Actual':test_val,'Predicted':prediction})
df1


# In[21]:


plt.figure(figsize=(10,5))
plt.title("Observed vs Prediction")
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.plot(dataset['Open'],color='blue',label='Training data')
plt.plot(test_set.index,test_set['Open'],'green',marker='o',label='Testing Value')#since training and testing was overlapping therfore marker was used
plt.plot(test_set.index,prediction,'red',label='predicted value')
plt.xticks(np.arange(0,241,25), dataset['Date'][0:241:25])
plt.legend()


# In[22]:


n=int(len(df1))
242-193


# In[26]:


plt.figure(figsize=(7,5))
plt.title("Plot of observed vs predicted")
plt.xlabel("Date")
plt.ylabel('Open Price')
plt.plot(test_set.index,test_val,"green",label="observed",marker='o')
plt.plot(test_set.index,prediction,"red",label="predicted")
plt.xticks(np.arange(193,242,10),dataset['Date'][193:242:10])
plt.legend()


# In[39]:


model_fit.summary()


# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[50]:


#dataset['Date']=pd.to_datetime(dataset.Date,format='%m/%d/%Y')
date_val=dataset.index
#df.index=df['Date']
dataset.sort_index(ascending=True,axis=0)
data=pd.DataFrame({"Date":date_val,'Open':dataset['Open']})


# In[51]:


dataset


# In[111]:


df2 = dataset.reset_index()
prices = df2['Open'].tolist()
dates = df2.index.tolist()
 
prices1=df2['Open'].values.reshape(-1,1) 
dates1=df2.index.values.reshape(-1,1)
prices1
    
    
    
#Convert to 1d Vector
#dates = np.reshape(dates, (len(dates), 1))
#prices = np.reshape(prices, (len(prices), 1))
regressor = LinearRegression()
regressor.fit(dates1, prices1)


# In[112]:


plt.scatter(dates1, prices1, color='yellow', label= 'Actual Price') #plotting the initial datapoints
plt.plot(dates1, regressor.predict(dates1), color='red', linewidth=3, label = 'Predicted Price') #plotting the line made by linear regression
plt.title('Linear Regression | Time vs. Price')
plt.legend()
plt.xlabel('Date Integer')
plt.show()
 


# In[114]:


#Predict Price on Given Date
date = 10
#predicted_price =regressor.predict(date)
#print(predicted_price[0][0],regressor.coef_[0][0] ,regressor.intercept_[0])
#xtrain, xtest, ytrain, ytest = train_test_split(dates, prices, test_size=0.2, random_state=42)
xtrain=dates1[0:a]
ytrain=prices1[0:a]
xtest=dates1[a:]
ytest=prices1[a:]
regressor.fit(xtrain, ytrain)
 


# In[115]:


#Test Set Graph
plt.scatter(xtest, ytest, color='yellow', label= 'Actual Price') #plotting the initial datapoints
plt.plot(xtest, regressor.predict(xtest), color='blue', linewidth=3, label = 'Predicted Price') #plotting the line made by linear regression
plt.title('Linear Regression | Time vs. Price')
plt.legend()
plt.xlabel('Date Integer')
plt.show()


# In[116]:


erro=metrics.mean_squared_error(ytest,regressor.predict(xtest))
erro


# In[117]:


plt.figure(figsize=(20,5))
plt.title("Linear Regression Model")
plt.xlabel("Date")
plt.ylabel("Open Price")
plt.plot(ytrain,"blue",label="Training data")
plt.plot(test_set.index,ytest,"green",label="testing data")
#plt.plot(test_set.index,y_pred,"red")
plt.plot(test_set.index,regressor.predict(xtest),"brown",label="predicted value")
#plt.xticks(dataset['Date'][0:242:20])
plt.legend()


# In[ ]:





# In[31]:


#now we will split data into training and testing data set
#80% training rest 20% testing
a=int(len(dataset)*0.8)
train=data[0:a]
test=data[a:]
test


# In[33]:


x_train=train['Date'].values.reshape(-1,1)
y_train=train['Open'].values.reshape(-1,1)
x_test=test['Date'].values.reshape(-1,1)
y_test=test['Open'].values.reshape(-1,1)
plt.scatter(x_train,y_train)
#prices = df['Close'].tolist()
#dates = df.index.tolist()
#x_train = np.reshape(x_train, (len(dates), 1))
#y_train = np.reshape(prices, (len(prices), 1))


# In[76]:


model=LinearRegression()
dates=data['Date'].values.reshape(-1,1)
Open=data['Open'].values.reshape(-1,1)
model.fit(dates,Open)
#model.fit(x_train,y_train)
y_pred=model.predict(x_test)
#df1=pd.DataFrame(y_test,y_pred)
#df1


# In[77]:


df1=pd.DataFrame({'Actual':y_test.flatten(),'Predicted':y_pred.flatten()})
#df1=pd.DataFrame({'Actual':test_val,'Predicted':prediction})
df1


# In[38]:


plt.plot(x_test,y_test,color='grey')
plt.plot(x_test,y_pred,color='red',marker='o')
plt.plot(x_test,model.predict(x_test),color='yellow')


# In[54]:


dataset['Date']


# In[74]:


plt.figure(figsize=(15,5))
plt.title("Linear Regression Model")
plt.xlabel("Date")
plt.ylabel("Open Price")
plt.plot(train.index,y_train,"blue",label='Training data')
plt.plot(test.index,y_test,"green",label='Testing data')
plt.plot(test.index,y_pred,"red",label='Predicted data')
plt.xticks(np.arange(0,242,25),dataset['Date'][0:242:25])
#plt.plot(test.index,regressor.predict(x_test),"yellow")
plt.legend()


# In[83]:


r_sq = model.score(x_train, y_train)
print('coefficient of determination:', r_sq)
print('Slope of model: ',model.coef_)
print('Intercept of model: ',model.intercept_)


# In[57]:


err=metrics.mean_squared_error(y_test,y_pred)
print("Mean squared error:  ",err)


# In[85]:


# k nearest neighbour
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))


# In[86]:


dataset['Date']=pd.to_datetime(dataset.Date,format='%m-%d-%Y')
dataset["Date"]=pd.to_datetime(dataset.Date,format="%m-%d-%Y")
dataset=dataset.sort_index(ascending=True,axis=0)
data=pd.DataFrame({"Date":dataset.index,'Open':dataset['Open']})
data


# In[60]:


train=data[0:int(len(data)*0.8)]
test=data[int(len(data)*0.8):]
x_train=train['Date'].values.reshape(-1,1)
y_train=train['Open'].values.reshape(-1,1)
x_test=test['Date'].values.reshape(-1,1)
y_test=test['Open'].values.reshape(-1,1)
plt.scatter(x_train,y_train)


# In[61]:


x_train_scaled=scaler.fit_transform(x_train)
xtrain=pd.DataFrame(x_train_scaled)
x_test_scaled=scaler.fit_transform(x_test)
xtest=pd.DataFrame(x_test_scaled)


# In[62]:


# using gridsearch to find the best value of k

params = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

# fitting the model and predicting
model.fit(xtrain, y_train)
preds = model.predict(xtest)


# In[71]:


plt.figure(figsize=(15,5))
plt.title("K nearest neighbour Model")
plt.xlabel("Date")
plt.ylabel("Open Price")
plt.plot(train.index,y_train,"blue",label='Training data')
plt.plot(test.index,y_test,"green",label='testing data')
plt.plot(test.index,preds,"red",label='Predicted data')
plt.xticks(np.arange(0,242,20),dataset['Date'][0:242:20])
plt.legend()


# In[126]:


err=metrics.mean_squared_error(y_test,preds)
errr2=metrics.mean_squared_error(y_train,model.predict(xtrain))
print("Testing error",err)
print("Training error",errr2)


# In[127]:


#error values
xhat=model.predict(xtrain)
dat2=pd.DataFrame({'Actual':y_train.flatten(),'Predicted':xhat.flatten()})#on testing
dat1=pd.DataFrame({'Actual':y_test.flatten(),'Predicted':preds.flatten()})#on testing
dat2

