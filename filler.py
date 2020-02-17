import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

d = pd.read_csv('health.csv')

dd=d.drop(columns=['date','bool_of_active','weight_kg'])

active=pd.DataFrame()
ac=np.array(d['bool_of_active'])
mo=np.array(d['mood'])
active['active']=ac
active['mood']=mo


df=dd.assign(mood=dd['mood']/100)

y=df['mood']
x=df.drop(columns=['mood'])
 x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()
nb.fit()
nb.fit(x_train,y_train)
y_pre=nb.predict(x_test)


pickle.dump(nb, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))


