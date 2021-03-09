import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
import math

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import datetime

from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import csv
from sklearn.metrics import f1_score

#import data...
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
y_train = df_train[['PAX']]


#print(np.shape(df_train))

# We reduce the dimentions here...
print('Pre-proccessing begins...')
print('- Dimentionality Reduction in progress...')

components = ['DateOfDeparture','Departure','CityDeparture','LongitudeDeparture','LatitudeDeparture','Arrival','CityArrival','LongitudeArrival','LatitudeArrival','WeeksToDeparture','std_wtd','PAX']

def normalize(data, e) :
	outp = []
	m = data.mean()
	v = 0
	for d in data :
		v = v + ((d-m)**2)
	v = v/(len(data) - 1)
	for d in data :
		temp = (d-m)/(math.sqrt(v) + e)
		outp.append(int(round(temp)))

	return outp
	
weekdaytable = []
distance = []
month_table = []

for x in range(len(df_train)) : 
	temp = df_train[components[0]][x]
	temp = temp+' 00:00:0.0' 
	date_time_obj = datetime.datetime.strptime(temp, '%Y-%m-%d %H:%M:%S.%f')
	weekdaytable.append(date_time_obj.weekday())
	month_table.append(date_time_obj.month)

	x1 = df_train[components[3]][x]
	y1 = df_train[components[4]][x]
	x2 = df_train[components[7]][x]
	y2 = df_train[components[8]][x]
	distance.append(math.sqrt( (x2-x1)**2 + (y2-y1)**2 ))

test_weekdaytable = []
test_distance = []
test_month_table = []

for x in range(len(df_test)) :
	temp = df_test[components[0]][x]
	temp = temp+' 00:00:0.0' 
	date_time_obj = datetime.datetime.strptime(temp, '%Y-%m-%d %H:%M:%S.%f')
	test_weekdaytable.append(date_time_obj.weekday())
	test_month_table.append(date_time_obj.month)

	x1 = df_test[components[3]][x]
	y1 = df_test[components[4]][x]
	x2 = df_test[components[7]][x]
	y2 = df_test[components[8]][x]
	test_distance.append(math.sqrt( (x2-x1)**2 + (y2-y1)**2 ))
	

# LAbel encoding...
# we use encoder in order to exchange the String values into ints...

#This part might get replaces with keras's one

le = LabelEncoder()
le.fit(df_train[components[0]])
df_train[components[0]] = le.transform(df_train[components[0]])
df_test[components[0]] = le.transform(df_test[components[0]])

le.fit(df_train[components[1]])
df_train[components[1]] = le.transform(df_train[components[1]])
df_test[components[1]] = le.transform(df_test[components[1]])

le.fit(df_train[components[2]])
df_train[components[2]] = le.transform(df_train[components[2]])
df_test[components[2]] = le.transform(df_test[components[2]])

le.fit(df_train[components[5]])
df_train[components[5]] = le.transform(df_train[components[5]])
df_test[components[5]] = le.transform(df_test[components[5]])

le.fit(df_train[components[6]])
df_train[components[6]] = le.transform(df_train[components[6]])
df_test[components[6]] = le.transform(df_test[components[6]])

#end_part

df_train.drop(df_train.columns[[11]], axis=1, inplace=True) # Removing the PAX...

norm_std_wtd = normalize(df_train[components[10]], 0)
test_norm_std_wtd = normalize(df_test[components[10]], 0)


df1 = pd.DataFrame({'WeekDay':weekdaytable})
df2 = pd.DataFrame({'Distance':distance})

df3 = pd.DataFrame({'WeekDay':test_weekdaytable})
df4 = pd.DataFrame({'Distance':test_distance})

df5 = pd.DataFrame({'Month':month_table})
df6 = pd.DataFrame({'Month':test_month_table})

df7 = pd.DataFrame({'NormSTD':norm_std_wtd})
df8 = pd.DataFrame({'NormSTD':test_norm_std_wtd})


df_train = pd.concat([df_train, df1], axis=1)
df_train = pd.concat([df_train, df2], axis=1)

df_test = pd.concat([df_test, df3], axis=1)
df_test = pd.concat([df_test, df4], axis=1)

df_train = pd.concat([df_train, df5], axis=1)
df_test = pd.concat([df_test, df6], axis=1)

df_train = pd.concat([df_train, df7], axis=1)
df_test = pd.concat([df_test, df8], axis=1)	
	
# start training	
import keras
from keras import backend as K
from keras import optimizers
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.metrics import categorical_crossentropy

attributes = ['DateOfDeparture','LongitudeDeparture','LatitudeDeparture','Arrival','CityArrival','LongitudeArrival','std_wtd','WeekDay','Distance']


df_new = df_train[attributes]
# labels are located in the y_train

X = np.array(df_new)
Y = np.array(y_train)

scaler = MinMaxScaler(feature_range=(0,1))
X_scaled  = scaler.fit_transform(X)

model = Sequential([
	Dense(16, input_shape=(9,)),
	Dense(32),
	Dense(1)
])

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(X_scaled,Y, batch_size=5, epoch=50, verbose=2)
y_pred = model.predict(X_test, batch_size=5, steps=None)
	
with open('y_pred.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['Id', 'Label'])
    for i in range(y_pred.shape[0]):
        writer.writerow([i, y_pred[i]])
	
	
	
	
