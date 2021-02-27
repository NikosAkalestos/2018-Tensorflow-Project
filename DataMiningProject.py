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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import csv
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import copy
from itertools import combinations

#Import data and libraries...
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
y_train = df_train[['PAX']]

print('Pre-proccessing begins...')
# --------------- In this section new attributes are created from the existing ones, in order to select the best combination of them for the classification. ----------------
print('- Attribute selection begins...')

# This is the list of the starting attributes.
components = ['DateOfDeparture','Departure','CityDeparture','LongitudeDeparture','LatitudeDeparture','Arrival','CityArrival','LongitudeArrival','LatitudeArrival','WeeksToDeparture','std_wtd','PAX']

# These are some support methods used.
def normalize(data, e) : # This is used in order to normalize attributes...
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

def normalizeOutput(x) : # This is CURRENTLY used as a method to convert our float output o int...

	f = float(x)
	if f < 0:
		return 0
	if f > 7:
		return 7
	temp = round(f,1)
	return int(temp)

def remove_from(a,b): # from a remove b
    return list(set(a)-set(b))

# Here the distance between the airports is calculated: d = squareroot( (x2-x1)^2 + (y2-y1)^2 )
# and we extract information about teh weekday and month from the given date.
# It is done for both the training and test data frame.

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

# Label encoding...
# LabelEncoder is used in order to exchange the String values into numbers...

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

df_train.drop(df_train.columns[[11]], axis=1, inplace=True) # Removing the PAX from the training set.

# Here information about the season is extracted from the month.
seasons = []
for month in month_table:
	if month in [1, 2, 12]:
		seasons.append(0) # winter
	elif month in [3, 4, 5]:
		seasons.append(1) # spring
	elif month in [6, 7, 8]:
		seasons.append(2) # summer
	elif month in [9, 10, 11]:
		seasons.append(3) # fall

test_seasons = []
for month in test_month_table:
	if month in [1, 2, 12]:
		test_seasons.append(0) # winter
	elif month in [3, 4, 5]:
		test_seasons.append(1) # spring
	elif month in [6, 7, 8]:
		test_seasons.append(2) # summer
	elif month in [9, 10, 11]:
		test_seasons.append(3) # fall

# Here 3 new attributes are created that combine WeeksToDeparture and std_wtd.
a_wtd = []
t = 0
for wtd in df_train[components[9]]:
	if (df_train[components[10]][t] == 0):
		a_wtd.append(wtd)
	else :
		a_wtd.append(wtd/df_train[components[10]][t])
	t = t + 1

test_a_wtd = []
t = 0
for wtd in df_test[components[9]]:
	if (df_test[components[10]][t] == 0):
		test_a_wtd.append(wtd)
	else :
		test_a_wtd.append(wtd/df_test[components[10]][t])
	t = t + 1
max_wtd = []
t = 0
for wtd in df_train[components[9]]:
	max_wtd.append(wtd+df_train[components[10]][t])
	t = t + 1 
test_max_wtd = []
t = 0
for wtd in df_test[components[9]]:
	test_max_wtd.append(wtd+df_test[components[10]][t])
	t = t + 1 

min_wtd = []
t = 0
for wtd in df_train[components[9]]:
	min_wtd.append(wtd-df_train[components[10]][t])
	t = t + 1 
test_min_wtd = []
t = 0
for wtd in df_test[components[9]]:
	test_min_wtd.append(wtd-df_test[components[10]][t])
	t = t + 1 

# Here the lists are converted into Data Frames
df1 = pd.DataFrame({'WeekDay':weekdaytable})
df2 = pd.DataFrame({'Distance':distance})

df3 = pd.DataFrame({'WeekDay':test_weekdaytable})
df4 = pd.DataFrame({'Distance':test_distance})

df5 = pd.DataFrame({'Month':month_table})
df6 = pd.DataFrame({'Month':test_month_table})

df7 = pd.DataFrame({'Season':seasons})
df8 = pd.DataFrame({'Season':test_seasons})

df9 = pd.DataFrame({'a_wtd':a_wtd})
df10 = pd.DataFrame({'a_wtd':test_a_wtd})

df11 = pd.DataFrame({'Max_wtd':max_wtd})
df12 = pd.DataFrame({'Max_wtd':test_max_wtd})

df13 = pd.DataFrame({'Min_wtd':min_wtd})
df14 = pd.DataFrame({'Min_wtd':test_min_wtd})

# Here those data frames are added into the train and test set.
df_train = pd.concat([df_train, df1], axis=1)
df_train = pd.concat([df_train, df2], axis=1)

df_test = pd.concat([df_test, df3], axis=1)
df_test = pd.concat([df_test, df4], axis=1)

df_train = pd.concat([df_train, df5], axis=1)
df_test = pd.concat([df_test, df6], axis=1)

df_train = pd.concat([df_train, df7], axis=1)
df_test = pd.concat([df_test, df8], axis=1)

df_train = pd.concat([df_train, df9], axis=1)
df_test = pd.concat([df_test, df10], axis=1)

df_train = pd.concat([df_train, df11], axis=1)
df_test = pd.concat([df_test, df12], axis=1)

df_train = pd.concat([df_train, df13], axis=1)
df_test = pd.concat([df_test, df14], axis=1)

# Now all attributes are normalized
# and added into the train and test set.
attributes = list(df_train.columns.values) # This list contains the names of all attributes
for i in range(len(attributes)):
	if (attributes[i] not in ['DateOfDeparture','WeekDay','Month','Season', 'a_wtd']) :
		train_temp = normalize(df_train[attributes[i]], 0)
		test_temp = normalize(df_test[attributes[i]], 0)

		train_temp_df = pd.DataFrame({('Norm_'+ attributes[i]):train_temp})
		test_temp_df = pd.DataFrame({('Norm_'+ attributes[i]):test_temp})

		df_train = pd.concat([df_train, train_temp_df], axis=1)
		df_test = pd.concat([df_test, test_temp_df], axis=1)

attributes = list(df_train.columns.values) # update the attributes
print('Starting with: ', len(attributes), ' attributes, 11 given and ', len(attributes)-11,', custom.')
# --------------- In this section new attributes are created from the existing ones, in order to select the best combination of them for the classification. ----------------

# Attribute selection...

# --------------- In this section all attributes are evaluated via a custom "trial and error" method in order to select the best combination of them for our classification. ----------------

# There were 2 different methods created for this section, as they have a different approach
# Please note that they are all commented out, since the attribute selection has already been done and the best attributes for the classifier have been selected
# they also would add a considerable run time...

# METHOD NUMBER 1: In short, a loop removes the "worst" attribute each time, keeping the ones that give the best score.
# In this method the starting attributes are evaluated using the cross_val_score from sklearn, the attributes and their score are kept in best_attrs and best_score.
# Then a loop is run removing each time an attribute and evaluating the performance again using cross_val_score from sklearn, for all attributes.
# Then if the removal of an attribute did not improve the score, all attributes are kept and the program proceeds, else:
# the best score and new attributes (containing 1 less attribute) are kept and the method loops again using the new attributes (best_attrs) as the starting ones
# until the removal of an attribute would not improve the score any more.
# Finally, the program proceeds with the best_attrs as its new reduced attributes.

'''
y = np.ravel(y_train)
rdf = RandomForestClassifier(max_features = 6, max_depth = 13, n_estimators = 111, random_state = 1, criterion='entropy', n_jobs = -1)

# Note that in this section you can reduce the amount of attributes you give to the method by changing the attributes list... 
# It is highly adviced to give fewer attributes to evaluate, as not all of them are usefull.
attributes = list(df_train.columns.values)
#attributes = ['DateOfDeparture','CityDeparture','Norm_LongitudeDeparture','Norm_LatitudeDeparture','Arrival','Norm_LongitudeArrival','LatitudeArrival','WeekDay', 'Distance', 'Month']

print('-> Starting attributes: ')
print(attributes)

flag = 1
removed_attrs = []
while (flag == 1):
	print('With all attributes: ')
	X = df_train[attributes]
	scores = cross_val_score(rdf, X, y, cv=10, scoring='f1_micro', n_jobs = -1)
	best_score = scores.mean()
	best_attrs = copy.deepcopy(attributes)
	best_run = -1

	print('-> Score: ', scores.mean())
	print('----------------------------')

	num_of_attrs = len(attributes)
	for i in range(num_of_attrs) :
		attrs = copy.deepcopy(attributes)
		print('- We remove: ', attrs[i])
		del attrs[i]	
		X = df_train[attrs]
		scores = cross_val_score(rdf, X, y, cv=10, scoring='f1_micro', n_jobs = -1)
		score = scores.mean()
		print('-> Score:', score)
		print('----------------------------')
		if score > best_score :
			best_score = score
			best_attrs = copy.deepcopy(attrs)
			best_run = i
	
	print('----- Run COMPLETED -----')	
	print('SCORE? ATTRS? WasDeleted?')
	if (best_run == -1) : 
		print('---> All attributes are important!')
		break
	else :
		print('Best score was: ', best_score)
		print('Best attributes: ')
		print(best_attrs)
		print('Attribute that we removed this run: ')
		print(attributes[best_run])
		removed_attrs.append(attributes[best_run])
		attributes = copy.deepcopy(best_attrs)
		print('----------------------------')

print('---> Final attributes: ')
print(attributes)
print('---> Removed attributes: ')
print(removed_attrs)

'''
# METHOD NUMBER 2: In short, a loop checks the score of every possible combination of attributes and keeps the best one, as well as a top 10.
# In this method the starting attributes are evaluated using the cross_val_score from sklearn, the attributes and their score are kept in best_attrs and best_score.
# Then a loop is run removing each time a combination of attributes and evaluating the performance again using cross_val_score from sklearn, for all remaining attributes.
# Until all combinations of attributes are checked.
# Finally, the program proceeds with the best_attrs as its new reduced attributes.
# It also prints a top 10 best combinations of attributes.
# This method has a massive run time, but by narrowing down your search (for instance by giving it less attributes and specifying the n number of attributes you want to remove) it can be reduced signifiacntly.
# This was the method used to get the final attributes, by giving it as starting attributes the ones that were commented out in line 357 (as they were the "best" subset from the starting attributes)
# n was set to 4 (14-4 = 10 attributes) and a break was added in line 426. (so that it would not check for more than combinations of 4 to remove...)
'''
# Note that in this section you can reduce the amount of attributes you give to the method by changing the attributes list... 
# It is highly adviced to give fewer attributes to evaluate, as not all of them are usefull.
attributes = list(df_train.columns.values)
#attributes = ['DateOfDeparture','CityDeparture','Norm_LongitudeDeparture','Norm_LatitudeDeparture','Arrival','Norm_LongitudeArrival','LatitudeArrival','Norm_LatitudeArrival','WeekDay', 'Distance', 'Month', 'Season', 'Max_wtd', 'Min_wtd']

y = np.ravel(y_train)
rdf = RandomForestClassifier(max_features = 6, max_depth = 13, n_estimators = 111, random_state = 1, criterion='entropy', n_jobs = -1)

print('Starting attributes: ')
print(attributes)

flag = 1
removed_attrs = []
print('With all attributes: ')
X = df_train[attributes]
scores = cross_val_score(rdf, X, y, cv=10, scoring='f1_micro', n_jobs = -1)
best_score = scores.mean()
best_attrs = copy.deepcopy(attributes)
best_run = -1

print('-> Score: ', scores.mean())
print('----------------------------')

num_of_attrs = len(attributes)
it_through = []
top_ten_scores = [0,0,0,0,0,0,0,0,0,0]
top_ten_attrs = [list(),list(),list(),list(),list(),list(),list(),list(),list(),list()]
top_ten_removed_attrs = [list(),list(),list(),list(),list(),list(),list(),list(),list(),list()]
n = 1 # the starting value of the number of combinations to try to remove (n = START, n++) -----------------------------------
# It is highly addviced to change the value of n to either 9, 10, 11 or 12 as it has been observed that around 10 attributes are ideal for our classifier.
while (n < num_of_attrs):
	for combo in combinations(attributes, n):  # n = 2 for pairs, 3 for triplets, etc
		print('- We remove: ',combo)
		rest = remove_from(attributes,combo)
		X = df_train[rest]
		col_vals = sorted(X.columns.values)
		if col_vals not in it_through : # This step makes sure that the same combo of attributes is not checked more than once.
			it_through.append(col_vals)
			scores = cross_val_score(rdf, X, y, cv=10, scoring='f1_micro', n_jobs = -1)
			score = scores.mean()
			print('-> Score:', score)
			print('----------------------------')
			if score > best_score :
				best_score = score
				best_attrs = copy.deepcopy(rest)
				removed_attrs = copy.deepcopy(combo)
			for i in range(10):
				if score > top_ten_scores[i]:
					top_ten_scores[i] = score
					top_ten_attrs[i] = copy.deepcopy(rest)
					top_ten_removed_attrs[i] = copy.deepcopy(combo)
					break
	

	print('------>> RUN FOR n = ', n, ' completed! <<----------')
	print('Best score was: ', best_score, ' so far...')
	print('Best attributes: ')
	print(best_attrs)
	print('Attribute that we removed this run: ')
	print(removed_attrs)
	print('----------------------------')

	print('NOW PRESENTING TOP 10!') # A top 10 is kept as a combination of attributes might have a worse score with the train data and a better one with the test data.
	for i in range(10):
		print('-------------- pos ',i,'--------------')
		print('SCORE : ', top_ten_scores[i])
		print('attributes: ')
		print(top_ten_attrs[i])
		print('attributes removed:')
		print(top_ten_removed_attrs[i])
		print('----------------------------')
	n = n + 1
	break # COMMENT THIS OUT FOR A FULL SCALE RUN! (from n to number of attributes)  (NOT RECOMMENDED)-------------------------------------------------

print(' ------------------------ FININSHED ------------------------')
print('Best score was: ', best_score)
print('Best attributes: ')
print(best_attrs)
print('Attribute that we removed this run: ')
print(removed_attrs)
attributes = copy.deepcopy(best_attrs)
print('----------------------------')

print('NOW PRESENTING TOP 10!')
for i in range(10):
	print('-------------- pos ',i,'--------------')
	print('SCORE : ', top_ten_scores[i])
	print('attributes: ')
	print(top_ten_attrs[i])
	print('attributes removed:')
	print(top_ten_removed_attrs[i])
	print('----------------------------')

print('---> Final attributes: ')
print(attributes)
print('---> Removed attributes: ')
print(removed_attrs)
'''

# --------------- In this section all attributes are evaluated via a custom "trial and error" method in order to select the best combination of them for our classification. ----------------
print('- Attribute selection DONE')

# PARAM SELECTION
print('- PARAM SELECTION begins!')

# --------------- In this section the parameters of the selected classifier are evaluated via a custom "trial and error" method in order to select the best combination of them for our classification. ----------------
'''
# Please note that this part is also commented out, since the parameter selection has already been done and the best params for the classifier have been selected.
# This also would add a considerable run time, depending on the params you select to fine tune...

# METHOD: 
# First of all, you select the classifier you wish to fine tune.
# Then you select which parameters you want to check and specify the range of the search. The more you add, the longer it takes to run.
# Then you create a parameter grid and add them to it.
# GridSearchCV is used with the given grid and the classifier.
# In the end the best parameters, as well as the best score are printed.

# In this method the starting attributes are evaluated using the cross_val_score from sklearn, the attributes and their score are kept in best_attrs and best_score.
# Then a loop is run removing each time an attribute and evaluating the performance again using cross_val_score from sklearn, for all attributes.
# Then if the removal of an attribute did not improve the score, all attributes are kept and the program proceeds, else:
# the best score and new attributes (containing 1 less attribute) are kept and the method loops again using the new attributes (best_attrs) as the starting ones
# until the removal of an attribute would not improve the score any more.
# Finally, the program proceeds with the best_attrs as its new reduced attributes.

# Here you can select the classifier you want to fine tune.
clf = RandomForestClassifier(max_features = 6, max_depth = 13, n_estimators = 111, random_state = 1, criterion='entropy', n_jobs = -1)
#clf = MLPClassifier(solver='adam', alpha=1e-5,activation = 'logistic')

# Here you select which parameters you want to check and specify the range.
# These change along with the classifier...

#solv = ['sgd','lbfgs','adam']
#hls = range(500,1550,50)
#act = ['identity', 'logistic', 'tanh', 'relu']
#min_ss = range(2,9)
#rs = range(1,10)
#max_ln = range(2,30)
n_range = range(100,130)
max_d = range(5,20)
max_feats = range(3,8)

# These are the "best attributes" for RandomForestClassifier, they are to be commented out if you want to give the ones that the attribute selection section has found as "best".
attributes = ['DateOfDeparture','CityDeparture','Norm_LongitudeDeparture','Norm_LatitudeDeparture','Arrival','Norm_LongitudeArrival','LatitudeArrival','WeekDay', 'Distance', 'Month']

# Here you create the parameter grid, you chose which attributes to fine tune.

#param_grid = dict(rdf_weight = w1, ada_weight = w2, gnb_weight = w3)
#param_grid = dict(hidden_layer_sizes= hls)
param_grid = dict(max_features = max_feats, max_depth = max_d, n_estimators = n_range)

grid = GridSearchCV(clf,param_grid,cv=10,scoring='f1_micro', n_jobs = -1)

X = df_train[attributes]
y = np.ravel(y_train)

grid.fit(X,y) # This may be fast or take ages to finish, depending on your input.

print('DONE')
print('BEST SCORE // BEST PARAMS // BEST ESTIMATOR')
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
'''

# --------------- In this section the parameters of the selected classifier are evaluated via a custom "trial and error" method in order to select the best combination of them for our classification. ----------------
print('- PARAM SELECTION DONE')

print('Pre-proccessing DONE')

# Training...
print('Training begins!')
# --------------- In this section the "best" attributes and parameters are used in order to train the selected classifier. ----------------

#attributes = list(df_train.columns.values)
# These are the "best" attributes for the selected classifier RandomForest.
attributes = ['DateOfDeparture','CityDeparture','Norm_LongitudeDeparture','Norm_LatitudeDeparture','Arrival','Norm_LongitudeArrival','LatitudeArrival','WeekDay', 'Distance', 'Month']

# The training and test data are finally selected.
X_train = df_train[attributes]
X_test = df_test[attributes]
y_train = np.ravel(y_train)

# Here the classifier is selected (RandomForest), along with the "best" parameters.
#clf = ExtraTreesClassifier(n_estimators=75, max_depth=12, min_samples_split=2, random_state=0)
#clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=34)
#clf = MLPClassifier(solver = 'adam', alpha = 1e-5, activation = 'logistic', hidden_layer_sizes = 1100)
#clf = KNeighborsClassifier(n_neighbors=5)
#clf = SVC(gamma='auto')
#clf = GaussianNB()
#clf = QuadraticDiscriminantAnalysis()
#clf = AdaBoostClassifier(clf,n_estimators=34)
clf = RandomForestClassifier(max_features = 6, max_depth = 13, n_estimators = 111, random_state = 1, criterion='entropy', n_jobs = -1) 

# Fit the classifier on the train data.
clf.fit(X_train,y_train) 

print('Training - DONE - ')
# --------------- In this section the "best" attributes and parameters are used in order to train the selected classifier. ----------------

# --------------- In this section the trained classifier is used in order to make a prediction on the test data. ----------------
y_pred = clf.predict(X_test)

with open('y_pred.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['Id', 'Label'])
    for i in range(y_pred.shape[0]):
        writer.writerow([i, y_pred[i]])

print('We made a prediction!')
print('Prediction was saved in "y_pred.csv".')

# --------------- THE END ----------------