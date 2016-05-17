#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt 
sys.path.append("E:\\version-control\\ud120-projects\\tools\\")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import cohen_kappa_score, make_scorer
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data,test_classifier
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from collections import OrderedDict
from operator import itemgetter
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Get a list of all the features in this data set. 
for k,v in data_dict.items():
    total_features=v.keys()
    break 

### Get an idear about the two classes: how many are pois  
poi_list =[]
for k,v in data_dict.items():
    if v['poi'] is True:
        poi_list.append(k)

print poi_list
print len(poi_list)


###Are there features with many missing values? etc.
###Define a function to count the # of NaN values for a specific feature in a dataset.
def count_na(feature,data_dict):    
    count = 0
    for k,v in data_dict.items():
        if v[feature]=="NaN":
            count +=1
    return count 

###Define a function to count the # of NaN values for a specific feature for the 
###person of interest class 
def count_na_in_poi(feature,data_dict):    
    count = 0
    for k,v in data_dict.items():
        if v["poi"] is True:
            if v[feature]=="NaN":
                count +=1
    return count 

###Get the # of "NaN" values for a specific feature in this data set
for i in total_features: 
    print i+" has # of NA Values:",count_na(i,data_dict)
    
###Get the # of "NaN" values for a specific feature of the poi class
for i in total_features:     
    print i+" has # of NA Values:",count_na_in_poi(i,data_dict)
    
### If more than 90% the datapoints have no information for this feature,
### and also since there are only 18 pois, if more than half of the pois have 
### "NaN" values for this feature, then this feature will not be considered. 
### So email_address, loan advances, deferral_payments, director_fees

features_list = ['poi', 'to_messages', 'expenses', 'deferred_income', 
'long_term_incentive', 'fraction_from_poi', 'shared_receipt_with_poi', 
 'from_messages', 'bonus', 'total_stock_value', 'from_poi_to_this_person', 
 'from_this_person_to_poi', 'restricted_stock', 'salary', 'total_payments',
 'fraction_to_poi',  'exercised_stock_options']

###############################################################################
### Task 2: Remove outliers
# Look for outliers points by salary and bonus values
features = ["salary","bonus","poi"]
data = featureFormat(data_dict, features)
max_salary = 0
max_bonus = 0
for point in data:
    salary = point[0]
    bonus = point[1]
    poi = point[2]
    if poi:
        plt.scatter(salary, bonus,c="r")
    else:
        plt.scatter(salary, bonus)
    if point[0] > max_salary:
        max_salary = point[0]
    if point[1] > max_bonus:
        max_bonus = point[1]
plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()
print max_salary
print max_bonus

### From the scatter plot, we see there is one point really far away from the
### cluster, I wanna know what this data point is. 
 
for k,v in data_dict.items():
    if v['salary'] == max_salary:
        print "Top salary", k
    if v['bonus'] == max_bonus:
        print "Top bonus", k      

### The first outlier identified is a "TOTAL" point, not a real data point
### thus I will remove it from the dataset.
data_dict.pop("TOTAL",0)

### Remove data points have little information: most variables have "NaN" values,
### and they are not poi.

key_remove_list = ["THE TRAVEL AGENCY IN THE PARK","LOCKHART EUGENE E"]

for i in key_remove_list:
    data_dict.pop(i,0)

###############################################################################
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

features_from_to_poi = ["from_poi_to_this_person","from_this_person_to_poi","poi"]
data_from_to_poi = featureFormat(data_dict, features_from_to_poi)
for point in data_from_to_poi:
    from_poi_to_this_person = point[0]
    from_this_person_to_poi = point[1]
    poi = point[2]
    if poi:
        plt.scatter(from_poi_to_this_person, from_this_person_to_poi,c="r")
    else:
        plt.scatter(from_poi_to_this_person, from_this_person_to_poi)
plt.xlabel("from_poi_to_this_person")
plt.ylabel("from_this_person_to_poi")
plt.show()

# Create two new features in the dataset.
# Define a function to calculate the fraction, if there is any NaN value, 
# return NaN, if not return the fraction. 
def computeFraction( poi_messages, all_messages ):
    if poi_messages =="NaN" or all_messages =="NaN":
        return 0
    else:
        fraction = float(poi_messages)/float(all_messages)
        return fraction
        
for name in data_dict:
    data_point = data_dict[name]
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    # Create a feature to show the fraction of emails this person received from poi
    data_point["fraction_from_poi"] = fraction_from_poi
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages ) 
     # Create a feature to show the fraction of emails this person received from poi
    data_point["fraction_to_poi"] = fraction_to_poi
    if data_point['poi']:    
        plt.scatter(fraction_from_poi,fraction_to_poi,c="r")
    else:    
        plt.scatter(fraction_from_poi,fraction_to_poi)
plt.xlabel("fraction_from_poi")
plt.ylabel("fraction_to_poi")
plt.show()
# From the visualization,if fraction_to_poi less than 0.2,it is highly possible
# not a poi.

# Store to my_dataset
my_dataset = data_dict

for k,v in data_dict.items():
    total_features=v.keys()
    break 
print total_features



### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list,sort_keys = True)
labels, features = targetFeatureSplit(data)
### Build a pipeline 
scaler = MinMaxScaler()
skb = SelectKBest(f_classif)
gnb = GaussianNB()
pipeline = Pipeline(steps=[("scaling",scaler),("SKB", skb), ("NaiveBayes", gnb)])
SKB_params = {"SKB__k":range(1,10)}
cv = StratifiedShuffleSplit(labels,n_iter=100, random_state = 42)
# Use Kappa_scorer as a metric to evaluate 
kappa_scorer = make_scorer(cohen_kappa_score)
gs= GridSearchCV(pipeline,SKB_params,scoring=kappa_scorer,cv=cv)
gs.fit(features,labels)
print "best # of parameters to choose:", gs.best_params_
clf = gs.best_estimator_
# Get the features selected by KBest 
clf.named_steps['SKB'].get_support(indices=True)
features_selected =[features_list[1:][i] for i in clf.named_steps['SKB'].get_support(indices=True)]
print features_selected 

feature_score = clf.named_steps['SKB'].scores_
score_summary = {}
for i in range(len(feature_score)):
    k = features_list[1:][i]
    v = feature_score[i]
    score_summary[k]=v   
print OrderedDict(sorted(score_summary.items(), key=itemgetter(1)))
    
features_list = ["poi"]+features_selected
print features_list
# Use the test_classifier to derive classification report
print " "
print "Tester Classification report:" 
print " "
test_classifier(clf, my_dataset, features_list)

 
###Intuitively, I think the two stock features may have interrelationship 
### with each other. So I create a scatter plot to visualize the relationship
### between those two features. 


stock_features= ['total_stock_value','exercised_stock_options',"poi"]
data_stock = featureFormat(data_dict, stock_features,sort_keys = True)
for point in data_stock:
    total_stock = point[0]
    exercised_stock = point[1]
    poi = point[2]
    if poi:
        plt.scatter(total_stock, exercised_stock,c="r")
    else:        
        plt.scatter(total_stock, exercised_stock)
plt.xlabel('total_stock_value')
plt.ylabel('exercised_stock_options')
plt.show()

###The scatter plot shows those two variables are highly correlated. By using 
### both of them, we gain a lot of repetitive information. Thus, conducting PCA
### to reduce the dimensions is very neccessary so that we could replace the 
### original features with an uncorrelated linear projection.  



###############################################################################
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

features_list = ['poi', 'to_messages', 'expenses', 'deferred_income', 
'long_term_incentive', 'fraction_from_poi', 'shared_receipt_with_poi', 
 'from_messages', 'bonus', 'total_stock_value', 'from_poi_to_this_person', 
 'from_this_person_to_poi', 'restricted_stock', 'salary', 'total_payments',
 'fraction_to_poi',  'exercised_stock_options']

### Using the selected features to preprocess the dataset 

data = featureFormat(my_dataset, features_list,sort_keys = True)
labels, features = targetFeatureSplit(data) 

### Using StratifiedShuffleSplit to provide train/test indices 
### Since the size of the data set is very small and the two classes are 
### very imblanced, this is the best way to split the dataset 
### Also, select a random state, so that the results are reproducible
sss = StratifiedShuffleSplit(labels,n_iter=100, random_state = 42)

### First classifier NaiveBayes.GausianNB, which achieved target results of 
### both precision & recall >0.3 
scaler = MinMaxScaler()
skb = SelectKBest(f_classif)
gnb = GaussianNB()
pca = PCA()
pca_params = {"PCA__n_components":[1, 2, 3, 4,5], "PCA__whiten": [True,False]}
kbest_params = {"SKB__k":range(5,17)}
pca_params.update(kbest_params)
pipeline = Pipeline(steps=[("scaling",scaler),("SKB", skb),("PCA", pca),("NaiveBayes", gnb)])
gs = GridSearchCV(pipeline,param_grid=pca_params,scoring=kappa_scorer,cv=sss)
gs.fit(features,labels)
print gs.best_params_
clf = gs.best_estimator_

# Check what features are selected 
clf.named_steps['SKB'].get_support(indices=True)
features_selected =[features_list[1:][i] for i in clf.named_steps['SKB'].get_support(indices=True)]
print features_selected 

# Use the test_classifier to derive classification report
print " "
print "Tester Classification report:" 
print " "
test_classifier(clf, my_dataset, features_list)
###The best set of parameters for GaussianNB() is K=11, n_components =5. 

###Try second classifier using Decision Tree
scaler = MinMaxScaler()
skb = SelectKBest(f_classif)
dt = tree.DecisionTreeClassifier()
pca = PCA()
pca_params = {"PCA__n_components":range(1,6), "PCA__whiten": [True,False]}
kbest_params = {"SKB__k":range(6,15)}
pca_params.update(kbest_params)
# Build a pipeline 
pipeline = Pipeline(steps=[("scaling",scaler),("SKB", skb),("PCA", pca),("DT", dt)])
gs = GridSearchCV(pipeline,param_grid=pca_params,scoring=kappa_scorer,cv=sss)
gs.fit(features,labels)
print gs.best_params_
clf = gs.best_estimator_
# Check what features are selected 
clf.named_steps['SKB'].get_support(indices=True)
features_selected =[features_list[1:][i] for i in 
clf.named_steps['SKB'].get_support(indices=True)]
print features_selected 

### The best parameter set is 
### {'PCA__n_components': 4, 'PCA__whiten': False, 'SKB__k': 7}

# Use the test_classifier to derive classification report
print " "
print "Tester Classification report:" 
print " "
test_classifier(clf, my_dataset, features_list)

### Accuracy: 0.81627 Precision: 0.28692 Recall: 0.25450 
### F1: 0.26974 F2: 0.26038

### Try third classifier using RandomForest (scaling is not
### neccessary for RandomForest) 

skb = SelectKBest()
rdf = RandomForestClassifier(random_state = 42)
pca = PCA()
pca_params = {"PCA__n_components":[2,3,4,5,6], "PCA__whiten": [True,False]}
kbest_params = {"SKB__k":range(6,15)}
pca_params.update(kbest_params)
pipeline = Pipeline(steps=[("SKB", skb),("PCA", pca),("Random", rdf)])
gs = GridSearchCV(pipeline,param_grid=pca_params,scoring=kappa_scorer,cv=sss)
gs.fit(features,labels)
print gs.best_params_
### The best set of paramters is 
### {'PCA__n_components': 2, 'PCA__whiten': True, 'SKB__k': 6}
clf = gs.best_estimator_
# Use the test_classifier to derive classification report
print " "
print "Tester Classification report:" 
print " "
test_classifier(clf, my_dataset, features_list)
### Performance result:
### Accuracy: 0.85527  Precision: 0.41028  Recall: 0.19550

### Try AdaBoost

scaler = MinMaxScaler()
features= scaler.fit_transform(features)
pca_params = {"PCA__n_components":[2,3,4,5,6], "PCA__whiten": [True,False]}
kbest_params = {"SKB__k":range(6,12)}
pca_params.update(kbest_params)
clf_adb = AdaBoostClassifier(random_state=42)
pipeline = Pipeline(steps=[("scaling",scaler),("SKB", skb),("PCA", pca),
                           ("ADB",clf_adb)])
gs = GridSearchCV(pipeline,param_grid=pca_params,scoring=kappa_scorer,cv=sss)
gs.fit(features,labels)
print gs.best_params_
### {'PCA__n_components': 2, 'PCA__whiten': True, 'SKB__k': 7}
clf = gs.best_estimator_
# Use the test_classifier to derive classification report
print " "
print "Tester Classification report:" 
print " "
test_classifier(clf, my_dataset, features_list)
### Performance result:
### Accuracy: 0.81240  Precision: 0.25333  Recall: 0.20900

###############################################################################

### Task 5: Tune your classifier to achieve better than .3 precision and recall 


### NaiveBayes and RandomForest are better performed algorithms to use.

### NaiveBayes has limited tuning options, in previous code, I've tuned 
### the model by setting a wide range of K values and n_components, 
### and then define the best set of parameters by using GridSearchCV. 
### Run the algorithem against the tester, achieved both precision and recall 
### >0.3. 

### So now I will focus on tuning the RandomForest classifier, 
### which obtained a good precision score, but still has a relatively low 
### recall score. Previous model performance run by tester.py:
### Accuracy: 0.85527  Precision: 0.41028  Recall: 0.19550

### Tune the RandomForest Algorithm

rdf = RandomForestClassifier(random_state = 42)
pipeline = Pipeline(steps=[("SKB", skb),("PCA",pca),("Random", rdf)])
param_grid = {"SKB__k":range(5,7),"PCA__n_components":range(2,4),
"Random__n_estimators":[20,30],"Random__bootstrap":[True,False]}
gs = GridSearchCV(pipeline,param_grid,scoring="f1_weighted",cv=sss)
gs.fit(features,labels)
print gs.best_params_

### The best set of parameters is: 
### {'PCA__n_components': 2, 'SKB__k': 5, 'Random__bootstrap': False, 
### 'Random__n_estimators': 20}

# Create a classifier with the optimized parameters 
clf = gs.best_estimator_
# Print feature scores 
feature_score = clf.named_steps['SKB'].scores_
score_summary = {}
for i in range(len(feature_score)):
    k = features_list[1:][i]
    v = feature_score[i]
    score_summary[k]=v   
print OrderedDict(sorted(score_summary.items(), key=itemgetter(1)))

# Use the test_classifier to derive classification report
print " "
print "Tester Classification report:" 
print " "
test_classifier(clf, my_dataset, features_list)

# After tuning, the model performance has improved significantly, 
# and achieved Precision:0.45253 & Recall:0.367.

### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

# The RandomForest classifier I created has the best performance in terms of
# precison & recall scores which will be used as the final classifier.

skb = SelectKBest(k=5)
pca = PCA(n_components=2)
rdf = RandomForestClassifier(random_state = 42,bootstrap=False,n_estimators=30)
clf = Pipeline(steps=[("SKB", skb),("PCA",pca),("Random", rdf)])


dump_classifier_and_data(clf, my_dataset, features_list)


