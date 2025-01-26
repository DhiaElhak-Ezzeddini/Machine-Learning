import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree 
from sklearn.tree import plot_tree
from sklearn import preprocessing
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics          
import graphviz 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


df = pd.read_excel("car_excel.xlsx" , engine='openpyxl')
print(df.shape)
print(df.info())
print(df.columns)
print(df.describe(include='all'))

#-----------------------
#--- Labels encoding ---
#-----------------------

print(df.BUYING.value_counts(normalize=True))
le=preprocessing.LabelEncoder()
le.fit(['med','high','low','vhigh' , 'small' ,'big' , 'unacc' , 'acc' , 'good' , 'vgood'])
df_bis = df

df_bis['BUYING'] = le.fit_transform(df['BUYING'])
df_bis['MAINT'] = le.fit_transform(df['MAINT'])
df_bis['SAFETY'] = le.fit_transform(df['SAFETY'])
df_bis['LUG_BOOT'] = le.fit_transform(df['LUG_BOOT'])

df_bis['DOORS'] = pd.to_numeric(df['DOORS'] , errors='coerce')
df_bis['DOORS']=df_bis['DOORS'].fillna(5)
df_bis['PERSONS'] = pd.to_numeric(df['PERSONS'] , errors='coerce')
df_bis['PERSONS']=df_bis['PERSONS'].fillna(5)

print(df_bis.info)
print(df_bis.describe(include='all'))
print(df_bis.columns)

dfTrain , dfTest = train_test_split(df_bis , train_size=0.7 , test_size=0.3 , random_state=1 , stratify=df_bis.DECISION)

#-------------------------------------------------
#------- saving of df_bis to a xlsx file ---------
#-------------------------------------------------

file_name = 'car_excel_bis.xlsx'
df_bis.to_excel(file_name,index=False)

#### training Data ####

Xtrain=dfTrain.iloc[:,0:6]
ytrain=dfTrain.iloc[:,6]
Xtrain_name = dfTrain.columns[0:6]
ytrain_name = dfTrain.iloc[6]

#### Test Data ####

Xtest=dfTest.iloc[:,0:6]
ytest=dfTest.iloc[:,6]
Xtest_name = dfTest.columns[0:6]
ytest_name = dfTest.iloc[6]

#--------------------------------
#------ Tree Instantiation ------
#--------------------------------

tree_entropy = DecisionTreeClassifier(min_samples_split=5,min_samples_leaf=1,criterion="entropy")

#-------------------------------
#------- Model Creation --------
#-------------------------------

tree_entropy.fit(Xtrain,ytrain)

#--------------------------------------------------
#---- first solution to visualize the tree --------
#--------------------------------------------------

plt.figure(figsize=(10,10))
plot_tree(tree_entropy,feature_names=Xtrain_name,filled=True)
#plt.show()

#---------------------------------------------------
#---- second solution to visualize the tree --------
#---------------------------------------------------

dot_data = export_graphviz(tree_entropy , out_file=None , 
                           feature_names=Xtrain_name , 
                           class_names=dfTrain['DECISION'].unique().astype(str),
                           filled=True , 
                           rounded=True, 
                           special_characters=True)

graph = graphviz.Source(dot_data)
graph.render('decision_tree_graph' , format="png",cleanup=True )
graph.view()

#-------------------------------------------------
#--- run the classification on the test Data -----
#-------------------------------------------------

y_pred = tree_entropy.predict(Xtest)
###### Confusion Matrix ######
conf_matrix = metrics.confusion_matrix(ytest , y_pred)
print("******* Confusion Matrix for Decisionn Tree : \n" , conf_matrix)
###### Classification Report  ######
class_report = metrics.classification_report(ytest , y_pred)
print("******* Classification Report for Decision Tree : \n" , class_report)

##################################################
########### Binary Classification ################
##################################################


df_bin = df_bis
df_bin['DECISION'] = df_bin['DECISION'].apply(lambda x: 1 if x in ['good', 'vgood'] else 0)
print(df_bin.tail())
#---------------------------------
#------ classification -----------
#---------------------------------

bin_train , bin_test = train_test_split(df_bin , train_size=0.7 , test_size=0.3 , random_state=1 , stratify=df_bis.DECISION)

X_bintrain=bin_train.iloc[:,0:6]
y_bintrain=bin_train.iloc[:,6]
X_bintrain_name = bin_train.columns[0:6]
y_bintrain_name = bin_train.iloc[6]

X_bintest=bin_test.iloc[:,0:6]
y_bintest=bin_test.iloc[:,6]
X_bintest_name = bin_test.columns[0:6]
y_bintest_name = bin_test.iloc[6]

tree_entropy_bin = DecisionTreeClassifier(min_samples_split=5,min_samples_leaf=1,criterion="entropy")
tree_entropy_bin.fit(X_bintrain, y_bintrain)

#---------------------------------------------------
#------------- visualizing the tree ----------------
#---------------------------------------------------

dot_data = export_graphviz(tree_entropy_bin , out_file=None , 
                           feature_names=X_bintrain_name , 
                           class_names=bin_train['DECISION'].unique().astype(str),
                           filled=True , 
                           rounded=True, 
                           special_characters=True)

graph = graphviz.Source(dot_data)
graph.render('decision_tree_bin_graph' , format="png",cleanup=True )
graph.view()

#-----------------------------------------------------------
#----- run the Binary Classification on the test Data ------
#-----------------------------------------------------------

y_bin_pred = tree_entropy_bin.predict(X_bintest)

###### Confusion Matrix ######
conf_matrix = metrics.confusion_matrix(y_bintest , y_bin_pred)
print("******* Confusion Matrix for Binary Classification  : \n" , conf_matrix)
###### Classification Report  ######
class_report = metrics.classification_report(y_bintest , y_bin_pred)
print("******* Classification Report for Binary Classification : \n" , class_report)

##################################################
################ Random Forest ###################
##################################################

rf_model = RandomForestClassifier(n_estimators=100 , max_depth=None , criterion='gini' , random_state=42)

## df_bis is the Dataset we will be working on 

rf_model.fit(Xtrain , ytrain)

dot_data = export_graphviz(rf_model.estimators_[0] , out_file=None , 
                           feature_names=Xtrain_name , 
                           class_names=dfTrain['DECISION'].unique().astype(str),
                           filled=True , 
                           rounded=True, 
                           special_characters=True)

graph = graphviz.Source(dot_data)
graph.render('Random_Forest_graph' , format="png",cleanup=True )
graph.view()

#-----------------------------------------------------------
#----- run the Binary Classification on the test Data ------
#-----------------------------------------------------------

y_rf_pred = rf_model.predict(Xtest)

###### Confusion Matrix ######
conf_matrix = metrics.confusion_matrix(ytest , y_rf_pred)
print("******* Confusion Matrix for Random Forest  : \n" , conf_matrix)
###### Classification Report  ######
class_report = metrics.classification_report(ytest , y_rf_pred)
print("******* Classification Report for Random Forest : \n" , class_report)

#######################################################
################# Gradient Boosting ###################
#######################################################


gb_model = GradientBoostingClassifier(learning_rate=0.1 , n_estimators=100 , max_depth=3 , random_state=42)

gb_model.fit(Xtrain , ytrain)

dot_data = export_graphviz(gb_model.estimators_[0 , 0] , out_file=None , 
                           feature_names=Xtrain_name , 
                           class_names=dfTrain['DECISION'].unique().astype(str),
                           filled=True , 
                           rounded=True, 
                           special_characters=True)

graph = graphviz.Source(dot_data)
graph.render('Gradient_Boosting_graph' , format="png",cleanup=True )
graph.view()

y_gb_pred = gb_model.predict(Xtest)

###### Confusion Matrix ######
conf_matrix = metrics.confusion_matrix(ytest , y_gb_pred)
print("******* Confusion Matrix for Gradient Boosting  : \n" , conf_matrix)
###### Classification Report  ######
class_report = metrics.classification_report(ytest , y_gb_pred)
print("******* Classification Report for Gradient Boosting : \n" , class_report)
















