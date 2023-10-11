# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 12:38:52 2020

@author: Ankita
"""


from imblearn.over_sampling import SMOTE 
from imblearn.over_sampling import KMeansSMOTE
import random
from sklearn.feature_selection import RFE
from sklearn.linear_model import  LogisticRegression
import matplotlib.pyplot as plt
import h5py
from sklearn.metrics import roc_auc_score, roc_curve, auc
from matplotlib.pyplot import yticks, xticks, subplots, set_cmap
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pickle
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib  import pyplot
import seaborn as sns
import scipy.stats as ss
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import os
import img2pdf
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import calendar
import util as utils
import visualisation as viz
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 10000)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import ensemble
import gc
import numpy as np
h = .02  # step size in the mesh
import logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logger = logging.getLogger('CHURN_PRED')
logging.basicConfig(filename='log_churn_1.log',level=logging.DEBUG)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.info('logger established')

names = ["XGB","GBM","Neural Net","Naive Bayes","Decision Tree", "Random Forest","Nearest Neighbors","LogisticRegression"]

classifiers = [XGBClassifier(max_delta_step=2,scale_pos_weight=100),
               GradientBoostingClassifier(n_estimators=100,random_state=0,verbose=1),
    MLPClassifier(alpha=0.00001,max_iter=27577,activation='logistic',batch_size=100,verbose=True,random_state=0),
    GaussianNB(),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=7, n_estimators=100,min_samples_split=2, max_features=3, class_weight='balanced'),
    KNeighborsClassifier(2),
    LogisticRegression(penalty='l2',class_weight='balanced', random_state=0, 
                                        solver='lbfgs', max_iter=100, multi_class='auto', verbose=0,
                                        warm_start=False, n_jobs=None, l1_ratio=None)]
def active_subs(data):
    '''Remove disconnected subs'''
    data=data[data['CURR_STATUS_DESC']==data['CURR_STATUS_DESC'].value_counts(dropna=False).index[0]]
    data.drop(columns=['CURR_STATUS_DESC'],inplace=True)
    return data
gc.collect()
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
######read_data and remove G1/G2 by applying filter of G1 Start date
#-----------------------------------------------------------------
Model_BuildDate='2020-05-16'
#BaseScored_Date='2020-04-03'
CIRCLE='MH'
RES_MMFA=pd.read_csv('../Model_base_16/2_RES_MMFA_'+CIRCLE+'_'+Model_BuildDate+'.csv')
RES_MAKO=pd.read_csv('../Model_base_16/3_RES_MAKO_'+CIRCLE+'_'+Model_BuildDate+'.csv')
RES_base_RFM=RES_MAKO.merge(RES_MMFA,how='inner',on=['SUBS_ID'])
Churn_base=pd.read_csv('../Model_base_16/4_Churn_'+CIRCLE+'_'+Model_BuildDate+'.csv')
Churn_base['classVar']=1
Churn_base=Churn_base[['SUBS_ID','classVar']]

RES_base=RES_base_RFM.merge(Churn_base,how='left',on='SUBS_ID')
RES_base['classVar'].fillna(0,inplace=True)
RES_base['G1_START_DATE'] = pd.to_datetime(RES_base['G1_START_DATE'])
#-------------removing G1--------------------------------------------------
RES_base=RES_base[RES_base['G1_START_DATE']>Model_BuildDate]
RES_target=RES_base.copy()

#--------------------------convert date to weekday name
RES_target['LST_RCHG_DT'] = pd.to_datetime(RES_target['LST_RCHG_DT'])
RES_target['LST_RCHG_DT_WEEKDAY'] = RES_target['LST_RCHG_DT'].dt.weekday_name

#------------------------- Drop unnecessary columns
RES_target.drop(columns=['CURR_TARIFF_PLAN_SER_CLASS','BASE_CIRCLE','G1_START_DATE','G2_START_DATE','SEGMENT_GROUP_DESC','LST_RCHG_DT','ACTIVATION_DATE','HANDSET_MANUFACTURER','PRE_POST_IND'],inplace=True)

#---------------------------Convert port out request to binary variable
RES_target['PORT_REQUEST_FLAG']=np.where(RES_target['PORT_OUTS']>0,'1','0')
RES_target.drop(columns=['PORT_OUTS'],inplace=True)

#-----------------------------keep active users only(removing discoonected ones)
print(RES_target['CURR_STATUS_DESC'].value_counts(dropna=False))
RES_target=active_subs(RES_target)
#-----------------------------APRU_1_mnth>=100 only
RES_target=RES_target[RES_target['ARPU_1_MNTH']>=100]

#--- Get product categories from two columns 'PRODUCT_TYPE_SRH and UL_CATEGORY_SRH
#----and combine to create 'product_cat' variable
RES_target_UL=RES_target[RES_target['PRODUCT_TYPE_SRH'].isin(['UNLIMITED'])]
RES_target_UL['product_cat']='UL'+RES_target_UL['UL_CATEGORY_SRH']

RES_target_NUL=RES_target[~RES_target['PRODUCT_TYPE_SRH'].isin(['UNLIMITED'])]
RES_target_NUL['product_cat']=np.where(RES_target_NUL['LAST_RECHARGE'].isin([49,79]),'COMBO',RES_target_NUL['PRODUCT_TYPE_SRH'])
RES_target=pd.concat([RES_target_UL,RES_target_NUL])

RES_target.drop(columns=['PRODUCT_TYPE_SRH','UL_CATEGORY_SRH'],inplace=True)

gc.collect(0)
#--------------------------- Get missing value counts----------------------------
RES_target_null_counts=RES_target.apply(lambda x: x.isnull().sum(axis=0))
print(RES_target_null_counts) 


#------------------- convert some character type variables to numeric------------
RES_target['DAYS_LEFT_TO_EXPIRY']=RES_target['DAYS_LEFT_TO_EXPIRY'].astype(int)

#----------------------- get list of categorical variables and numeric ones------
categorical_vars= RES_target.select_dtypes(exclude=['float64', 'int64','int32']).columns
numerical_vars = RES_target.select_dtypes(include=['float64', 'int64','int32']).columns
numerical_vars=numerical_vars.drop(['SUBS_ID','MSISDN_x','MSISDN_y','classVar'])
#--------------------------- Get classwise stats------------------------------------
df_stats_categorical_vars=utils.getClasswiseStats(RES_target,'classVar',categorical_vars)
print(df_stats_categorical_vars)
df_stats_numerical_vars=utils.getClasswiseStats(RES_target,'classVar',numerical_vars)
print(df_stats_numerical_vars)

#----------------------- Check Cramer's V for the categorical vars--------------------
for i in categorical_vars:
    # Strength of association between the categorical column & classVar
    val = utils.cramers_v(RES_target[i],RES_target['classVar'])
    print('Coeff for {} = {}'.format(i,val))
    
#----- Rename less popular categories to others-----------------------------------
print(RES_target['TWOG_3G_FLAG'].value_counts())
RES_target=utils.reduceNumOfCategoriesByMaxCategorySize(RES_target,'TWOG_3G_FLAG')
print(RES_target['TWOG_3G_FLAG'].value_counts())

#------ Rename less popular categories to others-----------------------------------
print(RES_target['HSET_2G_3G_INDICATOR'].value_counts(dropna=False))
RES_target=utils.reduceNumOfCategoriesByMaxCategorySize(RES_target,'HSET_2G_3G_INDICATOR')
print(RES_target['HSET_2G_3G_INDICATOR'].value_counts())

#------- Rename less popular categories to others----------------------------------
print(RES_target['SEGMENT_DESC'].value_counts(dropna=False))
RES_target=utils.reduceNumOfCategoriesByMaxCategorySize(RES_target,'SEGMENT_DESC')
print(RES_target['SEGMENT_DESC'].value_counts())

#------- Rename less popular categories to others------------------------------------
print(RES_target['product_cat'].value_counts(dropna=False))
RES_target=utils.reduceNumOfCategoriesByMaxCategorySize(RES_target,'product_cat')
print(RES_target['product_cat'].value_counts(dropna=False))



#------------------------ Get missing value counts--------------------------------
RES_target_null_counts=RES_target.apply(lambda x: x.isnull().sum(axis=0))
print(RES_target_null_counts)


#--------------------------------------------------------------------------------
# -----------------------Missing value treatment-------------------------------
#------------------------------------------------------------------------------
print(RES_target['GPRS_IND'].value_counts(dropna=False))
RES_target['GPRS_IND'].fillna(RES_target['GPRS_IND'].value_counts(dropna=False).index[0], inplace=True)
RES_target['LST_RCHG_DT_WEEKDAY'].fillna(RES_target['LST_RCHG_DT_WEEKDAY'].value_counts(dropna=False).index[0], inplace=True)

#Gender has missing values and unknowns
RES_target['GENDER'].value_counts(dropna=False)
RES_target['GENDER']=RES_target['GENDER'].apply(lambda x: str(x))
# Get the Unknown gender resolved
# Get mean of AON for those with Gender = U (unknown)
# Get most common gender for subs with AON >  mean AON of Unknowns
# Replace unknown gender to the gender deduced in previous step
try:
    AON_to_deduce_U_Gender=RES_target[RES_target['GENDER'].astype(str).str[0]=='U'].AON.mean() #+RES_target['AON'].describe().loc['std']
    unknownToThisGender=RES_target.GENDER[RES_target['AON'] > AON_to_deduce_U_Gender].value_counts().index[0]
    RES_target['GENDER']=RES_target['GENDER'].apply(lambda x: unknownToThisGender if str(x[0])=='U' else x)
except:
    None
# Get the missing Gender Resolved
# get mean of AON for those with Gender =nan
# Get most common gender for subs with AON >  mean AON of missing
# Replace missing gender to the gender deduced in previous step.
try:
    AON_to_deduce_Gender=RES_target[RES_target['GENDER']=='nan'].AON.mean()
    missingToThisGender=RES_target.GENDER[RES_target['AON'] > 
                                          AON_to_deduce_Gender].value_counts().index[0]
    RES_target['GENDER']=RES_target['GENDER'].apply(lambda x: missingToThisGender if x=='nan' else x)
except:
    None

RES_target['GENDER'].value_counts(dropna=False)
                                  


# Replace missing AON values by the gender-wise mean
RES_target['AON'].value_counts(dropna=False)
RES_target['AON'].isnull().sum()

RES_target_genderwise = RES_target.groupby('GENDER')['AON'].mean()
RES_target[RES_target['AON'].isnull()]['AON']=RES_target_genderwise[RES_target['GENDER']]

#------------Kramer's V---------------------------------------------------------
for i in categorical_vars:
    val = utils.cramers_v(RES_target[i],RES_target['classVar'])
    print('Coeff for {} = {}'.format(i,val))
    

#----------------------------------------------------------
#----------------------------------------------------------
# Handle NULL values and unrealistic Values
#----------------------------------------------------------
#----------------------------------------------------------

# replace negative values in any column with  zero 
#for col in numerical_vars:
#    RES_target[col][RES_target[col] < 0 ] = 0

gc.collect(0)
df_stats_numerical_vars = RES_target.groupby('classVar')[numerical_vars].describe().T

# Get classwise percentage missing values. Features with non-zero missing 
# values will be returned with respective values for the classes
nullPercentage=utils.getClasswiseNullValuePercent(RES_target,'classVar')
RES_target_na_replaced=RES_target.copy()

#---------------
for i in numerical_vars:
    if i in ('DAYS_SINCE_LST_RCHG','DAYS_SINCE_LST_USG'):
        RES_target_na_replaced[i].fillna(RES_target_na_replaced[i].max(),inplace=True)
    if i in ('PORT_OUTS','OG_DAYS','IC_DAYS','DATA_DAYS','IC_MOU_30_TO_45','IC_MOU_15_TO_30','OG_MOU_15_TO_30','OG_MOU_30_TO_45','DATA_USAGE_15_TO_30','DATA_USAGE_30_TO_45','DIFF_DATA_USAGE','PORT_FLAG','TOTAL_CRM','CRM_FLAG','MRP_6MNTH', 'FREQ','NUM_UL_RECH','DATA_USAGE','OG_MOU','IC_MOU'):
        RES_target_na_replaced[i].fillna(0,inplace=True)
    if i in ('OG_CALL_COUNT'):
        RES_target_na_replaced[i].fillna(RES_target_na_replaced['OG_MOU']/10,inplace=True)
    if i in ('IC_CALL_COUNT'):
        RES_target_na_replaced[i].fillna(RES_target_na_replaced['IC_MOU']/10,inplace=True)
        
RES_target_na_replaced['HSET_2G_3G_INDICATOR'].fillna(RES_target_na_replaced.HSET_2G_3G_INDICATOR.value_counts().index[0],inplace=True)
RES_target_na_replaced['TWOG_3G_FLAG'].fillna(RES_target_na_replaced.TWOG_3G_FLAG.value_counts().index[0],inplace=True)
RES_target_na_replaced['AGE'].fillna(RES_target_na_replaced.AGE.mean(),inplace=True)
RES_target_na_replaced['product_cat'].fillna(RES_target_na_replaced.product_cat.value_counts().index[0],inplace=True)
RES_target_na_replaced['LAST_RECHARGE'].fillna(RES_target_na_replaced.MRP_6MNTH/RES_target_na_replaced.FREQ,inplace=True)
RES_target_na_replaced['LAST_RECHARGE'].fillna(0,inplace=True)

#-------------------creating difference variables for Data and Outgoing calls
RES_target_na_replaced['Diff_Data']=RES_target_na_replaced['DATA_USAGE_15_TO_30']-RES_target_na_replaced['DATA_USAGE_0_TO_15']
RES_target_na_replaced['Diff_OG']=RES_target_na_replaced['OG_MOU_15_TO_30']-RES_target_na_replaced['OG_MOU_0_TO_15']

num_vars=pd.Index(['Diff_OG','Diff_Data'])
numerical_vars=numerical_vars.append(num_vars)


# ------------------------------Get missing value counts------------------------------
RES_target_null_counts=RES_target_na_replaced.apply(lambda x: x.isnull().sum(axis=0))
print(RES_target_null_counts)
#-------------------------------Get classwise stats----------------------------------
df_stats_categorical_vars=utils.getClasswiseStats(RES_target_na_replaced,'classVar',categorical_vars)
df_stats_numerical_vars=utils.getClasswiseStats(RES_target_na_replaced,'classVar',numerical_vars)
#------------------------count of classwise uptliers---------------------------------
lower_out,upper_out=utils.count_classwise_outliers(RES_target_na_replaced,0.05,0.95)
print(lower_out)
print(upper_out)

#----------------------------- Change result path------------------------------------
result_path = "../out/"
#-------------------------- get variable stats in a file before outlier correction
with pd.ExcelWriter(result_path+'Variables stats.xlsx') as writer:
    df_stats_categorical_vars.to_excel(writer, sheet_name='categorical_vars')
    df_stats_numerical_vars.to_excel(writer, sheet_name='numerical_vars')
    lower_out.to_excel(writer, sheet_name='lower_range_outliers_.05')
    upper_out.to_excel(writer, sheet_name='upper_range_outliers.95')
print(utils.getClasswiseNullValuePercent(RES_target_na_replaced,'classVar'))


#----------------------------------------------------------------------------------
#------------------------------drop outliers
#RES_target_na_replaced=RES_target_na_replaced[RES_target_na_replaced['SUBS_BALANCE_AMT']<500]
RES_target_na_replaced=RES_target_na_replaced[RES_target_na_replaced['DAYS_LEFT_TO_EXPIRY']<180]

#RES_target_na_replaced=RES_target_na_replaced[RES_target_na_replaced['Diff_Data']>RES_target_na_replaced['Diff_Data'].quantile(0.05)]
#RES_target_na_replaced2=RES_target_na_replaced[RES_target_na_replaced['DATA_USAGE_0_TO_15']<RES_target_na_replaced['DATA_USAGE_0_TO_15'].quantile(0.95)]
#RES_target_na_replaced=RES_target_na_replaced2[RES_target_na_replaced2['AON']<RES_target_na_replaced2['AON'].quantile(0.95)]


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#------------------------    DATA VISUALIZATION---------------------------------

#---------------------    1. Categorical Variables-------------------------------

viz.getCategoricalVariableWithTargetViz(RES_target_na_replaced,categorical_vars)
viz.getCategoricalVariablePropWithTargetViz(RES_target_na_replaced,categorical_vars)


#---------------------     2.Numerical Variables (box plot and distribution)-----
viz.getNumericalVariableWithTargetViz(RES_target_na_replaced,numerical_vars) 
viz.plotboxplotsfornumerical(RES_target_na_replaced,numerical_vars,0) 

# FOR AON fix high values outliers to mean+-\std. 

# For all numerical variables, fix lower and higher outliers to mean +-std

for i in numerical_vars:
    if i in ('ARPU_6_MNTH','MRP_6MNTH','FREQ','DAYS_SINCE_LST_RCHG','OG_CALL_COUNT',
             'IC_CALL_COUNT','SUBS_BALANCE_AMT','OG_MOU','IC_MOU','DATA_USAGE',
             'OG_REV'):
        RES_target_na_replaced[i]=utils.fix_outliers(RES_target_na_replaced[i])
    if i in ('AON','NUM_UL_RECH','DAYS_SINCE_LST_RCHG'):
        RES_target_na_replaced[i]=utils.fix_upper(RES_target_na_replaced[i])

#---------------------Get variables stats post outlier correction
df_stats_categorical_vars=utils.getClasswiseStats(RES_target_na_replaced,'classVar',categorical_vars)
df_stats_numerical_vars=utils.getClasswiseStats(RES_target_na_replaced,'classVar',numerical_vars)
lower_out,upper_out=utils.count_classwise_outliers(RES_target_na_replaced,0.05,0.95)
print(lower_out)
print(upper_out)

# Change result path
result_path = "../out/"
#-------------------------- get variable stats in a file before outlier correction
with pd.ExcelWriter(result_path+'Variables stats_post_outlier_correction.xlsx') as writer:
    df_stats_categorical_vars.to_excel(writer, sheet_name='categorical_vars')
    df_stats_numerical_vars.to_excel(writer, sheet_name='numerical_vars')
    lower_out.to_excel(writer, sheet_name='lower_range_outliers_.05')
    upper_out.to_excel(writer, sheet_name='upper_range_outliers.95')

#-------------------------------------------------------------------------------
#-----------------------Image to PDF
#--------------------------------------------------------------------------------
current_path = '../../src/'
with open("../out/Churn_EDA_categorical_variables.pdf", "wb") as f:
    os.chdir(r"../out/cat")
    f.write(img2pdf.convert([i for i in os.listdir(os.getcwd()) if i.endswith(".jpeg")]))
with open("../Churn_EDA_numerical_var_distribution.pdf", "wb") as f:
    os.chdir(r"../num_dist")
    f.write(img2pdf.convert([i for i in os.listdir(os.getcwd()) if i.endswith(".jpeg")]))
with open("../Churn_EDA_box_plots.pdf", "wb") as f:
    os.chdir(r"../box_plot_before")
    f.write(img2pdf.convert([i for i in os.listdir(os.getcwd()) if i.endswith(".jpeg")]))
    # change the directory back to current directory
    os.chdir(current_path)
# box plt after fixing outliers    
viz.plotboxplotsfornumerical(RES_target_na_replaced,numerical_vars,1) 
with open("../out/Churn_EDA_box_plots2.pdf", "wb") as f:
    os.chdir(r"../out/box_plot_after")
    f.write(img2pdf.convert([i for i in os.listdir(os.getcwd()) if i.endswith(".jpeg")]))
    # change the directory back to current directory
    os.chdir(current_path)
    
#--------------------Correlation plot for churners  
churner=RES_target[RES_target['classVar']==1]
viz.plotCorrelationMatrix(churner)
plt.savefig('correlation_churner.jpeg')
plt.close()

#--------------------Correlation plot for non-churners
nonchurner=RES_target[RES_target['classVar']==0]
viz.plotCorrelationMatrix(nonchurner)
plt.savefig('correlation_non_churner.jpeg')
plt.close()

#-----------------difference in correlation
viz.CorrelationMatrixDiff(RES_target)
plt.savefig('correlation_diff.jpeg')
plt.close()
#---------------------------------------------
#---------------------------variable addition
#-----------------------------------------------
RES_target_na_replaced['rat_usage_expiry']=np.exp(RES_target_na_replaced['DAYS_SINCE_LST_USG'])/RES_target_na_replaced['DAYS_LEFT_TO_EXPIRY']
RES_target_na_replaced['ratio_MRP_to_recharge']=RES_target_na_replaced['MRP_6MNTH']/(RES_target_na_replaced['NUM_UL_RECH']+0.1)
RES_target_na_replaced['ratio_ARP6month_1mnth']=RES_target_na_replaced['ARPU_6_MNTH']/RES_target_na_replaced['ARPU_1_MNTH']


RES_target_na_replaced['ratio_data_drop']=RES_target_na_replaced['DATA_USAGE_30_TO_45']/(RES_target_na_replaced['DATA_USAGE_0_TO_15']+0.1)
RES_target_na_replaced['ratio_OG_drop']=RES_target_na_replaced['OG_MOU_30_TO_45']/(RES_target_na_replaced['OG_MOU_0_TO_15']+0.1)
RES_target_na_replaced['ratio_usage_drop']=(RES_target_na_replaced['OG_MOU_30_TO_45']+RES_target_na_replaced['DATA_USAGE_30_TO_45'])/(RES_target_na_replaced['OG_MOU_0_TO_15']+RES_target_na_replaced['DATA_USAGE_0_TO_15']+0.1)
RES_target_na_replaced['ratio_OG_IC']=RES_target_na_replaced['OG_CALL_COUNT']/(RES_target_na_replaced['IC_CALL_COUNT']+0.1)
RES_target_na_replaced['prod_AON_OG']=RES_target_na_replaced['AON']*RES_target_na_replaced['OG_CALL_COUNT']

#-----------------------discretizing numerical variables to categorical variables
#RES_target_na_replaced['expiry_slab']=RES_target_na_replaced.apply (lambda row: utils.label_Days_to_expiry(row,'DAYS_LEFT_TO_EXPIRY'), axis=1)   
#RES_target_na_replaced['balance_slab']=RES_target_na_replaced.apply (lambda row: utils.label_Balance_bucket(row,'SUBS_BALANCE_AMT'), axis=1)   
#RES_target_na_replaced['DATA_slab']=RES_target_na_replaced.apply (lambda row: utils.label_data(row,'DATA_USAGE_0_TO_15'), axis=1)   

num_vars=pd.Index(['ratio_ARP6month_1mnth','ratio_MRP_to_recharge',
                   'prod_AON_OG','ratio_OG_IC','rat_usage_expiry',
                   'ratio_data_drop','ratio_OG_drop','ratio_usage_drop'])
numerical_vars=numerical_vars.append(num_vars)
#cat_vars=pd.Index(['expiry_slab','balance_slab','DATA_slab'])
#categorical_vars=categorical_vars.append(cat_vars)

RES_AllVariables=RES_target_na_replaced.copy()
#----------------------------------------------------------------------------------
#-------------------Variable Selection
#---------------------------------------------------------------------------------

random.seed(0)
numerical_vars_k=utils.getTopFeaturesByTreeClassifier(RES_target_na_replaced, numerical_vars,'classVar',10)
l=[]
for i in categorical_vars:
    val = utils.cramers_v(RES_target_na_replaced[i],RES_target_na_replaced['classVar'])
    print('Coeff for {} = {}'.format(i,val))
    l.append((i,val))
l_sorted=sorted(l, key = lambda x: x[1],reverse=True)[0:6]
categorical_vars_k=[i[0] for i in l_sorted][0:3]
numerical_vars_k=numerical_vars_k['TreeClassifier_Variable'].tolist()

'''for using all the variables'''
numerical_vars_k=[i for i in numerical_vars]
categorical_vars_k=[i for i in categorical_vars]

RES_target_na_replaced=RES_target_na_replaced[numerical_vars_k+categorical_vars_k+['classVar','SUBS_ID']]

#------------------CORRELATION
viz.plotCorrelationMatrix(RES_target_na_replaced)
plt.savefig('../out/corr_plot.jpeg')
plt.close()

#-----------add correlated variables to this list---------------------------
correlated_vars=[]
RES_target_na_replaced=RES_target_na_replaced.drop(correlated_vars,axis=1)

#------split data-------------------------------------------------
#------------------------shuffle before splitting-------------------------
RES_target_na_replaced=RES_target_na_replaced.sample(frac=1,random_state=0).reset_index(drop=True)
train_RES,test_RES= utils.splitData(RES_target_na_replaced,0.7)
#-----one hot encoding of categorical variables
train_RES,test_RES=utils.convert_cat2float(train_RES,test_RES)
y_train=train_RES[['classVar']]
y_test=test_RES[['classVar']]

X_train=train_RES.drop(['classVar','SUBS_ID'],axis=1)
X_test=test_RES.drop(['classVar','SUBS_ID'],axis=1)

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_t, y_t = sm.fit_resample(X_train, y_test)
#-----------------Scaling----------------------------------------------------------
scaler = StandardScaler()   
scaler=scaler.partial_fit(X_train)
pickle.dump(scaler,open('natural_dist/scaler11.pkl','wb'))

scale=pickle.load(open('natural_dist/scaler11.pkl','rb'))
X_train = scaler.transform(X_train)
X_test= scaler.transform(X_test)

gc.collect(0)




#-----------------------------------------------------------------------
#---------Model run
#-----------------------------------------------------------------------
#------------------iterate over classifiers
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
gc.collect()
logger.info('Running classifier {}'.format(name))
#-------------------------------Logitic regression-----------------------
#name='LogisticRegression'
#clf=LogisticRegression(penalty='l2',class_weight='balanced', random_state=0, 
#                                   solver='lbfgs', max_iter=100, multi_class='auto', verbose=0,
#                                    warm_start=False, n_jobs=None, l1_ratio=None)
#-------------------------------Random Forest-------------------------------
#name="Random Forest"
#clf=RandomForestClassifier(max_depth=7, n_estimators=100,min_samples_split=2,
#                          max_features=3, class_weight='balanced',random_state=0)

#--------------------------XGBoost-----------------------------------------
name="XGB"
clf=XGBClassifier(base_score=0.0135,max_delta_step=2,scale_pos_weight=73,max_depth=4)
clf.fit(np.array(X_train), y_train)
logger.info('Trained {}'.format(clf))
score = clf.score(np.array(X_test), y_test)

if name=="Random Forest":        
    # Extract single tree
    estimator =clf.estimators_[5]
    from sklearn.tree import export_graphviz
    # Export as dot file
    export_graphviz(estimator, out_file='tree1.dot', feature_names= RES_target_na_replaced.columns[:-2],
                    rounded = True, proportion = False, 
                    precision = 2, filled = True)
    
    # Convert to png using system command (requires Graphviz)
    from graphviz import render
    render('dot', 'png', 'tree1.dot')
    
probs=[prob[1] for prob in clf.predict_proba(X_test)]
probs_train=[prob[1] for prob in clf.predict_proba(X_train)]
y_pred=[]
for prob in probs:
    y_pred.append((prob>=0.70).astype(int))
logger.info('Scored classifier {} as {}'.format(name,score))
print()
logger.info(confusion_matrix(y_test, y_pred))
print()
logger.info(classification_report(y_test, y_pred))

    '''''''''threshold kept at 0.7'''
#---------------------training performance------------
y_pred_train=[]
for prob in probs_train:
    y_pred_train.append((prob>=0.70).astype(int))
logger.info(confusion_matrix(y_train, y_pred_train))
logger.info(classification_report(y_train, y_pred_train))   
#----------------------SAVING THE MODEL  -------------- 
file='natural_dist/model_rf_MH_HVC_12_using more_all_vars_xgboost.pkl'
pickle.dump(clf,open(file,'wb'))

#--------------------Deciling----------------------- 
agg2,test_X_df=utils.decile_report(X_test, probs, y_test, test_RES ,segments=20)
test_X_df1 = utils.roc_info(test_X_df)   
agg2['Cumulative_Precision_rate']= 100*agg2['Target'].cumsum()/(agg2.Total.cumsum())
agg2['ROC_AUC'] = list(test_X_df1.values()) 
base_rate=100*y_test.sum()/y_test.shape[0]
agg2['base_rate_in_%']=base_rate[0]
agg2['lift']=100*agg2['Target'].cumsum()/(agg2.Total.cumsum()*agg2['base_rate_in_%'])
#agg2.to_csv('../out/model_report_MH.csv',index=False)

agg3,train_X_df=utils.decile_report(X_train, probs_train, y_train,train_RES ,segments=20)
train_X_df1 = utils.roc_info(train_X_df) 
agg3['Cumulative_Precision_rate']= 100*agg3['Target'].cumsum()/(agg3.Total.cumsum())
agg3['ROC_AUC'] = list(train_X_df1.values()) 
base_rate=100*y_train.sum()/y_train.shape[0]
agg3['base_rate_in_%']=base_rate[0]
agg3['lift']=100*agg3['Target'].cumsum()/(agg3.Total.cumsum()*agg3['base_rate_in_%'])

#------------------Classification report
cr_trn = utils.getClassificationReport(y_train,y_pred_train)
cr_trn['True_Negative'],cr_trn['False_Positive'],cr_trn['False_Negative'],cr_trn['True_Positive'],cr_trn['Target_Rate'],cr_trn['Capture_Rate']=utils.getConfusionMatrixValues(y_train,y_pred_train)
cr_trn['AUC_ROC'] = roc_auc_score(y_train, y_pred_train)
cr_trn['Dataset'],cr_trn['Prob_Th'] = 'Train',0.7
base_rate=(y_train.sum()/y_train.shape[0])*100
lift=(cr_trn['Target_Rate'][0]*y_train.shape[0]/(100*y_train.sum()))
cr_trn['base_rate']=base_rate[0]
cr_trn['lift']=lift[0]
cr_tst = utils.getClassificationReport(y_test,y_pred)
cr_tst['True_Negative'],cr_tst['False_Positive'],cr_tst['False_Negative'],cr_tst['True_Positive'],cr_tst['Target_Rate'],cr_tst['Capture_Rate']=utils.getConfusionMatrixValues(y_test,y_pred)
cr_tst['AUC_ROC'] = roc_auc_score(y_test, probs)
cr_tst['Dataset'],cr_tst['Prob_Th'] = 'Test',0.7
lift=(cr_trn['Target_Rate'][0]*y_test.shape[0]/(100*y_test.sum()))
base_rate=(y_test.sum()/y_test.shape[0])*100
cr_tst['base_rate']=base_rate[0]
cr_tst['lift']=lift[0]
cr = pd.concat([cr_trn,cr_tst])
modelInfo=pd.DataFrame(columns=['info'],index=['categorical_variables_used','numerical_variables_used','dropped_correlated_variables','model_name','model_parameter'])
modelInfo.loc['categorical_variables_used','info']=categorical_vars_k
modelInfo.loc['numerical_variables_used','info']=numerical_vars_k
modelInfo.loc['dropped_correlated_variables','info']=correlated_vars
modelInfo.loc['model_name','info']=name
modelInfo.loc['model_parameter','info']=clf

# Change result path
result_path = "../out/"
#--------------------final model training report------------------------------
with pd.ExcelWriter(result_path+'MH_voda_XGB12_using more_all_vars_xgboost.xlsx') as writer:
    cr.to_excel(writer, sheet_name='CLASSIFICATION_REPORT')
    pd.DataFrame(agg3).to_excel(writer, sheet_name='DECILE WISE ROC AUC - TRAIN',index=None)
    pd.DataFrame(agg2).to_excel(writer, sheet_name='DECILE WISE ROC AUC - TEST',index=None)
    modelInfo.to_excel(writer, sheet_name='Model_info',index=True)
#-------------Probability distribution-----------------------------------------
churn_yes = test_X_df[test_X_df['B'] == 1]
churn_no = test_X_df[test_X_df['B'] == 0]
plt.figure(figsize=(12,8))
    
sns.distplot(churn_no['P_1'],label='0')
sns.distplot(churn_yes['P_1'],label='1')
plt.legend(title='left',loc='best') 
plt.savefig('../out/prob_dist_xgboost_20.jpeg')
plt.close()

test=test_X_df[['SUBS_ID','B','P_1']]
test1=test.merge(RES_target_na_replaced,on=['SUBS_ID'],how='left')
test1=test1[test1['P_1']]
df_stats_numerical_vars=utils.getClasswiseStats(test1,'B',numerical_vars)

'''---------------------Grid search-------------------------------------
from sklearn.model_selection import GridSearchCV
parameters = {'max_depth':[7,9],'min_samples_split':[2,5,10], 'max_features':[3,4],'n_estimators':[100,150,200,250]}
clf = GridSearchCV(clf, param_grid=parameters,cv=5,verbose=10,scoring='f1')
print(clf)
clf.fit(X_train, y_train)

print(sorted(clf.cv_results_.keys()))
print(clf.best_estimator_.feature_importances_)
print(clf.best_estimator_)
'''

#--------------------------------------------------------------
#---------------------TP vs FP vs FN---------------------------
test_check_0=test_X_df[test_X_df['Decile']==0]
test_check_0_Nontarget=test_check_0[test_check_0['B']==0]
RES_check_0=test_check_0_Nontarget.merge(RES_AllVariables,on='SUBS_ID',how='left')
RES_check_0_Nontarget_numerical=utils.getClasswiseStats(RES_check_0,'classVar',numerical_vars)
RES_check_0_Nontarget_categorical=utils.getClasswiseStats(RES_check_0,'classVar',categorical_vars)

test_check_0_target=test_check_0[test_check_0['B']==1]
RES_check_0=test_check_0_target.merge(RES_AllVariables,on='SUBS_ID',how='left')
RES_check_0_target_numerical=utils.getClasswiseStats(RES_check_0,'classVar',numerical_vars)
RES_check_0_target_categorical=utils.getClasswiseStats(RES_check_0,'classVar',categorical_vars)

compare_numericals=RES_check_0_Nontarget_numerical.merge(RES_check_0_target_numerical,left_index=True,right_index=True)
compare_numericals.columns=['Decile_0,_nonTarget','Decile_0_target']

compare_categoricals=RES_check_0_Nontarget_categorical.merge(RES_check_0_target_categorical,left_index=True,right_index=True)
compare_categoricals.columns=['Decile_0,_nonTarget','Decile_0_target']


test_check_not_0=test_X_df[test_X_df['Decile']!=0]
test_check_not_0_target=test_check_not_0[test_check_not_0['B']==1]
RES_check_not_0=test_check_not_0_target.merge(RES_AllVariables,on='SUBS_ID',how='left')
RES_check_not_0_target_numerical=utils.getClasswiseStats(RES_check_not_0,'classVar',numerical_vars)
RES_check_not_0_target_categorical=utils.getClasswiseStats(RES_check_not_0,'classVar',categorical_vars)

compare_numericals1=compare_numericals.merge(RES_check_not_0_target_numerical,left_index=True,right_index=True)
compare_numericals1.columns=['Decile_0,_nonTarget','Decile_0_target','Decile_not_0_target']

compare_categoricals1=compare_categoricals.merge(RES_check_not_0_target_categorical,left_index=True,right_index=True)
compare_categoricals1.columns=['Decile_0,_nonTarget','Decile_0_target','Decile_not_0_target']

with pd.ExcelWriter(result_path+'TP_vs_FP_vs_TN.xlsx') as writer:
    compare_numericals1.to_excel(writer, sheet_name='compare_numericals')
    compare_categoricals1.to_excel(writer, sheet_name='compare_categoricals')



