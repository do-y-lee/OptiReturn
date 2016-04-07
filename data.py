import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import train_test_split

####################################################################
## 2007 - 2014 LC loan data
####################################################################

df3a = pd.read_csv('./LoanStats3a_securev1.csv', skiprows=1)
df3b = pd.read_csv('./LoanStats3b_securev1.csv', skiprows=1)
df3c = pd.read_csv('./LoanStats3c_securev1.csv', skiprows=1)
# df3d = pd.read_csv('./LoanStats3d_securev1.csv', skiprows=1)

## filter by loan status
df3aV1 = df3a[df3a['loan_status'].isin(['Charged Off','Default','Fully Paid'])]
df3bV1 = df3b[df3b['loan_status'].isin(['Charged Off','Default','Fully Paid'])]
df3cV1 = df3c[df3c['loan_status'].isin(['Charged Off','Default','Fully Paid'])]

# merge three dataframes
dfLC = pd.concat([df3aV1, df3bV1, df3cV1], ignore_index=True)


####################################################################
## Sampling Approach 1 - undersampling majority class
####################################################################

## get the total count of low-frequency group
sample_size = len(dfLC[dfLC['loan_status'].isin(['Charged Off','Default'])])

bad_indices = dfLC[dfLC['loan_status'].isin(['Charged Off','Default'])].index
bad_sample = dfLC.loc[bad_indices]

good_indices = dfLC[dfLC['loan_status']=='Fully Paid'].index
## use the low-frequency group count to randomly sample from high-frequency group
random_indices = np.random.choice(good_indices, sample_size, replace=False) 
good_sample = dfLC.loc[random_indices]

## merge bad and good data frames
dfLC_train_test = pd.concat([bad_sample, good_sample], ignore_index=True)

## create y column - 1 if charged off/default and 0 if fully paid
dfLC_train_test['y'] = dfLC_train_test['loan_status'].map(lambda x: 1 if (x=='Charged Off' or x=='Default') else 0)

## remove annual_inc greater than $250k
dfLC_train_test = dfLC_train_test[dfLC_train_test['annual_inc'] <= 250000]

print dfLC_train_test.shape
print dfLC_train_test.groupby('loan_status')['loan_status'].count()


####################################################################
## Selecting Raw Features
####################################################################

## df with only features for modeling
dfLC_train_test_v2 = dfLC_train_test[['y',
                                      'acc_now_delinq',
                                      'addr_state',
                                      'annual_inc',
                                      'pub_rec',
                                      'pub_rec_bankruptcies',
                                      'application_type',
                                      'installment',
                                      'int_rate',
                                      'total_acc',
                                      'collections_12_mths_ex_med',
                                      'delinq_2yrs',
                                      'dti',
                                      'emp_length',
                                      'home_ownership',
                                      'loan_amnt',
                                      'grade',
                                      'sub_grade',
                                      'tax_liens',
                                      'term',
                                      'total_rec_late_fee',
                                      'purpose',
                                      'revol_util',
                                      'revol_bal',
                                      'open_acc',
                                      'inq_last_6mths',
                                      'verification_status',
                                      'fico_range_high',
                                      'fico_range_low']]

## drop NaN rows
dfLC_train_test_v2 = dfLC_train_test_v2.dropna()

dfLC_train_test_v2.shape
## (94552, 29)


####################################################################
## Dummy variable features
####################################################################

dfLC_purpose = pd.get_dummies(dfLC_train_test_v2['purpose'], prefix='purpose')

dfLC_states = pd.get_dummies(dfLC_train_test_v2['addr_state'], prefix='state')

dfLC_apptype = pd.get_dummies(dfLC_train_test_v2['application_type'], prefix='apptype')

dfLC_emp_length = pd.get_dummies(dfLC_train_test_v2['emp_length'], prefix='emp_length')
dfLC_emp_length.drop('emp_length_n/a', axis=1, inplace=True) # drop column

dfLC_grade = pd.get_dummies(dfLC_train_test_v2['grade'], prefix='grade')

dfLC_subgrade = pd.get_dummies(dfLC_train_test_v2['sub_grade'], prefix='subgrade')

dfLC_home_ownership = pd.get_dummies(dfLC_train_test_v2['home_ownership'], prefix='home_ownership')

dfLC_term = pd.get_dummies(dfLC_train_test_v2['term'], prefix='term')


####################################################################
## Continuous variable features
####################################################################

## create new feature - avg fico scores - averaging between high and low
dfLC_train_test_v2['avg_fico_score'] = dfLC_train_test_v2.apply(lambda row: (row['fico_range_high']+row['fico_range_low'])/2, axis=1)

## create a new feature: loan_amnt / annual_inc = lta
dfLC_train_test_v2['lta'] = dfLC_train_test_v2.apply(lambda row: round(float(row['loan_amnt']) / row['annual_inc'], 4), axis=1)

## create a new feature: installment / loan_amnt = instl
dfLC_train_test_v2['instl'] = dfLC_train_test_v2.apply(lambda row: round(float(row['installment']) / row['loan_amnt'], 6), axis=1)

## change interest rate to decimal float
dfLC_train_test_v2['int_rate'] = dfLC_train_test_v2['int_rate'].apply(lambda x: round(float(x.replace('%',''))/100,6))

## change 'revol_util' into decimal float
dfLC_train_test_v2['revol_util'] = dfLC_train_test_v2['revol_util'].apply(lambda x: round(float(x.replace('%',''))/100),6)

## 'verification_status' = make it binary
dfLC_train_test_v2['verification_status'] = dfLC_train_test_v2['verification_status'].apply(lambda x: \
                                                                                          x.replace('Source Verified','1').\
                                                                                          replace('Verified','1').\
                                                                                          replace('Not 1','0'))

dfLC_continuous = dfLC_train_test_v2[['y',
                                      'avg_fico_score',
                                      'lta',
                                      'instl',
                                      'acc_now_delinq',
                                      'annual_inc',
                                      'pub_rec',
                                      'pub_rec_bankruptcies',
                                      'installment',
                                      'int_rate',
                                      'total_acc',
                                      'collections_12_mths_ex_med',
                                      'delinq_2yrs',
                                      'dti',
                                      'loan_amnt',
                                      'tax_liens',
                                      'total_rec_late_fee',
                                      'revol_util',
                                      'open_acc',
                                      'inq_last_6mths',
                                      'verification_status']]


####################################################################
## Normalizing Selected Continuous Features
####################################################################

## 'annual_inc', 'installment', 'total_acc', 'loan_amnt', 'total_rec_late_fee', 'avg_fico_score'
from sklearn import preprocessing

## separate the data from the target attributes
dfLC_cont_X = dfLC_continuous[['avg_fico_score', 'annual_inc', 'installment', 'total_acc', 'loan_amnt', 'total_rec_late_fee', 'dti']]

## normalize the data attributes
norm_dfLC_cont_X = preprocessing.normalize(dfLC_cont_X)

## convert norm_dfLC_cont_X into dataframe
norm_dfLC_cont_X = pd.DataFrame(norm_dfLC_cont_X)
norm_dfLC_cont_X.columns = ['avg_fico_score', 'annual_inc', 'installment', 'total_acc', 'loan_amnt', 'total_rec_late_fee', 'dti']

## replace old features with new normalized features in dfLC_continuous
dfLC_continuous['avg_fico_score'] = norm_dfLC_cont_X['avg_fico_score'].apply(lambda x: x)
dfLC_continuous['annual_inc'] = norm_dfLC_cont_X['annual_inc'].apply(lambda x: x)
dfLC_continuous['installment']= norm_dfLC_cont_X['installment'].apply(lambda x: x)
dfLC_continuous['total_acc'] = norm_dfLC_cont_X['total_acc'].apply(lambda x: x)
dfLC_continuous['loan_amnt'] = norm_dfLC_cont_X['loan_amnt'].apply(lambda x: x)
dfLC_continuous['total_rec_late_fee'] = norm_dfLC_cont_X['total_rec_late_fee'].apply(lambda x: x)
dfLC_continuous['dti'] = norm_dfLC_cont_X['dti'].apply(lambda x: x)


####################################################################
## Merge Dummy and Continuous Variabes into one DF
####################################################################

dfLC_train_test_v3 = pd.concat([dfLC_continuous, dfLC_purpose, dfLC_states, dfLC_apptype, dfLC_emp_length, dfLC_grade, \
                                dfLC_subgrade, dfLC_home_ownership, dfLC_term], axis=1)


####################################################################
## Split into Train and Test
####################################################################

## split train and test
X = dfLC_train_test_v3.dropna().iloc[:, 1:]
y = dfLC_train_test_v3.dropna().iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4444)


## VarianceThreshold - Removing features with low variance - DID NOT WORK WELL
# from sklearn.feature_selection import VarianceThreshold
# sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
# X_new = sel.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=4444)



