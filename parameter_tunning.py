# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 23:48:34 2020

@author: think
"""

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print('Load data...')
df=pd.read_csv(r'D:\prp\new_data备份\new_data_20.csv')

NUMERIC_COLS = [
        '1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'
]

#print(df_test.head(10))
X=df[NUMERIC_COLS]
y=df['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.3)

params = {    
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': 'auc',
          'nthread':4,
          'learning_rate':0.1,
          'num_leaves':50, 
          'max_depth': 7,   
          'subsample': 1.0, 
          'colsample_bytree': 0.6, 
    }

#data_train = lgb.Dataset(X_train, y_train)
#cv_results = lgb.cv(params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='auc',early_stopping_rounds=50,seed=0)
#print('best n_estimators:', len(cv_results['auc-mean']))
#print('best cv score:', pd.Series(cv_results['auc-mean']).max())
    
#我们可以创建lgb的sklearn模型，使用上面选择的(学习率，评估器数目)，从test1到test5依次迭代进行调参
#model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=50,
#                              learning_rate=0.1, n_estimators=1,max_depth=7,
#                              metric='rmse', bagging_fraction = 0.6,feature_fraction = 0.6)
#
#params_test1={
#    'max_depth': range(3,8,2),
#    'num_leaves':range(50, 170, 30)
#}
#gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test1, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
#gsearch1.fit(X_train, y_train)
#print(gsearch1.best_params_, gsearch1.best_score_)

#params_test2={'max_bin': range(5,256,10), 'min_data_in_leaf':range(1,102,10)}
#model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=50,
#                              learning_rate=0.1, n_estimators=1,max_depth=7,
#                              metric='rmse')      
#gsearch2 = GridSearchCV(estimator=model_lgb, param_grid=params_test2, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)        
#
#gsearch2.fit(X_train,y_train)
#print(gsearch2.best_params_, gsearch2.best_score_)


#params_test3={'feature_fraction': [0.6,0.7,0.8,0.9,1.0],
#              'bagging_fraction': [0.6,0.7,0.8,0.9,1.0],
#              'bagging_freq': range(0,81,10)
#}
#model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=50,
#                              learning_rate=0.1, n_estimators=1, max_depth=7, 
#                              metric='rmse', 
#                              max_bin=55,min_data_in_leaf=61)
#gsearch3 = GridSearchCV(estimator=model_lgb, 
#                       param_grid = params_test3, scoring='neg_mean_squared_error',cv=5,verbose=1,n_jobs=-1)
#gsearch3.fit(X_train,y_train)
#print(gsearch3.best_params_, gsearch3.best_score_)




#params_test4={'lambda_l1': [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0],
#              'lambda_l2': [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0]
#}
#model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=50,
#                              learning_rate=0.1, n_estimators=1, max_depth=7,
#                              max_bin=55,min_data_in_leaf=61,bagging_fraction=1.0,bagging_freq= 0, feature_fraction= 0.6,
#                              metric='rmse')
#gsearch4 = GridSearchCV(estimator=model_lgb, param_grid=params_test4, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
#gsearch4.fit(X_train, y_train)
#print(gsearch4.best_params_, gsearch4.best_score_)


params_test5={
    'min_split_gain':[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
}
model_lgb = lgb.LGBMRegressor(objective='regression',metric='rmse',learning_rate=0.1, n_estimators=1,
                              max_depth=7, num_leaves=50,max_bin=55,min_data_in_leaf=61,bagging_fraction=1.0,
                              bagging_freq=0,feature_fraction=0.6,lambda_l1=0.0, lambda_l2=0.0)
gsearch5 = GridSearchCV(estimator=model_lgb, param_grid=params_test5, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch5.fit(X_train, y_train)
print(gsearch5.best_params_, gsearch5.best_score_)


#model=lgb.LGBMRegressor(objective='regression',metric='rmse',learning_rate=0.1, n_estimators=1000, 
#                         max_depth=7, num_leaves=110,max_bin=105,min_data_in_leaf=61,bagging_fraction=0.6,
#                         bagging_freq= 0, feature_fraction= 0.6,
#                         lambda_l1=0.001,lambda_l2=0.5,min_split_gain=0.6)
#model.fit(X_train,y_train)
#y_pred=model.predict(X_test)
#print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)


