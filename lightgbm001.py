# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 22:01:20 2019

@author: think
"""
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print('Load data...')
df=pd.read_csv(r'D:\prp\new_data.csv')

NUMERIC_COLS = [
    '1','2','3','4','5','6','7','8','9','10',
    '11','12','13','14','15','16','17','18','19','20','n'
]

#print(df_test.head(10))
X=df[NUMERIC_COLS]
y=df['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#去掉了random_state=0，在不同的划分方式下rmse维持在300-500范围

# 创建成lgb特征的数据集格式
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
 
# 将参数写成字典下形式
#params = {
#    'task': 'train',
#    'boosting_type': 'gbdt',  # 设置提升类型
#    'objective': 'regression',  # 目标函数
#    'metric': {'l2', 'auc'},  # 评估函数
#    'num_leaves':125 ,  # 叶子节点数
#    'max_depth':7,
#    'num_trees':500,
#    'learning_rate': 0.1,  # 学习速率
#    'feature_fraction': 0.55,  # 建树的特征选择比例
#    'bagging_fraction': 0.92,  # 建树的样本采样比例
#    'bagging_freq': 10,  # k 意味着每 k 次迭代执行bagging
#    'verbose': 1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
#    'min_data_in_leaf':31,
#    'max_bin':105,
#    'lambda_l1': 0.0, 
#    'lambda_l2': 0.0,
#    'min_split_gain': 0.5
#}
#The rmse of training prediction is: 1.4828631594872625
#The rmse of prediction is: 2.3090931965477473

params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'regression',  # 目标函数
    'metric': {'l2', 'auc'},  # 评估函数
    'num_leaves':110 ,  # 叶子节点数
    'max_depth':7,
    'num_trees':300,
    'learning_rate': 0.1,  # 学习速率
    'feature_fraction': 0.6,  # 建树的特征选择比例
    'bagging_fraction': 0.6,  # 建树的样本采样比例
    'bagging_freq': 0,  # k 意味着每 k 次迭代执行bagging
    'verbose': 1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    'min_data_in_leaf':61,
    'max_bin':105,
    'lambda_l1': 0.001, 
    'lambda_l2': 0.5,
    'min_split_gain': 0.6
}
#The rmse of training prediction is: 1.3489468748718778
#The rmse of prediction is: 2.25666226243407


# 训练 cv and train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=lgb_train)
# 保存模型到文件
gbm.save_model('model lightgbm001.txt')
 
# 预测数据集
y_pred_train= gbm.predict(X_train, num_iteration=gbm.best_iteration)
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
 
# 评估模型
print('The rmse of training prediction is:', mean_squared_error(y_train, y_pred_train) ** 0.5)
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

#画出预测值与实际值
plt.scatter(y_test,y_pred)
plt.show()
