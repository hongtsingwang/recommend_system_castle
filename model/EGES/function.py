'''
Author: your name
Date: 2022-01-23 13:56:19
LastEditTime: 2022-01-23 13:56:19
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /recommend_system_castle/model/EGES/function.py
'''
def basic_loss_function(y_true, y_pred):
    return tf.math.reduce_mean(y_pred)