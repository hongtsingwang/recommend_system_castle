# coding=utf-8

import os
import sys

from gmf_model import GmfModel

"""
GMF 模型： 即广义矩阵分解模型， 是一个非常初级，简单的模型
如果你想对矩阵分解模型有了解， 可以参考下面三篇文章
1-
推荐系统之矩阵分解模型-科普篇 - 腾讯技术工程的文章 - 知乎
https://zhuanlan.zhihu.com/p/69662980
2-
推荐系统之矩阵分解模型-原理篇 - 腾讯技术工程的文章 - 知乎
https://zhuanlan.zhihu.com/p/360689325
3-
推荐系统之矩阵分解模型-实践篇 - 腾讯技术工程的文章 - 知乎
https://zhuanlan.zhihu.com/p/363259363
"""

gmf_model = GmfModel(user_num, item_num, dim=32)
