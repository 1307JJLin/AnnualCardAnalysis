# 2022年度数据分析报告 ｜杰然不同之GR


# 本报告由公众号【杰然不同之GR】编写，整理并发布，仅作个人学习使用，请勿用于任何商业用途，转载及其他形式合作请与我联系 

简易目录

1.数据预处理
   
1.1 数据合并和一些纠错

1.2 新特征生成

1.2.1) 根据date生成月日还有星期

1.2.2) 根据start time 和end time 生成时和分

1.2.3) 根据start和end生成中间时间

1.2.4) 根据事项的中间时间确定事项做的时候处于一天的时段

1.2.5) 根据时长大小分为长中短

1.2.6) 修改一些前后差别不大的事件

1.2.7) 统一事项与属性的对应关系 

2.数据可视化

2.1 变量分类

2.1.1) 日期类变量: date

2.1.2) 数值类变量: duration, phone 

2.1.3) 类别变量: event, year, month, day, weekday, mid_hour, day_period, week_order, duration_attr, attr, 

2.2 单变量分析

2.2.1) 数值变量的直方图

2.2.2) 类别变量的饼图，柱形图

2.2.3) 专注力分析

2.2.4) 各个月份，各年, 各小时的event关键词词云图

2.3 双变量分析

2.3.1) PyCatFlow图

2.3.2) 单个类别变量 vs 数值 sns.stripplot  sns.swarmplot  sns.boxplot   sns.violinplot  

2.3.3) 两个类别变量 VS 数值 sns.countplot  sns.barplot  sns.factorplot  sns.pointplot 

2.4 衍生数据可视化

2.4.1) 日度汇总各类时间总时长

2.4.2) 每天的第一件和最后一件事

2.4.3) 每天的第一件和最后一件事的时间

2.4.4) 每天做的事项数目的统计

2.4.5) 每天做事项先后编号并统计

2.4.6) 各事项关于日期求和的透视表

2.4.7) 每年, 月, 星期的事项关键词词云图

2.4.8) 每天事项的对应先后以及月度关联的网络图

2.5 一键式EDA

3.数据分析

3.1 方差分析每月或是每年的事项是否有差异

3.1.1) 原始事项时长数据的方差分析

3.1.2) 衍生数据的方差分析

3.2 LSTM预测前后事项

3.2.1) 建立字符索引

3.2.2) 建立LSTM模型

3.2.3) 模型训练

3.3 时长数据的简单聚类

3.4 关于日期的缺失值处理

3.5 关于日期的傅里叶分解求周期

3.6 关于日期的传统时间序列分析

3.6.1) ARIMA分析

3.6.2) GARCH分析

3.6.3) VAR分析

3.7 关于日期的NeuralProphet 时间序列

4.数据挖掘

4.1 通过生成聚合特征做日度时长数据的特征工程

4.2 类别变量的one-hot encoding

4.3 利用boruta筛选特征

4.4 利用PCA降维

4.5 利用optuna优化参数

4.6 stacking 融合

# 更多学习笔记，请关注公众号【杰然不同之GR】 #
![公众号二维码](http://r.photo.store.qq.com/psc?/V10twqic2oh0r6/TmEUgtj9EK6.7V8ajmQrEG6xI.X7icgy*l8zd9O9qB3X6.AQyIe0uOSHtI7ti9nULpRDinQuLz61UqAz2Qxai3hPPThtIDGcEZu3WfoP84I!/r)
