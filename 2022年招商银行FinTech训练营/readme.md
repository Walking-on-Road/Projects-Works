# 招商银行第六季FinTech精英训练营线上竞赛

## 比赛介绍
赛题开放A榜数据（test_A榜），预测结果数据每天限提交3次；5月10日00:00-5月12日17:00，赛题开放B榜数据（test_B榜），预测结果数据每天限提交3次。

## 赛题背景
公司存款是商业银行以信用方式吸收的企事业单位的存款，与零售存款相比，公司存款具备数额大、成本低、流动性强等特点。通过对公司存款流失的预测，银行一方面可以对公司客户的流失原因进行归因，不断提升服务水平；另一方面可以提前规划资产负债结构，保证充裕的流动性。因此，使用金融科技手段对公司存款流失进行预测具有重要的业务指导意义。

## 课题研究要求
本次比赛为参赛选手提供了两个数据集，即训练数据集（train）和测试数据集（test_A榜/ test_B榜）。参赛选手需要基于训练数据集，通过有效的特征提取，使用分类算法构建公司客户存款流失预测模型，并将模型应用在测试数据集上，输出对测试数据集中公司客户存款流失概率的预测。

## 模型评价指标
TPR为召回率，即在所有正样本中被模型识别为正样本的比例；FPR为拒真率，即在所有正样本中被模型识别为负样本的比例。以TPR为纵轴，FPR为横轴，将不同取值下的TPR和FPR通过二维坐标系中的曲线图形式表现出来，就是ROC曲线。

## 数据说明
训练数据集中共包含4万条脱敏数据，CUST_UID为公司客户唯一标识，LABEL为公司客户存款是否流失的标志，其中1表示该公司客户在三个月后存款发生流失，0表示该公司客户在三个月后存款未发生流失。后续的数据列是每个公司客户当月的数据情况，可作为模型使用的变量，需要注意的是，不是所有变量都适合入模，参赛选手需要对变量进行过滤和筛选。\
测试数据集中包含1.2万条脱敏数据 ，CUST_UID为公司客户唯一标识（训练集和测试集中若存在相同的CUST_UID，不代表同一用户），其余数据列为每个公司客户当月的数据情况。

# 竞赛结果
***比赛共参与人数为1855人，最终比赛排名为318名，比赛成绩在前20%。***
