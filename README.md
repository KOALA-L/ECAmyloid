# ECAmyloid

## dataset:
  
  positive_dataset.fasta
  
  negative_dataset.fasta
 
  分别为阳性数据集和阴性数据集。

## Code:
  feature_acid.py：将溶剂可及性的结果，包含B,b,E,e的特征向量转换为数值向量。
  
  feature_protein.py：将蛋白质二级结构信息的结果，包含C,H,I的特征向量转换为数值向量。
  
  train_model.py：通过十折交叉验证训练模型，并计算各个指标以及ROC曲线。
  
