#
#
# # 检查mass CSV的列名是否不同
# def check_all_csv_columns(csv_dir):
#     """检查所有CSV文件的列名"""
#
#     csv_files = {
#         'calc_train': 'calc_case_description_train_set.csv',
#         'calc_test': 'calc_case_description_test_set.csv',
#         'mass_train': 'mass_case_description_train_set.csv',
#         'mass_test': 'mass_case_description_test_set.csv'
#     }
#
#     column_info = {}
#
#     for name, filename in csv_files.items():
#         csv_path = os.path.join(csv_dir, filename)
#         if os.path.exists(csv_path):
#             df = pd.read_csv(csv_path)
#             column_info[name] = {
#                 'columns': df.columns.tolist(),
#                 'row_count': len(df)
#             }
#             print(f"\n=== {name} ===")
#             print(f"列名: {df.columns.tolist()}")
#             print(f"行数: {len(df)}")
#
#     return column_info
#
#
# # 运行检查
# column_info = check_all_csv_columns('./csv')
#







# from args import get_args
# import os
# import pandas as pd
# import numpy as np
# from breast_dataset import BreastMammoDataset  # 改为我们的数据集类
# from torch.utils.data import DataLoader
# from model import BreastModel  # 可以继续使用老师的模型
# from trainer import train_model
#
# # 快速测试修改后的模型和数据
# def test_modified_pipeline():
#     """测试修改后的模型和数据管道"""
#
#     # 测试数据加载
#     train_csv = pd.read_csv('breast_CSVs/fold_0_train.csv')
#     dataset = BreastMammoDataset(train_csv.head(4))
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
#
#     # 测试模型
#     model = BreastModel(backbone='resnet18', num_classes=3)
#
#     for batch in dataloader:
#         images = batch['img']
#         labels = batch['label']
#
#         print(f"图像形状: {images.shape}")  # 应该是 [2, 1, 512, 512]
#         print(f"标签: {labels.numpy()}")
#
#         # 测试前向传播
#         outputs = model(images)
#         print(f"模型输出形状: {outputs.shape}")  # 应该是 [2, 3]
#         print(f"模型输出: {outputs}")
#         break
#
#     print("✅ 修改后的管道测试成功!")
#
#
# test_modified_pipeline()


# # 诊断脚本：检查实际加载的图像类型
# def diagnose_image_types():
#     """诊断模型实际使用的图像类型"""
#     import pandas as pd
#     from breast_dataset import BreastMammoDataset
#     from torch.utils.data import DataLoader
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     # 加载metadata
#     metadata = pd.read_csv('official_breast_metadata.csv')
#     print(f"总样本数: {len(metadata)}")
#
#     # 分析图像路径特征
#     print("\n=== 图像路径分析 ===")
#     metadata['path_contains_full'] = metadata['image_path'].str.contains('full', case=False)
#     metadata['path_contains_roi'] = metadata['image_path'].str.contains('roi', case=False)
#     metadata['path_contains_crop'] = metadata['image_path'].str.contains('crop', case=False)
#
#     print(f"包含'full'的图像: {metadata['path_contains_full'].sum()}")
#     print(f"包含'roi'的图像: {metadata['path_contains_roi'].sum()}")
#     print(f"包含'crop'的图像: {metadata['path_contains_crop'].sum()}")
#
#     # 检查文件命名模式
#     print("\n=== 文件命名模式 ===")
#     metadata['filename'] = metadata['image_path'].apply(lambda x: x.split('/')[-1] if '/' in x else x)
#     print("常见文件名模式:")
#     print(metadata['filename'].value_counts().head(10))
#
#     return metadata
#
#
# # 运行诊断
# metadata = diagnose_image_types()


# import pandas as pd
# # 测试最终训练管道
# def test_training_pipeline():
#     """测试训练管道"""

#     from breast_dataset import BreastMammoDataset
#     from torch.utils.data import DataLoader
#     from model import BreastModel

#     # 加载数据
#     train_csv = pd.read_csv('breast_CSVs/fold_0_train.csv')
#     val_csv = pd.read_csv('breast_CSVs/fold_0_val.csv')

#     print("=== 训练管道测试 ===")
#     print(f"训练集: {len(train_csv)} 样本")
#     print(f"验证集: {len(val_csv)} 样本")

#     # 创建数据集
#     train_dataset = BreastMammoDataset(train_csv)
#     val_dataset = BreastMammoDataset(val_csv)

#     train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

#     # 创建模型
#     model = BreastModel(backbone='resnet18', num_classes=3)

#     # 测试一个batch
#     for batch in train_loader:
#         images = batch['img']
#         labels = batch['label']

#         outputs = model(images)

#         print(f"✅ 训练管道正常!")
#         print(f"图像形状: {images.shape}")
#         print(f"标签: {labels.numpy()}")
#         print(f"模型输出形状: {outputs.shape}")
#         break


# # 运行测试
# test_training_pipeline()

# rsync -avP -e "ssh -p 2162" user@ailab.samk.fi:/home/user/persistent/breast_cancer_final/breast_session "/Users/charlotteyu/Downloads/2025AutumnCourses/Image classification/"

#本地文件上传到服务器
rsync -avzP -e "ssh -p 2403" knee_xray_project.tar.gz user@ailab.samk.fi:/home/user/persistent
# import pandas as pd

# df = pd.read_csv("breast_CSVs_grouped/fold_0_train.csv")
# print(df["pathology"].value_counts())

import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))