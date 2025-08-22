import os
os.chdir('mmsegmentation')

from mmengine import Config
cfg = Config.fromfile('./configs/knet/knet-s3_swin-l_upernet_8xb2-adamw-80k_ade20k-512x512.py')

dataset_cfg = Config.fromfile('./configs/_base_/datasets/UpsDataset_pipeline.py')
cfg.merge_from_dict(dataset_cfg)

# 类别个数
NUM_CLASS = 14

# 单卡训练时，需要把 SyncBN 改成 BN
cfg.norm_cfg = dict(type='BN', requires_grad=True) # 只使用GPU时，BN取代SyncBN
cfg.model.data_preprocessor.size = cfg.crop_size

# 模型 decode/auxiliary 输出头，指定为类别个数
# cfg.model.decode_head.num_classes = NUM_CLASS
cfg.model.decode_head.kernel_generate_head.num_classes = NUM_CLASS
cfg.model.auxiliary_head.num_classes = NUM_CLASS

# 训练 Batch Size
cfg.train_dataloader.batch_size = 4

# 结果保存目录
cfg.work_dir = './work_dirs/UpsDataset-KNet'

cfg.train_cfg.max_iters = 20000 # 训练迭代次数
cfg.train_cfg.val_interval = 500 # 评估模型间隔
cfg.default_hooks.logger.interval = 100 # 日志记录间隔
cfg.default_hooks.checkpoint.interval = 2500 # 模型权重保存间隔
cfg.default_hooks.checkpoint.max_keep_ckpts = 2 # 最多保留几个模型权重
cfg.default_hooks.checkpoint.save_best = 'mIoU' # 保留指标最高的模型权重

# 随机数种子
cfg['randomness'] = dict(seed=0)

cfg.dump('MyConfigs/UpsDataset_KNet.py')
