exp_name = 'basicvsraft_s_reduced_precomp_flow_bs_1_1k_iters'

# model settings
model = dict(
    type='BasicVSR',
    generator=dict(
        type='BasicVSRAFT_precomp',
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=True,
        small = True),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean')
    )

model_name = "rafts" if model['generator']["small"] else "raft"

# model training and testing settings
train_cfg = dict(fix_iter=5000)
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)

# dataset settings
train_dataset_type = 'SRFolderPrecomputedFlowDataset'
val_dataset_type = 'SRFolderPrecomputedFlowDataset'
test_dataset_type = 'SRFolderPrecomputedFlowDataset'

train_pipeline = [
    dict(type='GenerateSegmentIndicesPrecomp', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(
        type='LoadNPYFromFileList',
        io_backend='disk',
        key='of_b' # backward optical flow
    ),
    dict(
        type='LoadNPYFromFileList',
        io_backend='disk',
        key='of_f' # forward optical flow
    ),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='QuadrupleRandomCrop', gt_patch_size=256),
    dict(
        type='Flip', keys=['lq', 'gt', 'of_b','of_f'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt', 'of_b', 'of_f'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt', 'of_b', 'of_f'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt', 'of_b', 'of_f'], meta_keys=['lq_path', 'gt_path', 'of_b_path', 'of_f_path'])
]

test_pipeline = [
    dict(type='GenerateSegmentIndicesPrecomp', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(
        type='LoadNPYFromFileList',
        io_backend='disk',
        key='of_b' # backward optical flow
    ),
    dict(
        type='LoadNPYFromFileList',
        io_backend='disk',
        key='of_f' # forward optical flow
    ),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt', 'of_b', 'of_f'],
        meta_keys=['lq_path', 'gt_path', 'of_b_path', 'of_f_path', 'key'])
]

demo_pipeline = [
    dict(type='GenerateSegmentIndicesPrecomp', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
]

data = dict(
    workers_per_gpu=6,
    train_dataloader=dict(samples_per_gpu=1, drop_last=True),  # 8 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='../REDS_ridotto/lq_sequences_train',
            gt_folder='../REDS_ridotto/gt_sequences_train',
            of_b_folder='../REDS_ridotto/lq_sequences_train_of/' + model_name + "/backward",
            of_f_folder='../REDS_ridotto/lq_sequences_train_of/' + model_name + "/forward",
            num_input_frames=30,
            pipeline=train_pipeline,
            scale=4,
            test_mode=False)),
    # val
    val=dict(
        type=val_dataset_type,
        lq_folder='../REDS_ridotto/lq_sequences_val',
        gt_folder='../REDS_ridotto/gt_sequences_val',
        of_b_folder='../REDS_ridotto/lq_sequences_val_of/' + model_name + "/backward",
        of_f_folder='../REDS_ridotto/lq_sequences_val_of/' + model_name + "/forward",
        num_input_frames=100,
        pipeline=test_pipeline,
        scale=4,
        test_mode=True),
    # test
    test=dict(
        type=test_dataset_type,
        lq_folder='../REDS_ridotto/lq_sequences_test',
        gt_folder='../REDS_ridotto/gt_sequences_test',
        pipeline=test_pipeline,
        scale=4,
        test_mode=True),
)

# optimizer
optimizers = dict(
    generator=dict(
        type='Adam',
        lr=1e-4,
        betas=(0.9, 0.99),
        # paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.25)})
        )
    )

# learning policy
total_iters = 1000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[1000],
    restart_weights=[1],
    min_lr=1e-7)

checkpoint_config = dict(interval=1000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=100, save_image=False, gpu_collect=True)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True

