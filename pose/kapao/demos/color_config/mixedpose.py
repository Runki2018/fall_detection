dataset_info = dict(
    dataset_name='mixedpose',
    paper_info=dict(
        author='huangzhiyong',
        title='mixed human pose data, including coco, crowdpose, mpii, and lsp.',
        container='None',
        year='2022',
        homepage='None',
    ),
    keypoint_info={
        0:
        dict(
            name='left_shoulder',
            id=5,
            color=[255, 0, 0],
            type='upper',
            swap='right_shoulder'),
        1:
        dict(
            name='right_shoulder',
            id=6,
            color=[0, 255, 0],
            type='upper',
            swap='left_shoulder'),
        2:
        dict(
            name='left_elbow',
            id=7,
            color=[255, 0, 0],
            type='upper',
            swap='right_elbow'),
        3:
        dict(
            name='right_elbow',
            id=8,
            color=[0, 255, 0],
            type='upper',
            swap='left_elbow'),
        4:
        dict(
            name='left_wrist',
            id=9,
            color=[255, 0, 0],
            type='upper',
            swap='right_wrist'),
        5:
        dict(
            name='right_wrist',
            id=10,
            color=[0, 255, 0],
            type='upper',
            swap='left_wrist'),
        6:
        dict(
            name='left_hip',
            id=11,
            color=[255, 0, 0],
            type='lower',
            swap='right_hip'),
        7:
        dict(
            name='right_hip',
            id=12,
            color=[0, 255, 0],
            type='lower',
            swap='left_hip'),
        8:
        dict(
            name='left_knee',
            id=13,
            color=[255, 0, 0],
            type='lower',
            swap='right_knee'),
        9:
        dict(
            name='right_knee',
            id=14,
            color=[0, 255, 0],
            type='lower',
            swap='left_knee'),
        10:
        dict(
            name='left_ankle',
            id=15,
            color=[255, 0, 0],
            type='lower',
            swap='right_ankle'),
        11:
        dict(
            name='right_ankle',
            id=16,
            color=[0, 255, 0],
            type='lower',
            swap='left_ankle')
    },
    skeleton_info={
        0:
        dict(link=('left_shoulder', 'right_shoulder'), id=0, color=[255, 255, 255]),
        1:
        dict(link=('left_hip', 'right_hip'), id=1, color=[255, 255, 255]),

        2:
        dict(link=('right_shoulder', 'right_hip'), id=2, color=[0, 0, 255]),
        3:
        dict(link=('right_shoulder', 'right_elbow'), id=3, color=[255, 0, 0]),
        4:
        dict(link=('right_elbow', 'right_wrist'), id=4, color=[0, 255, 0]),
        5:
        dict(link=('right_knee', 'right_hip'), id=5, color=[0, 255, 255]),
        6:
        dict(link=('right_ankle', 'right_knee'), id=6, color=[255, 0, 255]),

        7:
        dict(link=('left_shoulder', 'left_hip'), id=7, color=[0,0,255]),
        8:
        dict(link=('left_shoulder', 'left_elbow'), id=8, color=[255,0,0]),
        9:
        dict(link=('left_elbow', 'left_wrist'), id=9, color=[0,255,0]),
        10:
        dict(link=('left_knee', 'left_hip'), id=10, color=[0,255,255]),
        11:
        dict(link=('left_ankle', 'left_knee'), id=11, color=[255,0,255]),
    },
    joint_weights=[
        1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5
    ],
    sigmas=[
        0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
    ])