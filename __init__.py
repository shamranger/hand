from .lemo_pose import LEMOPose
from .multidex_shadowhand_ur import MultiDexShadowHandUR
from .lemo_motion import LEMOMotion
from .scannet_path import ScanNetPath
from .fk2plan import FK2Plan


from .hoi4d_dataset import HOI4D

def dataset_entry(config):
    return globals()[config['data']['type']](config)