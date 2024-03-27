from yacs.config import CfgNode as CN
import os

_C = CN(new_allowed=True)


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    defaults_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'configs.yaml'))
    _C.merge_from_file(defaults_abspath)
    _C.set_new_allowed(False)
    return _C.clone()

def load_cfg(path=None):
    _C.merge_from_file(path)
    _C.set_new_allowed(False)
    return _C.clone()
