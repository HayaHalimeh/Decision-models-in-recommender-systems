from .crossenc import CrossEnc
from .embsim import EmbSim
from .itemcls import ItemCLS

CLASSES = {"embsim": EmbSim, "crossenc": CrossEnc, "itemcls": ItemCLS}


def build(mcfg, cfg):
    return CLASSES[mcfg["name"]](mcfg, cfg)
