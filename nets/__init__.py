from .mobilenet import mobilenet_v2
from .resnet50 import resnet50
from .vgg16 import vgg16
from .vit import vit
from .googlenet import googlenet
from .shufflenet import shufflenetv2
from .inception import inception_v3
from .squeezenet import squeezenet
from .efficientnet import efficientnet
from .densenet   import densenet
from .alexnet    import alexnet

class ModelType:
    mobilenet   = "mobilenet"
    resnet50    = "resnet50" 
    vgg16       = "vgg16"
    vit         = "vit" 
    googlenet   = "googlenet" 
    shufflenet  = "shufflenet"
    inception   = "inception"
    squeezenet  = "squeezenet"
    efficientnet = "efficientnet"
    densenet     = "densenet"
    alexnet      = "alexnet"


get_model_from_name = {
    "mobilenet"     : mobilenet_v2,
    "resnet50"      : resnet50,
    "vgg16"         : vgg16,
    "vit"           : vit,
    "googlenet"     : googlenet,
    "shufflenet"    : shufflenetv2,
    "inception"     : inception_v3,
    "squeezenet"    : squeezenet,
    "efficientnet"  : efficientnet,
    "densenet"      : densenet,
    "alexnet"       : alexnet,
}