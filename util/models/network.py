import torch
from torch.nn import init

from util.models.baseline.unet import Unet
# from util.models.baseline.drunet import DRUnet
# from util.models.baseline.seunet import SEUnet
# from util.models.baseline.rrunet import RRUnet
# from util.models.baseline.resunet import ResUnet
# from util.models.baseline.attunet import AttUnet
# from util.models.baseline.attrrunet import AttRRUnet



from util.models.experiment.mv import MV
# from util.models.experiment.mv_se import MV_SE
# from util.models.experiment.mv_se_res import MV_SE_RES
#from util.models.experiment.mv_f import MV_F
# from util.models.experiment.mv_eval import MV_eval

# from util.models.experiment.tlunet import TLUnet

# from util.models.experiment.burger_unet.burger import Burger
# from util.models.experiment.tensorly_unet import TensorlyUnet
# from util.models.experiment.tensorlyse_unet import TensorlySEUnet
# from util.models.experiment.burger_unet_tcl.burger_tcl import Burger_TCL
# from util.models.experiment.burger_unet_plus.burger_plus import Burger_Plus



def get_model(name, device, level=0):

    if name[0] == "M":
        model = globals()[name](level)
    else:
        model =  globals()[name]()
    
    model.to(device)
    # test model
    model.eval()
    test_input = torch.rand(2,1,512,512).to(device)
    prediction = model(test_input.float())
    assert prediction.shape == test_input.shape, f"weird output shape: {prediction.shape}"

    # Weights initialization
    model.apply(_weights_init) 

    return model

    
def _weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):

        init.xavier_normal_(m.weight.data, gain=0.02)

        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)




