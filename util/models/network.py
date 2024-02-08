import torch
from torch.nn import init

from util.models.baseline.unet import Unet
from util.models.experiment.mv import MV

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




