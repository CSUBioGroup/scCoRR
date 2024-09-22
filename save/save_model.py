import os
import torch
from collections import OrderedDict

def save_model(model_path, model, optimizer, current_epoch):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    out = os.path.join(model_path, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    
    torch.save(state, out)
    

