import torch
import torch.nn as nn
from pathlib import Path
from thirdparty.lightglue.lightglue import LightGlue


class LightGlueMatcher(nn.Module):
    def __init__(self, device = 'cuda', feature_type = 'superpoint'):
        super().__init__()
        self.device = device
        self.model = LightGlue(features=feature_type).to(device).eval()

    @torch.no_grad()
    def forward(self, data0, data1):
        '''
        superpoint outputs : {'keypoints': [B, M, 2], 'descriptors': [B, M, D], 'image_size': [B, 2]}
        '''
        pred = self.model({'image0': data0, 'image1': data1})
        
        return {
            'matches': pred['matches'],          
            'scores': pred['scores'],            
            'stop_layer': pred['stop'],       
            'prune0': pred['prune0'],             
            'prune1': pred['prune1']              
        }