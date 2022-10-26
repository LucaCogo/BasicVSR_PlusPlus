import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from mmedit.models.common import flow_warp


class Network(torch.nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        class Preprocess(torch.nn.Module):
            def __init__(self):
                super().__init__()
            # end

            def forward(self, tenInput):
                tenInput = tenInput.flip([1])
                tenInput = tenInput - torch.tensor(data=[0.485, 0.456, 0.406], dtype=tenInput.dtype, device=tenInput.device).view(1, 3, 1, 1)
                tenInput = tenInput * torch.tensor(data=[1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225], dtype=tenInput.dtype, device=tenInput.device).view(1, 3, 1, 1)

                return tenInput
            # end
        # end

        class Basic(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()

                self.netBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                )
            # end

            def forward(self, tenInput):
                return self.netBasic(tenInput)
            # end
        # end

        self.netPreprocess = Preprocess()

        self.netBasic = torch.nn.ModuleList([ Basic(intLevel) for intLevel in range(6)])

        # self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-spynet/network-' + MODEL + '.pytorch', file_name='spynet-' + MODEL).items() })
        

        # if MODEL == 'mmediting':
        #     model_dict = {format_model_dict_key(strKey): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth', file_name='spynet-' + MODEL).items()}
        #     self.load_state_dict(model_dict)
        
        # else:
        self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load(pretrained, map_location=torch.device('cuda')).items()})
            # print({strKey.replace('basic_net', 'netBasic'): tenWeight for strKey, tenWeight in torch.load.hub.load_state_dict_from_url("models/network-" + MODEL + ".pytorch", map_location=torch.device(DEVICE)).items()})

            # self.load_state_dict({strKey.replace('basic_net', 'netBasic'): tenWeight for strKey, tenWeight in torch.load.hub.load_state_dict_from_url("models/network-" + MODEL + ".pytorch", map_location=torch.device(DEVICE)).items()})

        self.backwarp_tenGrid = {}
    # end

    def forward(self, tenOne, tenTwo):
        tenFlow = []
        self.backwarp_tenGrid = {}

        tenOne = [ self.netPreprocess(tenOne) ]
        tenTwo = [ self.netPreprocess(tenTwo) ]

        for intLevel in range(5):
        # if tenOne[0].shape[2] > 32 or tenOne[0].shape[3] > 32:
            tenOne.insert(0, torch.nn.functional.avg_pool2d(input=tenOne[0], kernel_size=2, stride=2, count_include_pad=False))
            tenTwo.insert(0, torch.nn.functional.avg_pool2d(input=tenTwo[0], kernel_size=2, stride=2, count_include_pad=False))
            
        

        tenFlow = tenOne[0].new_zeros([ tenOne[0].shape[0], 2, int(math.floor(tenOne[0].shape[2] / 2.0)), int(math.floor(tenOne[0].shape[3] / 2.0)) ])

        for intLevel in range(len(tenOne)):
            tenUpsampled = torch.nn.functional.interpolate(input=tenFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if tenUpsampled.shape[2] != tenOne[intLevel].shape[2]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 0, 0, 1 ], mode='replicate')
            if tenUpsampled.shape[3] != tenOne[intLevel].shape[3]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')

            # tenFlow = self.netBasic[intLevel](torch.cat([ tenOne[intLevel], self.backwarp(tenInput=tenTwo[intLevel], tenFlow=tenUpsampled), tenUpsampled ], 1)) + tenUpsampled
            tenFlow = self.netBasic[intLevel](torch.cat([ tenOne[intLevel], flow_warp(tenTwo[intLevel], tenUpsampled.permute(0,2,3,1))], 1)) + tenUpsampled
            

        return tenFlow
    # end

    
    def backwarp(self, tenInput, tenFlow):

        tenInput.cuda()
        tenFlow.cuda()
        if str(tenFlow.shape) not in self.backwarp_tenGrid:
            tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
            tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])
            
            self.backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1)
        # end

        tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

        return torch.nn.functional.grid_sample(input=tenInput, grid=(self.backwarp_tenGrid[str(tenFlow.shape)].cuda() + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=False)


# netNetwork = None



# def estimate(tenOne, tenTwo):

#     global netNetwork

#     if netNetwork is None:
#         netNetwork = Network().eval()


#     assert(tenOne.shape[2] == tenTwo.shape[2])
#     assert(tenOne.shape[3] == tenTwo.shape[3])

#     intWidth = tenOne.shape[3]
#     intHeight = tenOne.shape[2]

#     # assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
#     # assert(intHeight == 416) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

#     # tenPreprocessedOne = tenOne.view(1, 3, intHeight, intWidth)
#     # tenPreprocessedTwo = tenTwo.view(1, 3, intHeight, intWidth)

#     intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
#     intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

#     tenPreprocessedOne = torch.nn.functional.interpolate(input=tenOne, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
#     tenPreprocessedTwo = torch.nn.functional.interpolate(input=tenTwo, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

#     tenFlow = torch.nn.functional.interpolate(input=netNetwork(tenPreprocessedOne, tenPreprocessedTwo), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

#     tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
#     tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

#     return tenFlow[:, :, :, :]