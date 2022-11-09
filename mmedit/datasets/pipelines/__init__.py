# Copyright (c) OpenMMLab. All rights reserved.
from .augmentation import (BinarizeImage, ColorJitter, CopyValues, Flip,
                           GenerateFrameIndices,
                           GenerateFrameIndiceswithPadding,
                           GenerateSegmentIndices,GenerateSegmentIndicesPrecomp, MirrorSequence, Pad,
                           Quantize, RandomAffine, RandomJitter,
                           RandomMaskDilation, RandomTransposeHW, Resize,
                           TemporalReverse, UnsharpMasking)
from .compose import Compose
from .crop import (Crop, CropAroundCenter, CropAroundFg, CropAroundUnknown,
                   CropLike, FixedCrop, ModCrop, PairedRandomCrop, QuadrupleRandomCrop,
                   RandomResizedCrop)
from .formating import (Collect, FormatTrimap, GetMaskedImage, ImageToTensor,
                        ToTensor)
from .generate_assistant import GenerateCoordinateAndCell, GenerateHeatmap
from .loading import (GetSpatialDiscountMask, LoadImageFromFile,
                      LoadImageFromFileList, LoadNPYFromFile, LoadNPYFromFileList, LoadMask, LoadPairedImageFromFile,
                      RandomLoadResizeBg)
from .matlab_like_resize import MATLABLikeResize
from .matting_aug import (CompositeFg, GenerateSeg, GenerateSoftSeg,
                          GenerateTrimap, GenerateTrimapWithDistTransform,
                          MergeFgAndBg, PerturbBg, TransformTrimap)
from .normalization import Normalize, RescaleToZeroOne
from .random_degradations import (DegradationsWithShuffle, RandomBlur,
                                  RandomJPEGCompression, RandomNoise,
                                  RandomResize, RandomVideoCompression)
from .random_down_sampling import RandomDownSampling

__all__ = [
    'Collect', 'FormatTrimap', 'LoadImageFromFile', 'LoadNPYFromFile', 'LoadMask',
    'RandomLoadResizeBg', 'Compose', 'ImageToTensor', 'ToTensor',
    'GetMaskedImage', 'BinarizeImage', 'Flip', 'Pad', 'RandomAffine',
    'RandomJitter', 'ColorJitter', 'RandomMaskDilation', 'RandomTransposeHW',
    'Resize', 'RandomResizedCrop', 'CenterCrop', 'Crop', 'CropAroundCenter',
    'CropAroundUnknown', 'ModCrop', 'PairedRandomCrop','QuadrupleRandomCrop', 'Normalize',
    'RescaleToZeroOne', 'GenerateTrimap', 'MergeFgAndBg', 'CompositeFg',
    'TemporalReverse', 'LoadImageFromFileList', 'LoadNPYFromFileList', 'GenerateFrameIndices',
    'GenerateFrameIndiceswithPadding', 'FixedCrop', 'LoadPairedImageFromFile',
    'GenerateSoftSeg', 'GenerateSeg', 'PerturbBg', 'CropAroundFg',
    'GetSpatialDiscountMask', 'RandomDownSampling',
    'GenerateTrimapWithDistTransform', 'TransformTrimap',
    'GenerateCoordinateAndCell', 'GenerateSegmentIndices', 'GenerateSegmentIndicesPrecomp', 'MirrorSequence',
    'CropLike', 'GenerateHeatmap', 'MATLABLikeResize', 'CopyValues',
    'Quantize', 'RandomBlur', 'RandomJPEGCompression', 'RandomNoise',
    'DegradationsWithShuffle', 'RandomResize', 'UnsharpMasking',
    'RandomVideoCompression', 'CropSequence'
]
