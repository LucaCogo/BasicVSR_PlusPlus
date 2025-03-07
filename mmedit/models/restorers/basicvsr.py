# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os.path as osp

import mmcv
import numpy as np
import torch

from mmedit.core import tensor2img
from ..registry import MODELS
from .basic_restorer import BasicRestorer


@MODELS.register_module()
class BasicVSR(BasicRestorer):
    """BasicVSR model for video super-resolution.

    Note that this model is used for IconVSR.

    Paper:
        BasicVSR: The Search for Essential Components in Video Super-Resolution
        and Beyond, CVPR, 2021

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        ensemble (dict): Config for ensemble. Default: None.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 generator,
                 pixel_loss,
                 ensemble=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(generator, pixel_loss, train_cfg, test_cfg,
                         pretrained)

        # fix pre-trained networks
        self.fix_iter = train_cfg.get('fix_iter', 0) if train_cfg else 0
        self.is_weight_fixed = False

        # count training steps
        self.register_buffer('step_counter', torch.zeros(1))

        # ensemble
        self.forward_ensemble = None
        if ensemble is not None:
            if ensemble['type'] == 'SpatialTemporalEnsemble':
                from mmedit.models.common.ensemble import \
                    SpatialTemporalEnsemble
                is_temporal = ensemble.get('is_temporal_ensemble', False)
                self.forward_ensemble = SpatialTemporalEnsemble(is_temporal)
            else:
                raise NotImplementedError(
                    'Currently support only '
                    '"SpatialTemporalEnsemble", but got type '
                    f'[{ensemble["type"]}]')

    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                is_mirror_extended = True

        return is_mirror_extended

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        # fix SPyNet and EDVR at the beginning
        if self.step_counter < self.fix_iter:
            if not self.is_weight_fixed:
                self.is_weight_fixed = True

                for k, v in self.generator.named_parameters():
                    # if 'spynet' in k or 'edvr' in k or 'raft' in k:
                    if True in [x in k for x in ['raft', 'spynet', 'edvr','pwc', 'lfn']]: # check if one of these substring is in k
                        v.requires_grad_(False)
        elif self.step_counter == self.fix_iter:
            # train all the parameters
            print("All the parameters (also the optical flow ones) will be trained from now on")
            self.generator.requires_grad_(True)

        outputs = self(**data_batch, test_mode=False) 
        output = outputs['results']['output']
        gt = outputs['results']['gt']

        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        self.step_counter += 1

        outputs.update({'log_vars': log_vars})
        return outputs

    def rgb2ycbcr(self, img, y_only=False):
        """Convert a RGR image to YCbCr image.
        The bgr version of rgb2ycbcr.
        input is mini-batch T x N x 3 x H x W of a RGB image
        It implements the ITU-R BT.601 conversion for standard-definition
        television. See more details in
        https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
        """
        output = torch.zeros_like(img).cuda()
        output[:, :,0, :, :] = img[:, :,0, :, :] * 65.48 + img[:, :,1, :, :] * 128.553 + img[:, :,2, :, :] * 24.966 + 16.0
        
        if y_only:
            return output[:, :,0, :, :]
        
        output[:, :,1, :, :] = img[:, :,0, :, :] * (-37.797) + img[:, :,1, :, :] * (-74.203) + img[:, :,2, :, :] * 112.0 + 128.0
        output[:, :,2, :, :] = img[:, :,0, :, :] * 112.0 + img[:, :,1, :, :] * (-93.786) + img[:, :,2, :, :] * (-18.214) + 128.0

        return output

    def compute_psnr(self, img, gt, convert_to):
        L = 1.
        if isinstance(convert_to, str) and convert_to.lower() == 'y':
            L = 255.
            img = self.rgb2ycbcr(img, y_only = True)
            gt = self.rgb2ycbcr(gt, y_only = True)

        elif convert_to is not None:
            raise ValueError('Wrong color model. Supported values are '
                '"Y" and None.')
        
        mse = torch.mean(torch.square(img[0] - gt[0]))
        psnr = 20. * torch.log10(L / torch.sqrt(mse))
        return psnr


    def compute_ssim(self, img, gt, convert_to):
        L = 1.
        if isinstance(convert_to, str) and convert_to.lower() == 'y':
            L = 255.
            img = self.rgb2ycbcr(img, y_only = True)
            gt = self.rgb2ycbcr(gt, y_only = True)

        elif convert_to is not None:
            raise ValueError('Wrong color model. Supported values are '
                '"Y" and None.')

        ssims = []
        for i in range(img.shape[2]):
            ssims.append(_ssim(img[..., i], gt[..., i]))
        return np.array(ssims).mean()

    def _ssim(img1, img2):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) *
                    (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                           (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()




    def evaluate(self, output, gt):
        """Evaluation function.

        If the output contains multiple frames, we compute the metric
        one by one and take an average.

        Args:
            output (Tensor): Model output with shape (n, t, c, h, w).
            gt (Tensor): GT Tensor with shape (n, t, c, h, w).

        Returns:
            dict: Evaluation results.
        """

        crop_border = self.test_cfg.crop_border
        convert_to = self.test_cfg.get('convert_to', None)

        eval_result = dict()
        # for metric in self.test_cfg.metrics:
        #     if output.ndim == 5:  # a sequence: (n, t, c, h, w)
        #         avg = []
        #         for i in range(0, output.size(1)):
        #             output_i = tensor2img(output[:, i, :, :, :])
        #             gt_i = tensor2img(gt[:, i, :, :, :])
        #             avg.append(self.allowed_metrics[metric](
        #                 output_i, gt_i, crop_border, convert_to=convert_to))
                    
        #         eval_result[metric] = np.mean(avg)
        #     elif output.ndim == 4:  # an image: (n, c, t, w), for Vimeo-90K-T
        #         output_img = tensor2img(output)
        #         gt_img = tensor2img(gt)
        #         value = self.allowed_metrics[metric](
        #             output_img, gt_img, crop_border, convert_to=convert_to)
        #         eval_result[metric] = value

        if output.ndim == 5: # a sequence: (n, t, c, h, w)
            psnr = self.compute_psnr(output, gt, convert_to)
            ssim = self.compute_ssim(output, gt, convert_to)
            eval_result['PSNR'] = psnr.item()
            eval_result['SSIM'] = ssim.item()

        elif output.ndim == 4: # an image: (n, c, t, w), for Vimeo-90K-T
            pass
        
        #loss = self.pixel_loss(output, gt)
        #eval_result['pixel_loss'] = loss.item()

        return eval_result

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None,
                     **kwargs):
        """Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, t, c, h, w).
            gt (Tensor): GT Tensor with shape (n, t, c, h, w). Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
        """
        with torch.no_grad():
            if 'of_b' in [k for k in kwargs.keys()] and 'of_f' in [k for k in kwargs.keys()]:
                if self.forward_ensemble is not None:
                  raise NotImplementedError(
                    'Currently the RAFT precomp optical flow does not'
                    'support SpatialTemporalEnsemble'
                  )
                else:
                  output = self.generator(lq, **kwargs)
            else:
                if self.forward_ensemble is not None:
                  output = self.forward_ensemble(lq, self.generator)
                else:
                  output = self.generator(lq)

        # If the GT is an image (i.e. the center frame), the output sequence is
        # turned to an image.
        if gt is not None and gt.ndim == 4:
            t = output.size(1)
            if self.check_if_mirror_extended(lq):  # with mirror extension
                output = 0.5 * (output[:, t // 4] + output[:, -1 - t // 4])
            else:  # without mirror extension
                output = output[:, t // 2]

        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()
      
       
        loss_pix = self.pixel_loss(output, gt)
        results['eval_result']['loss'] = loss_pix.item()

        # save image
        if save_image:
            if output.ndim == 4:  # an image, key = 000001/0000 (Vimeo-90K)
                img_name = meta[0]['key'].replace('/', '_')
                if isinstance(iteration, numbers.Number):
                    save_path = osp.join(
                        save_path, f'{img_name}-{iteration + 1:06d}.png')
                elif iteration is None:
                    save_path = osp.join(save_path, f'{img_name}.png')
                else:
                    raise ValueError('iteration should be number or None, '
                                     f'but got {type(iteration)}')
                mmcv.imwrite(tensor2img(output), save_path)
            elif output.ndim == 5:  # a sequence, key = 000
                folder_name = meta[0]['key'].split('/')[0]
                for i in range(0, output.size(1)):
                    if isinstance(iteration, numbers.Number):
                        save_path_i = osp.join(
                            save_path, folder_name,
                            f'{i:08d}-{iteration + 1:06d}.png')
                    elif iteration is None:
                        save_path_i = osp.join(save_path, folder_name,
                                               f'{i:08d}.png')
                    else:
                        raise ValueError('iteration should be number or None, '
                                         f'but got {type(iteration)}')
                    mmcv.imwrite(
                        tensor2img(output[:, i, :, :, :]), save_path_i)

        return results
