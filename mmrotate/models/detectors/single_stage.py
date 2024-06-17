# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch
from PIL import Image
from mmcv import Config

from mmrotate.core import rbbox2result
from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector
from matplotlib import pyplot as plt

from ... import build_dataset
from ...core.bbox.iou_calculators import build_iou_calculator


@ROTATED_DETECTORS.register_module()
class RotatedSingleStageDetector(RotatedBaseDetector):
    """Base class for rotated single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RotatedSingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(RotatedSingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes. \
                The outer list corresponds to each image. The inner list \
                corresponds to each class.
        """

        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    # # ======================================= Visualization problems2 =======================================
    # def simple_test(self, img, img_metas, rescale=False):
    #     """Test function without test time augmentation.
    #
    #     Args:
    #         imgs (list[torch.Tensor]): List of multiple images
    #         img_metas (list[dict]): List of image information.
    #         rescale (bool, optional): Whether to rescale the results.
    #             Defaults to False.
    #
    #     Returns:
    #         list[list[np.ndarray]]: BBox results of each image and classes. \
    #             The outer list corresponds to each image. The inner list \
    #             corresponds to each class.
    #     """
    #     x = self.extract_feat(img)
    #     outs = self.bbox_head(x)
    #     bbox_list = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
    #
    #     # =================== 获取真实框 ===================
    #     cfg = Config.fromfile('../configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_6x_hrsc_rr_le90.py')
    #     dataset = build_dataset(cfg.data.test)
    #     iou_calculator = build_iou_calculator(dict(type='RBboxOverlaps2D'))
    #     data_info = [i for i in dataset.data_infos if i.get('filename') == img_metas[0]['ori_filename']]
    #     gt_bboxes = data_info[0]['ann'].get('bboxes')
    #
    #     bbox_results = [[]]
    #     for det_bboxes, det_labels in bbox_list:
    #         bboxes = rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
    #         ious = [iou_calculator(torch.from_numpy(gt_bboxes), torch.from_numpy(bbox)) for bbox in bboxes]
    #         for bbox, iou in zip(bboxes, ious):
    #             if iou.size(1) == 0:
    #                 iou = torch.zeros(len(bbox), 1)
    #             else:
    #                 iou = torch.full((len(bbox), 1), iou.max(1)[0].item())
    #             bbox = np.concatenate((bbox, iou.numpy()), axis=1)
    #             bbox_results[0].append(bbox)
    #     return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes. \
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results