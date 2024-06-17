# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch
from mmcv import Config

from mmrotate.core import rbbox2result
from ..detectors import RotatedRetinaNet
from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector
from .utils import AlignConvModule
from ... import build_dataset
from ...core.bbox.iou_calculators import build_iou_calculator


@ROTATED_DETECTORS.register_module()
class RefinedDet(RotatedRetinaNet):
    """Implementation of RefinedDet.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RefinedDet, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                                               pretrained)
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
        self.align_conv = AlignConvModule(256, [8, 16, 32, 64, 128], 3)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    # # ======================================= Align_Conv =======================================
    # def forward_train(self,
    #                   img,
    #                   img_metas,
    #                   gt_bboxes,
    #                   gt_labels,
    #                   gt_bboxes_ignore=None):
    #     losses = dict()
    #     x = self.extract_feat(img)
    #
    #     outs = self.bbox_head(x)
    #     loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
    #     loss_base = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
    #     for name, value in loss_base.items():
    #         losses[f'base.{name}'] = value
    #
    #     rois = self.bbox_head.refine_bboxes(*outs)
    #     align_feat = self.align_conv(x, rois)
    #     outs = self.bbox_head(align_feat)
    #
    #     loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
    #     loss_refine = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, rois=rois)
    #     for name, value in loss_refine.items():
    #         losses[f'refine.{name}'] = value
    #
    #     return losses
    #
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
    #
    #     # align_conv
    #     rois = self.bbox_head.refine_bboxes(*outs)
    #     align_feat = self.align_conv(x, rois)
    #     outs = self.bbox_head(align_feat)
    #
    #     bbox_list = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale, rois=rois)
    #     bbox_results = [
    #         rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
    #         for det_bboxes, det_labels in bbox_list
    #     ]
    #     return bbox_results

    # ======================================= FRHead =======================================
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        losses = dict()
        x = self.extract_feat(img)

        coarse_cls_score, coarse_bbox_pred, refine_cls_score, refine_bbox_pred, cls_feat_sigmoid = self.bbox_head(x)
        loss_inputs = (coarse_cls_score, coarse_bbox_pred, gt_bboxes, gt_labels, img_metas)
        loss_base = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        for name, value in loss_base.items():
            losses[f'base.{name}'] = value

        # element-wise product/summation
        refine_features = tuple([i + i * j for i, j in zip(x, cls_feat_sigmoid)])
        rois = self.bbox_head.refine_bboxes(coarse_cls_score, coarse_bbox_pred)
        align_feat = self.align_conv(refine_features, rois)
        cls_score, bbox_pred, *_ = self.bbox_head(align_feat)

        # element-wise weighting
        cls_score = [i + i * j for i, j in zip(cls_score, refine_cls_score)]
        bbox_pred = [i + i * j for i, j in zip(bbox_pred, refine_bbox_pred)]

        loss_inputs = (cls_score, bbox_pred, gt_bboxes, gt_labels, img_metas)
        loss_refine = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, rois=rois)
        for name, value in loss_refine.items():
            losses[f'refine.{name}'] = value

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

        coarse_cls_score, coarse_bbox_pred, refine_cls_score, refine_bbox_pred, cls_feat_sigmoid = self.bbox_head(x)

        # element-wise product/summation
        refine_features = tuple([i + i * j for i, j in zip(x, cls_feat_sigmoid)])
        rois = self.bbox_head.refine_bboxes(coarse_cls_score, coarse_bbox_pred)
        align_feat = self.align_conv(refine_features, rois)
        cls_score, bbox_pred, *_ = self.bbox_head(align_feat)

        # element-wise weighting
        cls_score = [i + i * j for i, j in zip(cls_score, refine_cls_score)]
        bbox_pred = [i + i * j for i, j in zip(bbox_pred, refine_bbox_pred)]

        bbox_list = self.bbox_head.get_bboxes(cls_score, bbox_pred, img_metas, rescale=rescale, rois=rois)
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
    #
    #     coarse_cls_score, coarse_bbox_pred, refine_cls_score, refine_bbox_pred, cls_feat_sigmoid = self.bbox_head(x)
    #
    #     # element-wise product/summation
    #     refine_features = tuple([i + i * j for i, j in zip(x, cls_feat_sigmoid)])
    #     rois = self.bbox_head.refine_bboxes(coarse_cls_score, coarse_bbox_pred)
    #     align_feat = self.align_conv(refine_features, rois)
    #     cls_score, bbox_pred, *_ = self.bbox_head(align_feat)
    #
    #     # element-wise weighting
    #     cls_score = [i + i * j for i, j in zip(cls_score, refine_cls_score)]
    #     bbox_pred = [i + i * j for i, j in zip(bbox_pred, refine_bbox_pred)]
    #
    #     bbox_list = self.bbox_head.get_bboxes(cls_score, bbox_pred, img_metas, rescale=rescale, rois=rois)
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

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError