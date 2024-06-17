# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmdet.core import images_to_levels, multi_apply, unmap
from .rotated_retina_head import RotatedRetinaHead
from mmrotate.core import (obb2hbb, rotated_anchor_inside_flags)
from ..builder import ROTATED_HEADS


@ROTATED_HEADS.register_module()
class FRHead(RotatedRetinaHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(FRHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, 3, padding=1)
        self.retina_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 5, 3, padding=1)

        # # ======================================= Align_Conv =======================================
        # self.sigmoid_conv = nn.Conv2d(self.feat_channels, 1, 1)

        # ======================================= FRHead =======================================
        self.conv_3_1 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.conv_1_3 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.cls_score_refine_conv = nn.Conv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, 1)
        self.bbox_pred_refine_conv = nn.Conv2d(self.feat_channels, self.num_anchors * 5, 1)
        self.sigmoid_conv = nn.Conv2d(self.num_anchors * self.cls_out_channels, 1, 1)

    # # ======================================= Align_Conv =======================================
    # def forward_single(self, x):
    #     """Forward feature of a single scale level.
    #
    #     Args:
    #         x (torch.Tensor): Features of a single scale level.
    #
    #     Returns:
    #         tuple (torch.Tensor):
    #
    #             - cls_score (torch.Tensor): Cls scores for a single scale level \
    #                 the channels number is num_anchors * num_classes.
    #             - bbox_pred (torch.Tensor): Box energies / deltas for a \
    #                 single scale level, the channels number is num_anchors * 5.
    #     """
    #     cls_feat = x
    #     reg_feat = x
    #     for cls_conv in self.cls_convs:
    #         cls_feat = cls_conv(cls_feat)
    #     for reg_conv in self.reg_convs:
    #         reg_feat = reg_conv(reg_feat)
    #     cls_score = self.retina_cls(cls_feat)
    #     bbox_pred = self.retina_reg(reg_feat)
    #
    #     return cls_score, bbox_pred

    # ======================================= FRHead =======================================
    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (torch.Tensor): Features of a single scale level.

        Returns:
            tuple (torch.Tensor):

                - cls_score (torch.Tensor): Cls scores for a single scale level \
                    the channels number is num_anchors * num_classes.
                - bbox_pred (torch.Tensor): Box energies / deltas for a \
                    single scale level, the channels number is num_anchors * 5.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        coarse_cls_score = self.retina_cls(cls_feat)
        coarse_bbox_pred = self.retina_reg(reg_feat)

        cls_feat = x
        reg_feat = x
        cls_feat = self.conv_3_1(self.conv_1_3(cls_feat))
        reg_feat = self.conv_3_1(self.conv_1_3(reg_feat))
        refine_cls_score = self.cls_score_refine_conv(cls_feat)
        refine_bbox_pred = self.bbox_pred_refine_conv(reg_feat)

        cls_feat_sigmoid = self.sigmoid_conv(coarse_cls_score).sigmoid()

        return coarse_cls_score, coarse_bbox_pred, refine_cls_score, refine_bbox_pred, cls_feat_sigmoid

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            rois (list[list[Tensor]]) bboxes of levels of images.
                before further regression just like anchors.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple (list[Tensor]):

                - anchor_list (list[Tensor]): Anchors of each image.
                - valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        if self.rois is None:
            num_imgs = len(img_metas)
            # since feature map sizes of all images are the same, we only compute anchors for one time
            multi_level_anchors = self.anchor_generator.grid_priors(featmap_sizes, device)
            anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        else:
            anchor_list = []
            for rois in self.rois:
                anchors = []
                for multi_level_anchors in rois:
                    anchors.append(multi_level_anchors.clone().detach())
                anchor_list.append(anchors)

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.anchor_generator.valid_flags(featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None,
             rois=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        self.rois = rois
        return super(FRHead, self).loss(
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            img_metas=img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   rois=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rois (list[list[Tensor]]): input rbboxes of each level of
                each image. rois output by former stages and are to be refined
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 6) tensor, where the first 5 columns
                are bounding box positions (cx, cy, w, h, a) and the
                6-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_priors(featmap_sizes, device=device)

        result_list = []
        for img_id, _ in enumerate(img_metas):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if rois is None:
                proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list, mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale)
            else:
                proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list, rois[img_id], img_shape,
                                                    scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list