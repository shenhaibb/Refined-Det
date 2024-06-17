# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
from matplotlib import pyplot as plt
from mmcv.ops import points_in_polygons

from assigner_visualization import AssignerVisualizer

from ..builder import ROTATED_BBOX_ASSIGNERS
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.max_iou_assigner import MaxIoUAssigner

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


@ROTATED_BBOX_ASSIGNERS.register_module()
class CLA(MaxIoUAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
            `min_pos_iou` is set to avoid assigning bboxes that have extremely
            small iou with GT as positive samples. It brings about 0.3 mAP
            improvements in 1x schedule but does not affect the performance of
            3x schedule. More comparisons can be found in
            `PR #7464 <https://github.com/open-mmlab/mmdetection/pull/7464>`_.
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonstrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(self, **kwargs):
        super(CLA, self).__init__(**kwargs)
        self.near_neighbor_thr = 0.5
        self.prior_match_thr = 4.0
        self.grid_offset = torch.tensor([
            [0, 0],  # center
            [1, 0],  # left
            [0, 1],  # up
            [-1, 0],  # right
            [0, -1],  # bottom
        ], device='cuda').float()[:, None]

    def assign(self, anchors, responsible_flags, inside_flags, gt_bboxes, gt_labels=None, anchor_generator=None,
               img_meta=None):
        """Assign gt to bboxes. The process is very much like the max iou
        assigner, except that positive samples are constrained within the cell
        that the gt boxes fell in.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, 0, or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to -1
        2. assign proposals whose iou with all gts <= neg_iou_thr to 0
        3. for each bbox within a cell, if the iou with its nearest gt >
            pos_iou_thr and the center of that gt falls inside the cell,
            assign it to that bbox
        4. for each gt bbox, assign its nearest proposals within the cell the
            gt bbox falls in to itself.

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            responsible_flags (Tensor): flag to indicate whether box is
                responsible for prediction, shape(n, )
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        self.anchor_generator = anchor_generator

        bboxes, concat_responsible_flags = torch.cat(anchors), torch.cat(responsible_flags)

        num_gts, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all gt and bboxes
        overlaps = self.iou_calculator(gt_bboxes, bboxes)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,), -1, dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,), -1, dtype=torch.long)
            return AssignResult(num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # 2. assign negative: below
        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        # shape of max_overlaps == argmax_overlaps == num_bboxes
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)

        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps <= self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, (tuple, list)):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps > self.neg_iou_thr[0]) & (max_overlaps <= self.neg_iou_thr[1])] = 0

        # 3. assign positive: falls into responsible cell and above positive IOU threshold, the order matters.
        # the prior condition of comparison is to filter out all unrelated anchors, i.e. not concat_responsible_flags
        overlaps[:, ~concat_responsible_flags.type(torch.bool)] = -1.

        # calculate max_overlaps again, but this time we only consider IOUs for anchors responsible for prediction
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)

        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        # shape of gt_max_overlaps == gt_argmax_overlaps == num_gts
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        pos_inds = (max_overlaps > self.pos_iou_thr) & concat_responsible_flags.type(torch.bool)
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        # 4. assign positive to max overlapped anchors within responsible cell
        for i in range(num_gts):
            if gt_max_overlaps[i] > self.min_pos_iou:
                if self.gt_max_assign_all:
                    max_iou_inds = (overlaps[i, :] == gt_max_overlaps[i]) & concat_responsible_flags.type(torch.bool)
                    assigned_gt_inds[max_iou_inds] = i + 1
                elif concat_responsible_flags[gt_argmax_overlaps[i]]:
                    assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        # pos_inds = assigned_gt_inds > 0
        # draw_assign(anchors, argmax_overlaps, pos_inds, gt_bboxes, gt_labels, img_meta, anchor_generator, 'without_cla')

        # =============================== 跨网格标签分配策略 ===============================
        from mmrotate import obb2xyxy, obb2poly

        pos_inds = assigned_gt_inds > 0
        pos_inds_array = torch.split(pos_inds, [anchor.size(0) for anchor in anchors], 0)
        argmax_overlaps_array = torch.split(argmax_overlaps, [anchor.size(0) for anchor in anchors], 0)

        device = gt_bboxes.device
        input_h, input_w = img_meta.get('pad_shape')[:2]
        multi_level_flags = []
        for i, stride in enumerate(self.anchor_generator.base_sizes):
            feat_h, feat_w = self.anchor_generator.featmap_sizes[i]

            valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
            valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
            valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
            valid = valid_xx & valid_yy

            bboxes_xyxy = obb2xyxy(anchors[i][pos_inds_array[i]], 'le90')
            # 判断当前层是否有阳性样本
            if bboxes_xyxy.numel() != 0:
                gt_bboxes_temp = gt_bboxes[argmax_overlaps_array[i][pos_inds_array[i]].unique()]
                gt_bboxes_xyxy = obb2xyxy(gt_bboxes_temp, 'le90')
                gt_bboxes_poly = obb2poly(gt_bboxes_temp, 'le90')

                # 真值框归一化
                h, w = input_h // stride, input_w // stride
                bboxes_normed = convert_to_norm_format(gt_bboxes_xyxy, (input_h, input_w), anchor_generator)
                scaled_factor = torch.tensor([w, h, w, h, 1.], device=device)
                bboxes_scaled = bboxes_normed * scaled_factor

                # 获取邻域网格
                bboxes_cxcy = bboxes_scaled[:, :2]
                grid_xy = scaled_factor[:2] - bboxes_cxcy

                # =============================== 消融实验 ===============================
                # 设置邻域网格的个数k
                # # k=1
                # left, up = torch.zeros_like(bboxes_cxcy.T, dtype=torch.bool)
                # right, bottom = torch.zeros_like(grid_xy.T, dtype=torch.bool)
                # offset_inds = torch.stack((torch.ones_like(left), left, up, right, bottom))

                # # k=2
                # bboxes_grid_xy = torch.cat((bboxes_cxcy, grid_xy), dim=1)
                # left, up, right, bottom = ((bboxes_grid_xy % 1 < self.near_neighbor_thr) & (bboxes_grid_xy > 1) & ((bboxes_grid_xy % 1) == (bboxes_grid_xy % 1).min())).T
                # offset_inds = torch.stack((torch.ones_like(left), left, up, right, bottom))

                # k=3
                left, up = ((bboxes_cxcy % 1 < self.near_neighbor_thr) & (bboxes_cxcy > 1)).T
                right, bottom = ((grid_xy % 1 < self.near_neighbor_thr) & (grid_xy > 1)).T
                offset_inds = torch.stack((torch.ones_like(left), left, up, right, bottom))
                # =============================== 消融实验 ===============================

                bboxes_scaled = bboxes_scaled.repeat((5, 1, 1))[offset_inds]
                retained_offsets = self.grid_offset.repeat(1, offset_inds.shape[1], 1)[offset_inds]

                grid_xy, grid_wh, anchor_id = bboxes_scaled.chunk(3, 1)
                grid_xy = grid_xy - retained_offsets * self.near_neighbor_thr
                grid_xy_long = grid_xy.long()
                grid_x_inds, grid_y_inds = grid_xy_long.T

                # 选取中心点在真值框内的邻域网格
                x, y = grid_x_inds.float() * stride, grid_y_inds.float() * stride
                bboxes_points = torch.stack((x, y), dim=-1)
                is_in_bboxes = points_in_polygons(bboxes_points, gt_bboxes_poly).type(torch.bool)
                bboxes_scaled = bboxes_scaled[:, None].repeat(1, is_in_bboxes.shape[-1], 1)[is_in_bboxes]
                retained_offsets = retained_offsets[:, None].repeat(1, is_in_bboxes.size(-1), 1)[is_in_bboxes]

                grid_xy, grid_wh, anchor_id = bboxes_scaled.chunk(3, 1)
                anchor_id = anchor_id.long().view(-1)
                grid_xy = grid_xy - retained_offsets * self.near_neighbor_thr
                grid_xy_long = grid_xy.long()
                grid_x_inds, grid_y_inds = grid_xy_long.T

                flags = valid[:, None].repeat(1, self.anchor_generator.num_base_anchors[i])
                valids = torch.stack([grid_y_inds * valid_x.size(0) + grid_x_inds, anchor_id], dim=1)

                flags[valids[:, 0], valids[:, 1]] = True
                multi_level_flags.append(flags.contiguous().view(-1))
            else:
                flags = valid[:, None].expand(valid.size(0), self.anchor_generator.num_base_anchors[i])
                flags = flags.contiguous().view(-1)
                multi_level_flags.append(flags)

        valid_flags = []
        for i, _ in enumerate(inside_flags):
            valid_flags.append(multi_level_flags[i][inside_flags[i]])

        concat_valid_flags = torch.cat(valid_flags)
        concat_valid_flags = concat_valid_flags & concat_responsible_flags.type(torch.bool)
        assigned_gt_inds[concat_valid_flags] = argmax_overlaps[concat_valid_flags] + 1
        # =============================== 跨网格标签分配策略 ===============================

        # pos_inds = assigned_gt_inds > 0
        # draw_assign(anchors, argmax_overlaps, pos_inds, gt_bboxes, gt_labels, img_meta, anchor_generator, 'with_cla')

        # assign labels of positive anchors
        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def _meshgrid(self, x, y, row_major=True):
        """Generate mesh grid of x and y.

        Args:
            x (torch.Tensor): Grids of x dimension.
            y (torch.Tensor): Grids of y dimension.
            row_major (bool, optional): Whether to return y grids first.
                Defaults to True.

        Returns:
            tuple[torch.Tensor]: The mesh grids of x and y.
        """
        # use shape instead of len to keep tracing while exporting to onnx
        xx = x.repeat(y.shape[0])
        yy = y.view(-1, 1).repeat(1, x.shape[0]).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx


def convert_to_norm_format(gt_bboxes_xyxy, input_shape, anchor_generator):
    xy1, xy2 = gt_bboxes_xyxy.split((2, 2), dim=-1)
    gt_bboxes_xywh = torch.cat([(xy2 + xy1) / 2, (xy2 - xy1)], dim=-1)
    gt_bboxes_xywh[:, 1::2] /= input_shape[0]
    gt_bboxes_xywh[:, 0::2] /= input_shape[1]

    # (num_base_priors, num_bboxes, 4)
    bboxes_normed = gt_bboxes_xywh

    # (num_base_priors, num_bboxes, 1)
    bboxes_anchor_inds = torch.arange(anchor_generator.num_base_anchors[0]).float().view(
        anchor_generator.num_base_anchors[0], 1).cuda()
    bboxes_anchor_inds = bboxes_anchor_inds.repeat(bboxes_normed.shape[0], 1)

    # (num_base_priors, num_bboxes, 5)
    # (bbox_cx, bbox_cy, bbox_w, bbox_h, anchor_ind)
    bboxes_normed = torch.cat((bboxes_normed, bboxes_anchor_inds), 1)

    return bboxes_normed

# ======================================= Visualization problems1 =======================================
def draw_assign(anchors, argmax_overlaps, pos_inds, gt_bboxes, gt_labels, img_meta, anchor_generator, is_cla):
    from mmrotate import obb2xyxy

    filename = img_meta.get('filename')
    img = mmcv.imread(filename)
    img = mmcv.bgr2rgb(img)

    pos_inds_array = torch.split(pos_inds, [anchor.size(0) for anchor in anchors], 0)
    argmax_overlaps_array = torch.split(argmax_overlaps, [anchor.size(0) for anchor in anchors], 0)

    input_h, input_w = img_meta.get('img_shape')[:2]
    for i, stride in enumerate(anchor_generator.base_sizes):
        feat_h, feat_w = anchor_generator.featmap_sizes[i]

        bboxes_xyxy = obb2xyxy(anchors[i][pos_inds_array[i]], 'le90')
        if len(bboxes_xyxy) > 0:
            gt_bboxes_temp = gt_bboxes[argmax_overlaps_array[i][pos_inds[i]].unique()]
            gt_bboxes_xyxy = obb2xyxy(gt_bboxes, 'le90')

            # 获取目标真实框
            h, w = input_h / stride, input_w / stride
            gt_bboxes_normed = convert_to_norm_format(gt_bboxes_xyxy, (input_h, input_w), anchor_generator)
            gt_scaled_factor = torch.tensor([w, h, w, h, 1.]).cuda()
            gt_bboxes_scaled = gt_bboxes_normed * gt_scaled_factor

            gt_bboxes_cxcy = gt_bboxes_scaled[:, :2]
            gt_bboxes_x_inds, gt_bboxes_y_inds = gt_bboxes_cxcy.T

            # 获取分配的阳性样本
            bboxes_normed = convert_to_norm_format(bboxes_xyxy, (input_h, input_w), anchor_generator)
            scaled_factor = torch.tensor([feat_w, feat_h, feat_w, feat_h, 1.]).cuda()
            bboxes_scaled = bboxes_normed * scaled_factor

            bboxes_cxcy = bboxes_scaled[:, :2]
            grid_xy = bboxes_cxcy
            grid_xy_long = grid_xy.long()
            grid_x_inds, grid_y_inds = grid_xy_long.T

            # =============================== 可视化左图 ===============================
            visualizer = AssignerVisualizer(save_dir='../work_dirs/vis/test')
            visualizer.set_image(img)
            visualizer.draw_grid(stride)

            visualizer.draw_instances_obb_assign((gt_bboxes_temp, gt_labels), '#00FF00', '-')
            visualizer.draw_instances_center(gt_bboxes_x_inds, gt_bboxes_y_inds, stride, '#00FF00')
            visualizer.draw_positive_assign(grid_x_inds, grid_y_inds, stride, '#FF0000')

            img_show = visualizer.get_image()
            plt.imshow(img_show)
            plt.show()
            mmcv.imwrite(mmcv.rgb2bgr(img_show), f'../work_dirs/vis/{is_cla}/left_{stride}_{img_meta.get("ori_filename")}')

            # =============================== 可视化右图 ===============================
            visualizer = AssignerVisualizer(save_dir='../work_dirs/vis/test')
            visualizer.set_image(img)

            visualizer.draw_instances_obb_assign((gt_bboxes, gt_labels), '#00FF00', '-')
            visualizer.draw_instances_xyxy_assign((bboxes_xyxy, gt_labels), '#FF0000', '--')
            visualizer.draw_positive_assign(grid_x_inds, grid_y_inds, stride, '#FF0000')

            img_show = visualizer.get_image()
            plt.imshow(img_show)
            plt.show()
            mmcv.imwrite(mmcv.rgb2bgr(img_show), f'../work_dirs/vis/{is_cla}/right_{stride}_{img_meta.get("ori_filename")}')

    return None
