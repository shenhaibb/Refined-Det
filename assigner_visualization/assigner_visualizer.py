# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Union

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from mmcv import Config
from mmengine.visualization import Visualizer
from mmengine.visualization.utils import tensor2ndarray
from torch import Tensor


class AssignerVisualizer(Visualizer):
    """MMYOLO Detection Assigner Visualizer.

    This class is provided to the `assigner_visualization.py` script.
    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
    """

    def __init__(self, name: str = 'visualizer', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        # need priors_size from config
        from mmrotate.datasets import build_dataset
        cfg = Config.fromfile('../configs/refined_det/hrsc/refined_det_cla_obb_r50_fpn_6x_hrsc_rr_le90.py')
        self.dataset_meta = build_dataset(cfg.data.train)
        self.priors_size = None
        self.text_color = (200, 200, 200)
        self.alpha = 0.8
        self.line_width = 2

    def draw_grid(self,
                  stride: int = 8,
                  line_styles: Union[str, List[str]] = ':',
                  colors: Union[str, tuple, List[str],
                                List[tuple]] = (200, 200, 200),
                  line_widths: Union[Union[int, float],
                                     List[Union[int, float]]] = 2):
        """Draw grids on image.

        Args:
            stride (int): Downsample factor of feature map.
            line_styles (Union[str, List[str]]): The linestyle
                of lines. ``line_styles`` can have the same length with
                texts or just single value. If ``line_styles`` is single
                value, all the lines will have the same linestyle.
                Reference to
                https://matplotlib.org/stable/api/collections_api.html?highlight=collection#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
                for more details. Defaults to ':'.
            colors (Union[str, tuple, List[str], List[tuple]]): The colors of
                lines. ``colors`` can have the same length with lines or just
                single value. If ``colors`` is single value, all the lines
                will have the same colors. Reference to
                https://matplotlib.org/stable/gallery/color/named_colors.html
                for more details. Defaults to (180, 180, 180).
            line_widths (Union[Union[int, float], List[Union[int, float]]]):
                The linewidth of lines. ``line_widths`` can have
                the same length with lines or just single value.
                If ``line_widths`` is single value, all the lines will
                have the same linewidth. Defaults to 1.
        """
        # assert self._image is not None, 'Please set image using `set_image`'
        # # draw vertical lines
        # x_datas_vertical = ((np.arange(self.width // stride - 1) + 1) * stride).reshape((-1, 1)).repeat(2, axis=1)
        # y_datas_vertical = np.array([[0, self.height - 1]]).repeat(self.width // stride - 1, axis=0)
        # self.draw_lines(
        #     x_datas_vertical,
        #     y_datas_vertical,
        #     colors=colors,
        #     line_styles=line_styles,
        #     line_widths=line_widths)
        #
        # # draw horizontal lines
        # x_datas_horizontal = np.array([[0, self.width - 1]]).repeat(self.height // stride - 1, axis=0)
        # y_datas_horizontal = ((np.arange(self.height // stride - 1) + 1) *
        #                       stride).reshape((-1, 1)).repeat(2, axis=1)
        # self.draw_lines(
        #     x_datas_horizontal,
        #     y_datas_horizontal,
        #     colors=colors,
        #     line_styles=line_styles,
        #     line_widths=line_widths)

        assert self._image is not None, 'Please set image using `set_image`'
        # draw vertical lines
        x_datas_vertical = ((np.arange(self.width // stride) + 1) * stride).reshape((-1, 1)).repeat(2, axis=1)
        y_datas_vertical = np.array([[0, self.height - 1]]).repeat(self.width // stride, axis=0)
        self.draw_lines(
            x_datas_vertical,
            y_datas_vertical,
            colors=colors,
            line_styles=line_styles,
            line_widths=line_widths)

        # draw horizontal lines
        x_datas_horizontal = np.array([[0, self.width - 1]]).repeat(self.height // stride, axis=0)
        y_datas_horizontal = ((np.arange(self.height // stride) + 1) *
                              stride).reshape((-1, 1)).repeat(2, axis=1)
        self.draw_lines(
            x_datas_horizontal,
            y_datas_horizontal,
            colors=colors,
            line_styles=line_styles,
            line_widths=line_widths)

    def draw_instances_obb_assign(self, instances, edge_colors='#00FF00', line_styles='-'):
        """Draw instances of GT.

        Args:
            instances (:obj:`InstanceData`): gt_instance. It usually
             includes ``bboxes`` and ``labels`` attributes.
            retained_gt_inds (Tensor): The gt indexes assigned as the
                positive sample in the current prior.
        """
        assert self.dataset_meta is not None
        classes = self.dataset_meta.CLASSES
        palette = self.dataset_meta.PALETTE
        # if len(retained_gt_inds) == 0:
        #     return self.get_image()
        # draw_gt_inds = torch.from_numpy(np.array(list(set(retained_gt_inds.cpu().numpy())), dtype=np.int64))
        # bboxes = instances.bboxes[draw_gt_inds]
        # labels = instances.labels[draw_gt_inds]
        bboxes, labels = instances

        if not isinstance(bboxes, Tensor):
            bboxes = bboxes.tensor

        # self.draw_bboxes(
        #     bboxes,
        #     edge_colors=edge_colors,
        #     alpha=self.alpha,
        #     line_widths=self.line_width)

        self.draw_rbboxes(
            bboxes,
            edge_colors=edge_colors,
            line_styles=line_styles,
            alpha=self.alpha,
            line_widths=self.line_width)

        # if labels is not None:
        #     max_label = int(max(labels) if len(labels) > 0 else 0)
        #     text_palette = get_palette(self.text_color, max_label + 1)
        #     text_colors = [text_palette[label] for label in labels]
        #
        #     positions = bboxes[:, :2] + self.line_width
        #     areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        #     scales = _get_adaptive_scales(areas.cpu())
        #     for i, (pos, label) in enumerate(zip(positions, labels)):
        #         label_text = classes[label] if classes is not None else f'class {label}'
        #
        #         self.draw_texts(
        #             label_text,
        #             pos,
        #             colors=text_colors[i],
        #             font_sizes=int(13 * scales[i]),
        #             bboxes=[{
        #                 'facecolor': 'black',
        #                 'alpha': 0.8,
        #                 'pad': 0.7,
        #                 'edgecolor': 'none'
        #             }])

    def draw_instances_xyxy_assign(self, instances, edge_colors='#00FF00', line_styles='-'):
        """Draw instances of GT.

        Args:
            instances (:obj:`InstanceData`): gt_instance. It usually
             includes ``bboxes`` and ``labels`` attributes.
            retained_gt_inds (Tensor): The gt indexes assigned as the
                positive sample in the current prior.
        """
        assert self.dataset_meta is not None
        classes = self.dataset_meta.CLASSES
        palette = self.dataset_meta.PALETTE
        # if len(retained_gt_inds) == 0:
        #     return self.get_image()
        # draw_gt_inds = torch.from_numpy(np.array(list(set(retained_gt_inds.cpu().numpy())), dtype=np.int64))
        # bboxes = instances.bboxes[draw_gt_inds]
        # labels = instances.labels[draw_gt_inds]
        bboxes, labels = instances

        if not isinstance(bboxes, Tensor):
            bboxes = bboxes.tensor

        self.draw_bboxes(
            bboxes,
            edge_colors=edge_colors,
            line_styles=line_styles,
            alpha=self.alpha,
            line_widths=self.line_width)

        # if labels is not None:
        #     max_label = int(max(labels) if len(labels) > 0 else 0)
        #     text_palette = get_palette(self.text_color, max_label + 1)
        #     text_colors = [text_palette[label] for label in labels]
        #
        #     positions = bboxes[:, :2] + self.line_width
        #     areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        #     scales = _get_adaptive_scales(areas.cpu())
        #     for i, (pos, label) in enumerate(zip(positions, labels)):
        #         label_text = classes[label] if classes is not None else f'class {label}'
        #
        #         self.draw_texts(
        #             label_text,
        #             pos,
        #             colors=text_colors[i],
        #             font_sizes=int(13 * scales[i]),
        #             bboxes=[{
        #                 'facecolor': 'black',
        #                 'alpha': 0.8,
        #                 'pad': 0.7,
        #                 'edgecolor': 'none'
        #             }])

    def draw_instances_center(self,
                             grid_x_inds: Tensor,
                             grid_y_inds: Tensor,
                             stride: int,
                             colors='#00FF00',
                             offset: float = 0.5):
        """

        Args:
            grid_x_inds (Tensor): The X-axis indexes of the positive sample
                in current prior.
            grid_y_inds (Tensor): The Y-axis indexes of the positive sample
                in current prior.
            class_inds (Tensor): The classes indexes of the positive sample
                in current prior.
            stride (int): Downsample factor of feature map.
            bboxes (Union[Tensor, HorizontalBoxes]): Bounding boxes of GT.
            retained_gt_inds (Tensor): The gt indexes assigned as the
                positive sample in the current prior.
            offset (float): The offset of points, the value is normalized
                with corresponding stride. Defaults to 0.5.
        """
        # The palette in the dataset_meta is required
        assert self.dataset_meta is not None
        palette = self.dataset_meta.PALETTE
        x = (grid_x_inds * stride)
        y = (grid_y_inds * stride)
        center = torch.stack((x, y), dim=-1)

        # retained_bboxes = bboxes[retained_gt_inds]
        # bbox_wh = retained_bboxes[:, 2:] - retained_bboxes[:, :2]
        # bbox_area = bbox_wh[:, 0] * bbox_wh[:, 1]
        # radius = _get_adaptive_scales(bbox_area) * 4
        radius = torch.full((len(center),), 8.)
        # radius = torch.full((len(center),), 5.)
        # colors = [palette[i] for i in class_inds]

        self.draw_circles(
            center,
            radius,
            colors,
            line_widths=0,
            face_colors=colors,
            alpha=1.0)

    def draw_positive_assign(self,
                             grid_x_inds: Tensor,
                             grid_y_inds: Tensor,
                             stride: int,
                             colors='#00FF00',
                             offset: float = 0.):
        """

        Args:
            grid_x_inds (Tensor): The X-axis indexes of the positive sample
                in current prior.
            grid_y_inds (Tensor): The Y-axis indexes of the positive sample
                in current prior.
            class_inds (Tensor): The classes indexes of the positive sample
                in current prior.
            stride (int): Downsample factor of feature map.
            bboxes (Union[Tensor, HorizontalBoxes]): Bounding boxes of GT.
            retained_gt_inds (Tensor): The gt indexes assigned as the
                positive sample in the current prior.
            offset (float): The offset of points, the value is normalized
                with corresponding stride. Defaults to 0.5.
        """
        # The palette in the dataset_meta is required
        assert self.dataset_meta is not None
        palette = self.dataset_meta.PALETTE
        x = ((grid_x_inds + offset) * stride).long()
        y = ((grid_y_inds + offset) * stride).long()
        center = torch.stack((x, y), dim=-1)

        # retained_bboxes = bboxes[retained_gt_inds]
        # bbox_wh = retained_bboxes[:, 2:] - retained_bboxes[:, :2]
        # bbox_area = bbox_wh[:, 0] * bbox_wh[:, 1]
        # radius = _get_adaptive_scales(bbox_area) * 4
        radius = torch.full((len(center),), 8.)
        # radius = torch.full((len(center),), 5.)
        # colors = [palette[i] for i in class_inds]

        self.draw_circles(
            center,
            radius,
            colors,
            line_widths=0,
            face_colors=colors,
            alpha=1.0)

    def draw_prior(self,
                   grid_x_inds: Tensor,
                   grid_y_inds: Tensor,
                   class_inds: Tensor,
                   stride: int,
                   feat_ind: int,
                   prior_ind: int,
                   offset: float = 0.5):
        """Draw priors on image.

        Args:
            grid_x_inds (Tensor): The X-axis indexes of the positive sample
                in current prior.
            grid_y_inds (Tensor): The Y-axis indexes of the positive sample
                in current prior.
            class_inds (Tensor): The classes indexes of the positive sample
                in current prior.
            stride (int): Downsample factor of feature map.
            feat_ind (int): Index of featmap.
            prior_ind (int): Index of prior in current featmap.
            offset (float): The offset of points, the value is normalized
                with corresponding stride. Defaults to 0.5.
        """

        palette = self.dataset_meta['palette']
        center_x = ((grid_x_inds + offset) * stride)
        center_y = ((grid_y_inds + offset) * stride)
        xyxy = torch.stack((center_x, center_y, center_x, center_y), dim=1)
        assert self.priors_size is not None
        xyxy += self.priors_size[feat_ind][prior_ind]

        colors = [palette[i] for i in class_inds]
        self.draw_bboxes(
            xyxy,
            edge_colors=colors,
            alpha=self.alpha,
            line_styles='--',
            line_widths=math.ceil(self.line_width * 0.3))

    def draw_assign(self,
                    image: np.ndarray,
                    assign_results: List[List[dict]],
                    gt_instances,
                    img_meta,
                    show_prior: bool = False,
                    not_show_label: bool = False) -> np.ndarray:
        """Draw assigning results.

        Args:
            image (np.ndarray): The image to draw.
            assign_results (list): The assigning results.
            gt_instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            show_prior (bool): Whether to show prior on image.
            not_show_label (bool): Whether to show gt labels on images.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        img_show_list = []
        for feat_ind, assign_results_feat in enumerate(assign_results):
            img_show_list_feat = []
            for prior_ind, assign_results_prior in enumerate(assign_results_feat):
                self.set_image(image)
                h, w = image.shape[:2]

                # draw grid
                stride = assign_results_prior['stride']
                self.draw_grid(stride)

                # draw prior on matched gt
                grid_x_inds = assign_results_prior['grid_x_inds']
                grid_y_inds = assign_results_prior['grid_y_inds']
                class_inds = assign_results_prior['class_inds']
                # prior_ind = assign_results_prior['prior_ind']
                # if show_prior:
                #     self.draw_prior(grid_x_inds, grid_y_inds, class_inds,
                #                     stride, feat_ind, prior_ind)

                # draw matched gt
                retained_gt_inds = assign_results_prior['retained_gt_inds']
                self.draw_instances_assign(gt_instances, retained_gt_inds, not_show_label)

                # draw positive
                self.draw_positive_assign(grid_x_inds, grid_y_inds, class_inds,
                                          stride, gt_instances[0], retained_gt_inds)

                # # draw title
                # base_prior = self.priors_size[feat_ind][prior_ind]
                # prior_size = (base_prior[2] - base_prior[0],
                #               base_prior[3] - base_prior[1])
                # pos = np.array((20, 20))
                # text = f'feat_ind: {feat_ind}  ' \
                #        f'prior_ind: {prior_ind} ' \
                #        f'prior_size: ({prior_size[0]}, {prior_size[1]})'
                # scales = _get_adaptive_scales(np.array([h * w / 16]))
                # font_sizes = int(13 * scales)
                # self.draw_texts(
                #     text,
                #     pos,
                #     colors=self.text_color,
                #     font_sizes=font_sizes,
                #     bboxes=[{
                #         'facecolor': 'black',
                #         'alpha': 0.8,
                #         'pad': 0.7,
                #         'edgecolor': 'none'
                #     }])

                img_show = self.get_image()
                img_show = mmcv.impad(img_show, padding=(5, 5, 5, 5))
                img_show_list_feat.append(img_show)
                plt.imshow(img_show)
                plt.show()

                # filename = img_meta.get('ori_filename').replace('.bmp', '.jpg')
                # out_file = os.path.join('../assigned_results', filename)
                # mmcv.imwrite(mmcv.rgb2bgr(img_show), out_file)
                # print("1")
            img_show_list.append(np.concatenate(img_show_list_feat, axis=1))

        # Merge all images into one image
        return np.concatenate(img_show_list, axis=0)

    def draw_rbboxes(self,
                     bboxes: Union[np.ndarray, torch.Tensor],
                     edge_colors: Union[str, tuple, List[str], List[tuple]] = 'g',
                     line_styles: Union[str, List[str]] = '-',
                     line_widths: Union[Union[int, float], List[Union[int, float]]] = 2,
                     face_colors: Union[str, tuple, List[str], List[tuple]] = 'none',
                     alpha: Union[int, float] = 0.8,
                     ) -> 'Visualizer':
        from matplotlib.collections import PolyCollection
        bboxes = [tensor2ndarray(bbox) for bbox in bboxes]

        polygons = []
        for i, bbox in enumerate(bboxes):
            xc, yc, w, h, ag = bbox[:5]
            wx, wy = w / 2 * np.cos(ag), w / 2 * np.sin(ag)
            hx, hy = -h / 2 * np.sin(ag), h / 2 * np.cos(ag)
            p1 = (xc - wx - hx, yc - wy - hy)
            p2 = (xc + wx - hx, yc + wy - hy)
            p3 = (xc + wx + hx, yc + wy + hy)
            p4 = (xc - wx + hx, yc - wy + hy)
            poly = np.int0(np.array([p1, p2, p3, p4]))
            polygons.append(Polygon(poly))
        polygon_collection = PatchCollection(
            polygons,
            alpha=alpha,
            facecolor=face_colors,
            linestyles=line_styles,
            edgecolors=edge_colors,
            linewidths=line_widths)

        self.ax_save.add_collection(polygon_collection)
        return self
