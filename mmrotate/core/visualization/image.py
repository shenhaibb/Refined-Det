# Copyright (c) OpenMMLab. All rights reserved.
import sys

import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from mmdet.core.visualization import palette_val
from mmdet.core.visualization.image import draw_labels, draw_masks

from mmrotate.core.visualization.palette import get_palette

EPS = 1e-2


def _get_adaptive_scales(areas, min_area=800, max_area=30000):
    """Get adaptive scales according to areas.

    The scale range is [0.5, 1.0]. When the area is less than
    ``'min_area'``, the scale is 0.5 while the area is larger than
    ``'max_area'``, the scale is 1.0.

    Args:
        areas (ndarray): The areas of bboxes or masks with the
            shape of (n, ).
        min_area (int): Lower bound areas for adaptive scales.
            Default: 800.
        max_area (int): Upper bound areas for adaptive scales.
            Default: 30000.

    Returns:
        ndarray: The adaotive scales with the shape of (n, ).
    """
    scales = 0.5 + (areas - min_area) / (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales


def draw_rbboxes(ax, bboxes, color='g', alpha=0.8, thickness=2):
    """Draw oriented bounding boxes on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        bboxes (ndarray): The input bounding boxes with the shape
            of (n, 5).
        color (list[tuple] | matplotlib.color): the colors for each
            bounding boxes.
        alpha (float): Transparency of bounding boxes. Default: 0.8.
        thickness (int): Thickness of lines. Default: 2.

    Returns:
        matplotlib.Axes: The result axes.
    """
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
    p = PatchCollection(
        polygons,
        facecolor='none',
        edgecolors=color,
        linewidths=thickness,
        alpha=alpha)
    ax.add_collection(p)

    return ax


def imshow_det_rbboxes(img,
                       bboxes=None,
                       labels=None,
                       segms=None,
                       class_names=None,
                       score_thr=0,
                       bbox_color='green',
                       text_color='green',
                       mask_color=None,
                       thickness=2,
                       font_size=13,
                       win_name='',
                       show=True,
                       wait_time=0,
                       out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str | ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 5) or
            (n, 6).
        labels (ndarray): Labels of bboxes.
        segms (ndarray | None): Masks, shaped (n,h,w) or None.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown. Default: 0.
        bbox_color (list[tuple] | tuple | str | None): Colors of bbox lines.
           If a single color is given, it will be applied to all classes.
           The tuple of color should be in RGB order. Default: 'green'.
        text_color (list[tuple] | tuple | str | None): Colors of texts.
           If a single color is given, it will be applied to all classes.
           The tuple of color should be in RGB order. Default: 'green'.
        mask_color (list[tuple] | tuple | str | None, optional): Colors of
           masks. If a single color is given, it will be applied to all
           classes. The tuple of color should be in RGB order.
           Default: None.
        thickness (int): Thickness of lines. Default: 2.
        font_size (int): Font size of texts. Default: 13.
        show (bool): Whether to show the image. Default: True.
        win_name (str): The window name. Default: ''.
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes is None or bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes is None or bboxes.shape[1] == 5 or bboxes.shape[1] == 6 or bboxes.shape[1] == 7, \
        f' bboxes.shape[1] should be 5 or 6 or 7, but its {bboxes.shape[1]}.'
    assert bboxes is None or bboxes.shape[0] <= labels.shape[0], \
        'labels.shape[0] should not be less than bboxes.shape[0].'
    assert segms is None or segms.shape[0] == labels.shape[0], \
        'segms.shape[0] and labels.shape[0] should have the same length.'
    assert segms is not None or bboxes is not None, \
        'segms and bboxes should not be None at the same time.'

    img = mmcv.imread(img).astype(np.uint8)

    # bboxes, ious = bboxes[:, :-1], bboxes[:, -1]
    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 6
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    max_label = int(max(labels) if len(labels) > 0 else 0)
    text_palette = palette_val(get_palette(text_color, max_label + 1))
    text_colors = [text_palette[label] for label in labels]

    num_bboxes = 0
    if bboxes is not None:
        num_bboxes = bboxes.shape[0]
        # bbox_color = [(0, 255, 0) for _ in range(15)]
        bbox_palette = palette_val(get_palette(bbox_color, max_label + 1))
        colors = [bbox_palette[label] for label in labels[:num_bboxes]]
        draw_rbboxes(ax, bboxes, colors, alpha=0.8, thickness=5)

        # for i in range(bboxes.shape[0]):
        #     plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        #     ax = plt.gca()
        #     ax.axis('off')
        #     draw_rbboxes(ax, bboxes[i:i + 1, :], colors, alpha=0.8, thickness=thickness)
        #     plt.imshow(img)
        #     plt.show()
        #     print("1")

        horizontal_alignment = 'left'
        positions = bboxes[:, :2].astype(np.int32) + thickness
        areas = bboxes[:, 2] * bboxes[:, 3]
        scales = _get_adaptive_scales(areas)
        scores = bboxes[:, 5] if bboxes.shape[1] == 6 else None
        draw_labels(
            ax,
            labels[:num_bboxes],
            positions,
            scores=scores,
            ious=None,
            class_names=class_names,
            color=text_colors,
            font_size=font_size,
            scales=scales,
            horizontal_alignment=horizontal_alignment)
        plt.imshow(img)
        plt.show()

    if segms is not None:
        mask_palette = get_palette(mask_color, max_label + 1)
        colors = [mask_palette[label] for label in labels]
        colors = np.array(colors, dtype=np.uint8)
        draw_masks(ax, img, segms, colors, with_edge=True)

        if num_bboxes < segms.shape[0]:
            segms = segms[num_bboxes:]
            horizontal_alignment = 'center'
            areas = []
            positions = []
            for mask in segms:
                _, _, stats, centroids = cv2.connectedComponentsWithStats(
                    mask.astype(np.uint8), connectivity=8)
                largest_id = np.argmax(stats[1:, -1]) + 1
                positions.append(centroids[largest_id])
                areas.append(stats[largest_id, -1])
            areas = np.stack(areas, axis=0)
            scales = _get_adaptive_scales(areas)
            draw_labels(
                ax,
                labels[num_bboxes:],
                positions,
                class_names=class_names,
                color=text_colors,
                font_size=font_size,
                scales=scales,
                horizontal_alignment=horizontal_alignment)

    plt.imshow(img)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    plt.close()

    return img


# def imshow_det_rbboxes(img,
#                        bboxes=None,
#                        labels=None,
#                        segms=None,
#                        class_names=None,
#                        score_thr=0,
#                        bbox_color='green',
#                        text_color='green',
#                        mask_color=None,
#                        thickness=2,
#                        font_size=13,
#                        win_name='',
#                        show=True,
#                        wait_time=0,
#                        out_file=None):
#     """Draw bboxes and class labels (with scores) on an image.
#
#     Args:
#         img (str | ndarray): The image to be displayed.
#         bboxes (ndarray): Bounding boxes (with scores), shaped (n, 5) or
#             (n, 6).
#         labels (ndarray): Labels of bboxes.
#         segms (ndarray | None): Masks, shaped (n,h,w) or None.
#         class_names (list[str]): Names of each classes.
#         score_thr (float): Minimum score of bboxes to be shown. Default: 0.
#         bbox_color (list[tuple] | tuple | str | None): Colors of bbox lines.
#            If a single color is given, it will be applied to all classes.
#            The tuple of color should be in RGB order. Default: 'green'.
#         text_color (list[tuple] | tuple | str | None): Colors of texts.
#            If a single color is given, it will be applied to all classes.
#            The tuple of color should be in RGB order. Default: 'green'.
#         mask_color (list[tuple] | tuple | str | None, optional): Colors of
#            masks. If a single color is given, it will be applied to all
#            classes. The tuple of color should be in RGB order.
#            Default: None.
#         thickness (int): Thickness of lines. Default: 2.
#         font_size (int): Font size of texts. Default: 13.
#         show (bool): Whether to show the image. Default: True.
#         win_name (str): The window name. Default: ''.
#         wait_time (float): Value of waitKey param. Default: 0.
#         out_file (str, optional): The filename to write the image.
#             Default: None.
#
#     Returns:
#         ndarray: The image with bboxes drawn on it.
#     """
#     assert bboxes is None or bboxes.ndim == 2, \
#         f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
#     assert labels.ndim == 1, \
#         f' labels ndim should be 1, but its ndim is {labels.ndim}.'
#     assert bboxes is None or bboxes.shape[1] == 5 or bboxes.shape[1] == 6, \
#         f' bboxes.shape[1] should be 5 or 6, but its {bboxes.shape[1]}.'
#     assert bboxes is None or bboxes.shape[0] <= labels.shape[0], \
#         'labels.shape[0] should not be less than bboxes.shape[0].'
#     assert segms is None or segms.shape[0] == labels.shape[0], \
#         'segms.shape[0] and labels.shape[0] should have the same length.'
#     assert segms is not None or bboxes is not None, \
#         'segms and bboxes should not be None at the same time.'
#
#     img = mmcv.imread(img).astype(np.uint8)
#
#     if score_thr > 0:
#         assert bboxes is not None and bboxes.shape[1] == 6
#         scores = bboxes[:, -1]
#         inds = scores > score_thr
#         bboxes = bboxes[inds, :]
#         labels = labels[inds]
#         if segms is not None:
#             segms = segms[inds, ...]
#
#     img = mmcv.bgr2rgb(img)
#     width, height = img.shape[1], img.shape[0]
#     img = np.ascontiguousarray(img)
#
#     fig = plt.figure(win_name, frameon=False)
#     plt.title(win_name)
#     canvas = fig.canvas
#     dpi = fig.get_dpi()
#     # add a small EPS to avoid precision lost due to matplotlib's truncation
#     # (https://github.com/matplotlib/matplotlib/issues/15363)
#     fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)
#
#     # remove white edges by set subplot margin
#     plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
#     ax = plt.gca()
#     ax.axis('off')
#
#     # np.set_printoptions(threshold=np.inf, suppress=True, precision=8)
#     # print(bboxes)
#
#     num_bboxes = 0
#     if bboxes is not None:
#         from mmrotate import obb2poly
#
#         num_bboxes = bboxes.shape[0]
#
#         # refined_det
#         refined_det_bboxes = np.array([[262.4471, 341.1879, 385.15024, 69.66197, -0.7401407, 0.9807444],
#                                        [853.977, 476.84784, 382.37787, 86.25415, -0.7523049, 0.9769369]],
#                                       dtype='float32')
#
#         max_label = int(max(labels) if len(labels) > 0 else 0)
#         text_palette = palette_val(get_palette([(68, 114, 196), ], max_label + 1))
#         text_colors = [text_palette[label] for label in labels]
#
#         bbox_palette = palette_val(get_palette([(68, 114, 196), ], max_label + 1))
#         colors = [bbox_palette[label] for label in labels[:num_bboxes]]
#
#         draw_rbboxes(ax, refined_det_bboxes, colors, alpha=0.8, thickness=thickness)
#
#         horizontal_alignment = 'left'
#         positions = obb2poly(torch.Tensor(refined_det_bboxes[:, :5]), version='oc')[:, :2].numpy().astype(
#             np.int32) + thickness
#         areas = refined_det_bboxes[:, 2] * refined_det_bboxes[:, 3]
#         scales = _get_adaptive_scales(areas)
#         scores = refined_det_bboxes[:, 5] if refined_det_bboxes.shape[1] == 6 else None
#
#         draw_labels(
#             ax,
#             labels[:num_bboxes],
#             positions,
#             scores=scores,
#             class_names=class_names,
#             color=text_colors,
#             font_size=font_size,
#             scales=scales,
#             horizontal_alignment=horizontal_alignment)
#
#         # retinanet
#         retinanet_bboxes = np.array([[851.4277, 479.69504, 340.38678, 88.15317, -0.603567, 0.6376234],
#                                      [264.70358, 342.5368, 302.36993, 97.3097, -0.5114058, 0.44625986]],
#                                     dtype='float32')
#
#         max_label = int(max(labels) if len(labels) > 0 else 0)
#         text_palette = palette_val(get_palette([(0, 255, 0), ], max_label + 1))
#         text_colors = [text_palette[label] for label in labels]
#
#         bbox_palette = palette_val(get_palette([(0, 255, 0), ], max_label + 1))
#         colors = [bbox_palette[label] for label in labels[:num_bboxes]]
#
#         draw_rbboxes(ax, retinanet_bboxes, colors, alpha=0.8, thickness=thickness)
#
#         horizontal_alignment = 'left'
#         positions = obb2poly(torch.Tensor(retinanet_bboxes[:, :5]), version='oc')[:, :2].numpy().astype(
#             np.int32) + thickness
#         areas = retinanet_bboxes[:, 2] * retinanet_bboxes[:, 3]
#         scales = _get_adaptive_scales(areas)
#         scores = retinanet_bboxes[:, 5] if retinanet_bboxes.shape[1] == 6 else None
#
#         draw_labels(
#             ax,
#             labels[:num_bboxes],
#             positions,
#             scores=scores,
#             class_names=class_names,
#             color=text_colors,
#             font_size=font_size,
#             scales=scales,
#             horizontal_alignment=horizontal_alignment)
#
#         # ground-truth box
#         gt_bboxes = np.array([[857.2496, 477.789, 386.8396, 75.0547, -0.7698947],
#                               [259.0615, 347.0315, 387.5469, 66.49368, -0.7750964]],
#                              dtype='float32')
#
#         max_label = int(max(labels) if len(labels) > 0 else 0)
#         text_palette = palette_val(get_palette([(255, 0, 0), ], max_label + 1))
#         text_colors = [text_palette[label] for label in labels]
#
#         bbox_palette = palette_val(get_palette([(255, 0, 0), ], max_label + 1))
#         colors = [bbox_palette[label] for label in labels[:num_bboxes]]
#
#         draw_rbboxes(ax, gt_bboxes, colors, alpha=0.8, thickness=thickness)
#
#         # plt.imshow(img)
#         # plt.show()
#         # print("1")
#
#     plt.imshow(img)
#
#     stream, _ = canvas.print_to_buffer()
#     buffer = np.frombuffer(stream, dtype='uint8')
#     img_rgba = buffer.reshape(height, width, 4)
#     rgb, alpha = np.split(img_rgba, [3], axis=2)
#     img = rgb.astype('uint8')
#     img = mmcv.rgb2bgr(img)
#
#     if show:
#         # We do not use cv2 for display because in some cases, opencv will
#         # conflict with Qt, it will output a warning: Current thread
#         # is not the object's thread. You can refer to
#         # https://github.com/opencv/opencv-python/issues/46 for details
#         if wait_time == 0:
#             plt.show()
#         else:
#             plt.show(block=False)
#             plt.pause(wait_time)
#     if out_file is not None:
#         mmcv.imwrite(img, out_file)
#
#     plt.close()
#
#     return img


def draw_labels(ax,
                labels,
                positions,
                scores=None,
                ious=None,
                class_names=None,
                color='w',
                font_size=8,
                scales=None,
                horizontal_alignment='left'):
    """Draw labels on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        labels (ndarray): The labels with the shape of (n, ).
        positions (ndarray): The positions to draw each labels.
        scores (ndarray): The scores for each labels.
        class_names (list[str]): The class names.
        color (list[tuple] | matplotlib.color): The colors for labels.
        font_size (int): Font size of texts. Default: 8.
        scales (list[float]): Scales of texts. Default: None.
        horizontal_alignment (str): The horizontal alignment method of
            texts. Default: 'left'.

    Returns:
        matplotlib.Axes: The result axes.
    """
    for i, (pos, label) in enumerate(zip(positions, labels)):
        # label_text = class_names[label] if class_names is not None else f'class {label}'
        label_text = ''
        if scores is not None:
            # label_text += f'Cls: {scores[i]:.02f}'
            label_text += f'Cls: {scores[i]:.02f} IoU: {ious[i]:.02f}'
        text_color = color[i] if isinstance(color, list) else color

        font_size_mask = font_size if scales is None else font_size * scales[i] * 2
        ax.text(
            pos[0],
            pos[1],
            f'{label_text}',
            bbox={
                'facecolor': text_color,
                'alpha': 0.6,
                'pad': 7,
                'edgecolor': 'none'
            },
            color='black',
            fontsize=font_size_mask,
            verticalalignment='top',
            horizontalalignment=horizontal_alignment)

    return ax


def draw_bboxes(ax, bboxes, color='g', alpha=0.8, thickness=2):
    """Draw bounding boxes on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        bboxes (ndarray): The input bounding boxes with the shape
            of (n, 4).
        color (list[tuple] | matplotlib.color): the colors for each
            bounding boxes.
        alpha (float): Transparency of bounding boxes. Default: 0.8.
        thickness (int): Thickness of lines. Default: 2.

    Returns:
        matplotlib.Axes: The result axes.
    """
    polygons = []
    for i, bbox in enumerate(bboxes):
        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
    p = PatchCollection(
        polygons,
        facecolor='none',
        edgecolors=color,
        linewidths=thickness,
        alpha=alpha)
    ax.add_collection(p)

    return ax


def imshow_det_bboxes(img,
                      bboxes=None,
                      labels=None,
                      segms=None,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      mask_color=None,
                      thickness=2,
                      font_size=8,
                      win_name='',
                      show=True,
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str | ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        segms (ndarray | None): Masks, shaped (n,h,w) or None.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown. Default: 0.
        bbox_color (list[tuple] | tuple | str | None): Colors of bbox lines.
           If a single color is given, it will be applied to all classes.
           The tuple of color should be in RGB order. Default: 'green'.
        text_color (list[tuple] | tuple | str | None): Colors of texts.
           If a single color is given, it will be applied to all classes.
           The tuple of color should be in RGB order. Default: 'green'.
        mask_color (list[tuple] | tuple | str | None, optional): Colors of
           masks. If a single color is given, it will be applied to all
           classes. The tuple of color should be in RGB order.
           Default: None.
        thickness (int): Thickness of lines. Default: 2.
        font_size (int): Font size of texts. Default: 13.
        show (bool): Whether to show the image. Default: True.
        win_name (str): The window name. Default: ''.
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes is None or bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels is None or labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes is None or bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    assert bboxes is None or bboxes.shape[0] <= labels.shape[0], \
        'labels.shape[0] should not be less than bboxes.shape[0].'
    assert segms is None or segms.shape[0] == labels.shape[0], \
        'segms.shape[0] and labels.shape[0] should have the same length.'
    # assert segms is not None or bboxes is not None, \
    #     'segms and bboxes should not be None at the same time.'

    img = mmcv.imread(img).astype(np.uint8)

    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    if labels is not None:
        max_label = int(max(labels) if len(labels) > 0 else 0)
        text_palette = palette_val(get_palette(text_color, max_label + 1))
        text_colors = [text_palette[label] for label in labels]

    num_bboxes = 0
    if bboxes is not None:
        num_bboxes = bboxes.shape[0]
        bbox_palette = palette_val(get_palette(bbox_color, max_label + 1))
        colors = [bbox_palette[label] for label in labels[:num_bboxes]]
        draw_bboxes(ax, bboxes, colors, alpha=0.8, thickness=thickness)

        horizontal_alignment = 'left'
        positions = bboxes[:, :2].astype(np.int32) + thickness
        areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        scales = _get_adaptive_scales(areas)
        scores = bboxes[:, 4] if bboxes.shape[1] == 5 else None
        draw_labels(
            ax,
            labels[:num_bboxes],
            positions,
            scores=scores,
            class_names=class_names,
            color=text_colors,
            font_size=font_size,
            scales=scales,
            horizontal_alignment=horizontal_alignment)

    if segms is not None:
        mask_palette = get_palette(mask_color, max_label + 1)
        colors = [mask_palette[label] for label in labels]
        colors = np.array(colors, dtype=np.uint8)
        draw_masks(ax, img, segms, colors, with_edge=True)

        if num_bboxes < segms.shape[0]:
            segms = segms[num_bboxes:]
            horizontal_alignment = 'center'
            areas = []
            positions = []
            for mask in segms:
                _, _, stats, centroids = cv2.connectedComponentsWithStats(
                    mask.astype(np.uint8), connectivity=8)
                largest_id = np.argmax(stats[1:, -1]) + 1
                positions.append(centroids[largest_id])
                areas.append(stats[largest_id, -1])
            areas = np.stack(areas, axis=0)
            scales = _get_adaptive_scales(areas)
            draw_labels(
                ax,
                labels[num_bboxes:],
                positions,
                class_names=class_names,
                color=text_colors,
                font_size=font_size,
                scales=scales,
                horizontal_alignment=horizontal_alignment)

    plt.imshow(img)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    if sys.platform == 'darwin':
        width, height = canvas.get_width_height(physical=True)
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    plt.close()

    return img
