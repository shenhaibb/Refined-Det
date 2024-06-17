# =========================================== train ===========================================
# hrsc
python tools/train.py \
  configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_6x_hrsc_rr_le90.py --work-dir work_dirs/hrsc/rotated_retinanet_obb_r50_fpn_6x_hrsc_rr_le90 --no-validate

python tools/train.py \
  configs/refined_det/hrsc/refined_det_cla_obb_r50_fpn_6x_hrsc_rr_le90.py --work-dir work_dirs/hrsc/refined_det_cla_obb_r50_fpn_6x_hrsc_rr_le90_2 --no-validate

python tools/train.py \
  configs/refined_det/hrsc/refined_det_frhead_obb_r50_fpn_6x_hrsc_rr_le90.py --work-dir work_dirs/hrsc/refined_det_frhead_obb_r50_fpn_6x_hrsc_rr_le90_align_conv --no-validate

python tools/train.py \
  configs/refined_det/hrsc/refined_det_obb_r50_fpn_6x_hrsc_rr_le90.py --work-dir work_dirs/hrsc/refined_det_obb_r50_fpn_6x_hrsc_rr_le90 --no-validate

python tools/train.py \
  configs/refined_det/hrsc/refined_det_obb_r101_fpn_6x_hrsc_rr_le90.py --work-dir work_dirs/hrsc/refined_det_obb_r101_fpn_6x_hrsc_rr_le90_800_800 --no-validate

# dota
python tools/train.py \
  configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py --work-dir work_dirs/dota/rotated_retinanet_obb_r50_fpn_1x_dota_le90 --no-validate

python tools/train.py \
  configs/refined_det/dota/refined_det_cla_obb_r50_fpn_1x_dota_le90.py --work-dir work_dirs/dota/refined_det_cla_obb_r50_fpn_1x_dota_le90 --no-validate

python tools/train.py \
  configs/refined_det/dota/refined_det_frhead_obb_r50_fpn_1x_dota_le90.py --work-dir work_dirs/dota/refined_det_frhead_obb_r50_fpn_1x_dota_le90_align_conv --no-validate

python tools/train.py \
  configs/refined_det/dota/refined_det_obb_r50_fpn_1x_dota_le90.py --work-dir work_dirs/dota/refined_det_obb_r50_fpn_1x_dota_le90 --no-validate

python tools/train.py \
  configs/refined_det/dota/refined_det_obb_r101_fpn_1x_dota_le90.py --work-dir work_dirs/dota/refined_det_obb_r101_fpn_1x_dota_le90 --no-validate

python tools/train.py \
  configs/refined_det/dota/refined_det_obb_r101_fpn_1x_dota_ms_rr_le90.py --work-dir work_dirs/dota/refined_det_obb_r101_fpn_1x_dota_ms_le90 --no-validate

# =========================================== evaluation ===========================================
# hrsc
python tools/test.py \
  configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_6x_hrsc_rr_le90.py  work_dirs/hrsc/rotated_retinanet_obb_r50_fpn_6x_hrsc_rr_le90/latest.pth --eval mAP

python tools/test.py \
  configs/refined_det/hrsc/refined_det_cla_obb_r50_fpn_6x_hrsc_rr_le90.py  work_dirs/hrsc/refined_det_cla_obb_r50_fpn_6x_hrsc_rr_le90_2/latest.pth --eval mAP

python tools/test.py \
  configs/refined_det/hrsc/refined_det_frhead_obb_r50_fpn_6x_hrsc_rr_le90.py  work_dirs/hrsc/refined_det_frhead_obb_r50_fpn_6x_hrsc_rr_le90_align_conv/latest.pth --eval mAP

python tools/test.py \
  configs/refined_det/hrsc/refined_det_obb_r50_fpn_6x_hrsc_rr_le90.py  work_dirs/hrsc/refined_det_obb_r50_fpn_6x_hrsc_rr_le90_416_416/latest.pth --eval mAP

python tools/test.py \
  configs/refined_det/hrsc/refined_det_obb_r101_fpn_6x_hrsc_rr_le90.py  work_dirs/hrsc/refined_det_obb_r101_fpn_6x_hrsc_rr_le90_800_800/latest.pth --eval mAP

# dota(official)
python ./tools/test.py  \
  configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py work_dirs/dota/rotated_retinanet_obb_r50_fpn_1x_dota_le90/latest.pth \
  --format-only --eval-options submission_dir=work_dirs/dota/rotated_retinanet_obb_r50_fpn_1x_dota_le90/rotated_retinanet_obb_r50_fpn_1x_dota_le90 nproc=1

python ./tools/test.py  \
  configs/refined_det/dota/refined_det_cla_obb_r50_fpn_1x_dota_le90.py work_dirs/dota/refined_det_cla_obb_r50_fpn_1x_dota_le90/latest.pth \
  --format-only --eval-options submission_dir=work_dirs/dota/refined_det_cla_obb_r50_fpn_1x_dota_le90/refined_det_cla_obb_r50_fpn_1x_dota_le90 nproc=1

python ./tools/test.py  \
  configs/refined_det/dota/refined_det_frhead_obb_r50_fpn_1x_dota_le90.py work_dirs/dota/refined_det_frhead_obb_r50_fpn_1x_dota_le90_align_conv/latest.pth \
  --format-only --eval-options submission_dir=work_dirs/dota/refined_det_frhead_obb_r50_fpn_1x_dota_le90_align_conv/refined_det_frhead_obb_r50_fpn_1x_dota_le90_align_conv nproc=1

python ./tools/test.py  \
  configs/refined_det/dota/refined_det_obb_r50_fpn_1x_dota_le90.py work_dirs/dota/refined_det_obb_r50_fpn_1x_dota_le90/latest.pth \
  --format-only --eval-options submission_dir=work_dirs/dota/refined_det_obb_r50_fpn_1x_dota_le90/refined_det_obb_r50_fpn_1x_dota_le90 nproc=1

python ./tools/test.py  \
  configs/refined_det/dota/refined_det_obb_r101_fpn_1x_dota_le90.py work_dirs/dota/refined_det_obb_r101_fpn_1x_dota_le90/latest.pth \
  --format-only --eval-options submission_dir=work_dirs/dota/refined_det_obb_r101_fpn_1x_dota_le90/refined_det_obb_r101_fpn_1x_dota_le90 nproc=1

python ./tools/test.py  \
  configs/refined_det/dota/refined_det_obb_r101_fpn_1x_dota_ms_rr_le90.py work_dirs/dota/refined_det_obb_r101_fpn_1x_dota_ms_le90/latest.pth \
  --format-only --eval-options submission_dir=work_dirs/dota/refined_det_obb_r101_fpn_1x_dota_ms_le90/refined_det_obb_r101_fpn_1x_dota_ms_le90 nproc=1

# =========================================== flops ===========================================
# hrsc
python tools/analysis_tools/get_flops.py \
  configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_6x_hrsc_rr_le90.py --shape 800 512

python tools/analysis_tools/get_flops.py \
  configs/refined_det/hrsc/refined_det_cla_obb_r50_fpn_6x_hrsc_rr_le90.py --shape 800 512

python tools/analysis_tools/get_flops.py \
  configs/refined_det/hrsc/refined_det_frhead_obb_r50_fpn_6x_hrsc_rr_le90.py --shape 800 512

python tools/analysis_tools/get_flops.py \
  configs/refined_det/hrsc/refined_det_frhead_obb_r50_fpn_6x_hrsc_rr_le90_abalation.py --shape 800 512

python tools/analysis_tools/get_flops.py \
  configs/refined_det/hrsc/refined_det_obb_r50_fpn_6x_hrsc_rr_le90.py --shape 800 512

# =========================================== visualize ===========================================
# hrsc
python tools/test.py \
  configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_6x_hrsc_rr_le90.py work_dirs/hrsc/rotated_retinanet_obb_r50_fpn_6x_hrsc_rr_le90/latest.pth \
  --show-dir work_dirs/vis/hrsc/rotated_retinanet_obb_r50_fpn_6x_hrsc_rr_le90

python tools/test.py \
  configs/refined_det/refined_det_cla_obb_r50_fpn_6x_hrsc_rr_le90.py work_dirs/hrsc/refined_det_cla_obb_r50_fpn_6x_hrsc_rr_le90/latest.pth \
  --show-dir work_dirs/vis/hrsc/refined_det_cla_obb_r50_fpn_6x_hrsc_rr_le90

python tools/test.py \
  configs/refined_det/refined_det_obb_r50_fpn_6x_hrsc_rr_le90.py work_dirs/hrsc/refined_det_obb_r50_fpn_6x_hrsc_rr_le90/latest.pth \
  --show-dir work_dirs/vis/hrsc/refined_det_obb_r50_fpn_6x_hrsc_rr_le90

# dota
python tools/test.py \
  configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py work_dirs/dota/rotated_retinanet_obb_r50_fpn_1x_dota_le90/latest.pth \
  --show-dir work_dirs/vis/dota/rotated_retinanet_obb_r50_fpn_1x_dota_le90

python tools/test.py \
  configs/refined_det/refined_det_obb_r50_fpn_1x_dota_le90.py work_dirs/dota/refined_det_obb_r50_fpn_1x_dota_le90/latest.pth \
  --show-dir work_dirs/vis/dota/refined_det_obb_r50_fpn_1x_dota_le90
