#!/usr/bin/env bash

GPU_ID=$1

CUDA_VISIBLE_DEVICES=$GPU_ID python tools/train.py configs/gta2city/r1_ckd_dlv2_r101v1c_1x4_512x512_40k_gta2city.py \
  --cfg-options data.train.post_pmd=False data.train.post_blur=False --work-dir work_dirs/tune_aug

CUDA_VISIBLE_DEVICES=$GPU_ID python tools/train.py configs/gta2city/r1_ckd_dlv2_r101v1c_1x4_512x512_40k_gta2city.py \
  --cfg-options data.train.post_pmd=False data.train.post_blur=False --work-dir work_dirs/tune_aug

CUDA_VISIBLE_DEVICES=$GPU_ID python tools/train.py configs/gta2city/r1_ckd_dlv2_r101v1c_1x4_512x512_40k_gta2city.py \
  --cfg-options data.train.post_pmd=False data.train.post_blur=False --work-dir work_dirs/tune_aug
