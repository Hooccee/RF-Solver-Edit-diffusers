#!/bin/bash
set -e

export http_proxy=http://oversea-squid2.ko.txyun:11080 https_proxy=http://oversea-squid2.ko.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

export WANDB_API_KEY=ad36981ec2025780579f44d03081d7524a222cef

# 日志目录
LOG_DIR="logs_0801"
mkdir -p $LOG_DIR

# 训练参数
PRETRAINED_MODEL="/mmu-vcg-hdd/caohaoxiang/model/FLUX.1-Kontext-dev"
OUTPUT_DIR="openrr1k_hf-i2i_RR4K_openrr"
# DATASET_NAME="/mmu-vcg-hdd/caohaoxiang/dataset/Single_Image_Reflection_Removal/openrr1k_hf_with_caption"
DATASET_NAME="/mmu-vcg-hdd/caohaoxiang/dataset/Single_Image_Reflection_Removal/openrr1k_rr4k_imagefolder_resized"
IMAGE_COLUMN="transmission_layer"
COND_IMAGE_COLUMN="blended"
CAPTION_COLUMN="caption"
# ASPECT_RATIO_BUCKETS="1024,768;768,1024;1024,576;1024,872;688,1024;576,1024;768,1024;616,872;760,600;928,608"
ASPECT_RATIO_BUCKETS="672,1568;688,1504;720,1456;752,1392;800,1328;832,1248;880,1184;944,1104;1024,1024;1104,944;1184,880;1248,832;1328,800;1392,752;1456,720;1504,688;1568,672"
TRAIN_BATCH_SIZE=8
GUIDANCE_SCALE=1
GRAD_ACC_STEPS=1
OPTIMIZER="adamw"
LEARNING_RATE=1e-4
LR_SCHEDULER="constant"
LR_WARMUP_STEPS=200
MAX_TRAIN_STEPS=2600
RANK=16
SEED=0

# 启动训练\  --instance_data_dir "./meta_data" 存放原数据
accelerate  launch train_dreambooth_lora_flux_kontext.py \
  --pretrained_model_name_or_path="$PRETRAINED_MODEL" \
  --output_dir="$OUTPUT_DIR" \
  --dataset_name="$DATASET_NAME" \
  --image_column="$IMAGE_COLUMN" \
  --cond_image_column="$COND_IMAGE_COLUMN" \
  --caption_column="$CAPTION_COLUMN" \
  --mixed_precision="bf16" \
  --aspect_ratio_buckets="$ASPECT_RATIO_BUCKETS" \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --guidance_scale=$GUIDANCE_SCALE \
  --gradient_accumulation_steps=$GRAD_ACC_STEPS \
  --gradient_checkpointing \
  --optimizer="$OPTIMIZER" \
  --use_8bit_adam \
  --learning_rate=$LEARNING_RATE \
  --lr_scheduler="$LR_SCHEDULER" \
  --lr_warmup_steps=$LR_WARMUP_STEPS \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --rank=$RANK \
  --seed="$SEED" \
  2>&1 | tee $LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log