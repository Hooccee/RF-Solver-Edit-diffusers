#!/bin/bash

export http_proxy=http://oversea-squid2.ko.txyun:11080
export https_proxy=http://oversea-squid2.ko.txyun:11080
export no_proxy="localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com"

python3 -u main_3op_enhance_39_SVD_shuffle_re.py \
    --model_path /mmu-vcg-hdd/caohaoxiang/FLUX.1-dev_rfsolver \
    --eval-datasets emu_edit_test_set_shuffle \
    --height 512     \
    --width 512     \
    --num-step 30     \
    --use_inversed_latents     \
    --ilist_json_path ./ilist_data_emu_edit_test_set.json     \
    --output_dir ./outputs/3op_enhance_emu_edit_test_set_SVD8     \
    --test     \
    --enhanced     \
    --stable_flow  \
    --guidance_scale 3     \
    --inject 6     \
    --nudge_source 1.15     \
    --nudge 1.05     \
    --SVD_start 0.96     \
    --SVD_end 0.85     \
    --K 3     \
    --alpha 3     \
    --v 3     \

    # --K 5     \
    # --alpha 3     \
    # --v 3     \

    #可调参数
    # parser.add_argument('--nudge', type=float, default=1.05, help='target inv分支潜变量标量偏移')参考范围：1.05~1.15，太大会使画质劣化
    # parser.add_argument('--guidance_scale', type=float, default=3.5, help='denoise 伪cfg 的引导比例') 参考范围：2.5~5.5，越小与原图越相似
    # parser.add_argument('--inject', type=int, default=8,help='inject步数') 参考范围：5~11，越大与原图越相似
    # parser.add_argument('--nudge_source', type=float, default=1.15, help='source inv分支潜变量标量偏移') 参考范围：1.05~1.15，越大与原图越相似
    # parser.add_argument('--K', type=int, default=5, help='SVD TopK 基数') 参考范围：2~7，越小与原图越相似
    # parser.add_argument('--alpha', type=float, default=3, help='SVD 增强 注意力权重放大系数') 参考范围：2~7，越小与原图越相似
    # parser.add_argument('--v: ', type=float, default=3, help='SVD 增强 Sigmoid斜率系数') 参考范围：2~7，越小与原图越相似
    # parser.add_argument('--SVD_start: ', type=float, default=0.96, help='SVD 增强 开始时间(denoise t 从1.0到0.0)') 不能太靠近1，又不能太远离1
    # parser.add_argument('--SVD_end: ', type=float, default=0.85, help='SVD 增强 结束时间(denoise t 从1.0到0.0)') 越小SVD增强越明显




# cd /mmu-vcg-hdd/caohaoxiang/mySrc/CausalCtrl/codes

# conda activate /mmu-vcg/caohaoxiang/env/test_1



