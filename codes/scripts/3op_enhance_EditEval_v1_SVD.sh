python3  main_3op_enhance_39_SVD.py \
    --model_path /data/chx/FLUX.1-dev_RF-Solver     \
    --eval-datasets EditEval_v1     \
    --height 1024     \
    --width 1024     \
    --num-step 30     \
    --use_inversed_latents     \
    --ilist_json_path ./ilist_data_EditEval_v1.json     \
    --output_dir ./outputs/3op_enhance_EditEval_v1_SVD     \
    --test     \
    --enhanced     \
    --stable_flow  \
    --guidance_scale 3.5     \
    --inject 8     \
    --nudge_source 1.15     \
    --nudge 1.05     \
    --SVD_start 0.96     \
    --SVD_end 0.85     \
    --K 5     \
    --alpha 3     \
    --v 3     \


    #可调参数
    # parser.add_argument('--nudge', type=float, default=1.05, help='target inv分支潜变量标量偏移')
    # parser.add_argument('--guidance_scale', type=float, default=3.5, help='denoise 伪cfg 的引导比例')
    # parser.add_argument('--inject', type=int, default=8,help='inject步数')
    # parser.add_argument('--nudge_source', type=float, default=1.15, help='source inv分支潜变量标量偏移')
    # parser.add_argument('--K', type=int, default=5, help='SVD TopK 基数')
    # parser.add_argument('--alpha', type=float, default=3, help='SVD 增强 注意力权重放大系数')
    # parser.add_argument('--v: ', type=float, default=3, help='SVD 增强 Sigmoid斜率系数')
    # parser.add_argument('--SVD_start: ', type=float, default=0.96, help='SVD 增强 开始时间(denoise t 从1.0到0.0)')
    # parser.add_argument('--SVD_end: ', type=float, default=0.85, help='SVD 增强 结束时间(denoise t 从1.0到0.0)')