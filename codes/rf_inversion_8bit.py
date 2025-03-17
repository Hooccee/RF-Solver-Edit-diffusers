import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import gc
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torchvision.transforms import functional as F
import math
import argparse
from PIL import Image
from io import BytesIO
import requests
from tqdm import tqdm

from diffusers import (
    DiffusionPipeline,
    RfSolverFluxPipeline,
    RfSolverFluxTransformer2DModel,
    FluxTransformer2DModel,
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig
)
from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
import logging
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from datasets import get_dataloader

from utils.utils import *
from utils.metrics import *

logger = logging.getLogger(__name__)



def tensor_to_pil_uint8(tensor: torch.Tensor) -> Image.Image:
    """
    将通过 ToTensorWithoutScaling 转换的张量恢复为PIL图像
    输入张量要求：
      - 数据类型: torch.uint8
      - 数值范围: 0-255
      - 张量格式: (C, H, W)
    
    返回: PIL.Image (RGB模式)
    """
    # 确保张量在CPU且无梯度
    tensor = tensor.detach().cpu()
    
    # 转换为HWC格式 (PIL需要的格式)
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)  # CxHxW → HxWxC
    elif tensor.dim() == 4:
        tensor = tensor.squeeze(0).permute(1, 2, 0)  # 处理批次维度
    
    # 转换为PIL图像
    return Image.fromarray(tensor.numpy())

@torch.inference_mode()
def process_single_image(
    pipe: RfSolverFluxPipeline,
    image: Image.Image,
    target_prompt: str,
    args: argparse.Namespace
) -> Tuple[Image.Image, dict]:
    """处理单个图像的完整流程"""
    
    # 反转流程
    inverted_latents, image_latents, latent_image_ids = pipe.invert(
        image=image,
        num_inversion_steps=args.num_inversion_steps,
        gamma=args.gamma
    )
    
    # 编辑生成
    edited_images = pipe(
        prompt=target_prompt,
        inverted_latents=inverted_latents,
        image_latents=image_latents,
        latent_image_ids=latent_image_ids,
        start_timestep=args.start_timestep,
        stop_timestep=args.stop_timestep,
        num_inference_steps=args.num_inference_steps,
        eta=args.eta,
        guidance_scale=args.guidance_scale
    ).images
    
    # 后处理
    edited_image = edited_images[0]
    
    return edited_image, {
        "inverted_latents": inverted_latents,
        "image_latents": image_latents,
        "latent_image_ids": latent_image_ids
    }

@torch.inference_mode()
def main(args):
    # 初始化配置
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设备与精度设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32
    }
    dtype = dtype_map[args.dtype]
    
    # 量化配置
    quant_config = DiffusersBitsAndBytesConfig(
        load_in_8bit=args.quant_8bit,
        load_in_4bit=args.quant_4bit
    )
    
    # 加载模型
    transformer = FluxTransformer2DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        quantization_config=quant_config,
        torch_dtype=dtype
    )
    pipe = DiffusionPipeline.from_pretrained(
        "/data/chx/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        transformer=transformer,
        custom_pipeline="/home/chx/mySrc/diffusers-dev-Bob/examples/community/pipeline_flux_rf_inversion")
    pipe.enable_model_cpu_offload()

    
    # 数据加载
    class ToTensorWithoutScaling(object):
        """增强鲁棒性的张量转换"""
        def __call__(self, pic):
            try:
                if pic is None:
                    raise ValueError("输入图像为空")
                    
                if isinstance(pic, Image.Image):
                    # 确保图像有效
                    pic.load()
                    # 转换逻辑
                    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
                    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
                    img = img.permute((2, 0, 1)).contiguous()
                    return img.to(dtype=torch.uint8)
                else:
                    raise TypeError(f"输入类型错误: {type(pic)}")
            except Exception as e:
                print(f"图像转换失败: {str(e)}")
                # 返回占位张量
                return torch.zeros(3, args.height, args.width, dtype=torch.uint8)
            
    transform = transforms.Compose([
                                    transforms.Resize((args.height, args.width), interpolation=transforms.InterpolationMode.BILINEAR) ,
                                    ToTensorWithoutScaling()  # 保持0-255范围
                                    ])
    
    dataset = get_dataloader(
        args.eval_dataset,
        transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 评估指标初始化
    metrics = {
        'clip_score': 0.0,
        'mse': 0.0,
        'psnr': 0.0,
        'lpips': 0.0,
        'ssim': 0.0
    }
    metric_calculator = metircs()
    
    # 处理循环
    progress = tqdm(dataloader, desc="Processing")
    for batch_idx, (images, source_prompts, target_prompts) in enumerate(progress):
        # 处理批量数据
        batch_metrics = {
                        'clip_score': 0.0,
                        'mse': 0.0,
                        'psnr': 0.0,
                        'lpips': 0.0,
                        'count': 0
                        }
        for idx in range(len(images)):
            # try:
                # 处理单张图像
                images_pil=tensor_to_pil_uint8(images[idx])
                edited_img, _ = process_single_image(
                    pipe=pipe,
                    image=images_pil,
                    target_prompt=target_prompts[idx],
                    args=args
                )
                
                # 转换回Tensor
                edited_tensor = transforms.Compose(
                    [
                    transforms.Resize((args.height, args.width), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                    ]
                    )(edited_img).unsqueeze(0).to(device)
                
                orig_tensor = transforms.Compose(
                    [
                    transforms.Resize((args.height, args.width), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                    ]
                    )(images_pil).unsqueeze(0).to(device)
                
                # 计算指标
                # try:
                # 分别计算各指标
                clip_score = metric_calculator.clip_scores(edited_tensor, target_prompts[idx])
                mse = metric_calculator.mse_scores(edited_tensor, orig_tensor)
                psnr_val = metric_calculator.psnr_scores(edited_tensor, orig_tensor)
                lpips_val = metric_calculator.lpips_scores(edited_tensor, orig_tensor)
                
                # 打印样本级指标

                print(f"\nSample {batch_idx}-{idx} Metrics:")
                print(source_prompts[idx])
                print(target_prompts[idx])
                print(f"CLIP Score: {clip_score:.4f}")
                print(f"MSE: {mse:.4f}")
                print(f"PSNR: {psnr_val:.4f} dB")
                print(f"LPIPS: {lpips_val:.4f}")
                
                # 累加到批指标
                batch_metrics['clip_score'] += clip_score
                batch_metrics['mse'] += mse
                batch_metrics['psnr'] += psnr_val
                batch_metrics['lpips'] += lpips_val
                batch_metrics['count'] += 1

                # except Exception as e:
                #     logger.error(f"指标计算失败: {str(e)}")
                #     continue
                
                # 保存结果
                if args.save_samples:
                    save_path = os.path.join(args.output_dir, f"result_{batch_idx}_{idx}.png")
                    edited_img.save(save_path)
                    
            # except Exception as e:
            #     logger.error(f"Error processing sample {batch_idx}-{idx}: {str(e)}")
        
        # 更新总指标
        print(f"\n{'='*40}")
        print(f"Batch {batch_idx} 指标汇总:")
        
        # 遍历所有指标进行更新和打印
        for key in metrics:
            # 计算本批次该指标的平均值
            batch_avg = batch_metrics[key] / batch_metrics['count']
            
            # 更新总指标
            metrics[key] += batch_avg
            
            # 格式化输出
            print(f"[{key.upper()}]:")
            print(f"  本批次平均值: {batch_avg:.4f}")
            print(f"  累计总指标值: {metrics[key]:.4f}")
            print("-"*30)
        
        # 释放显存
        torch.cuda.empty_cache()
        print(f"已清理显存，当前占用: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
    # 计算平均指标
    num_samples = len(dataloader.dataset)
    final_metrics = {
        key: metrics[key] / num_samples for key in metrics
    }
    
    # 输出结果
    print("\nFinal Evaluation Metrics:")
    for metric, value in final_metrics.items():
        print(f"{metric.upper():<10} | {value:.4f}")
        
    # 清理资源
    del pipe, transformer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量评估图像编辑流程")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--quant_8bit", action="store_true")
    parser.add_argument("--quant_4bit", action="store_true")
    
    # 数据参数
    parser.add_argument("--eval_dataset", type=str, required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    
    # 流程参数
    parser.add_argument("--num_inversion_steps", type=int, default=28)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=0.9)
    parser.add_argument("--start_timestep", type=float, default=0.0)
    parser.add_argument("--stop_timestep", type=float, default=0.25)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--save_samples", action="store_true")
    
    args = parser.parse_args()
    
    # 参数校验
    if args.quant_4bit and args.quant_8bit:
        raise ValueError("不能同时启用4bit和8bit量化")
        
    main(args)