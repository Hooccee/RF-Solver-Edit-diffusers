import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import gc
from typing import Any, Dict, Optional, Tuple, Union
import torch
import math
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading

import json
import time
from openai import OpenAI


from PIL import Image
from diffusers import FluxPipeline,RfSolverFluxPipeline, RfSolverFluxTransformer2DModel
from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
import logging

from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from datasets import get_dataloader

from utils.utils import *
from utils.metrics import *


logger = logging.getLogger(__name__)   # pylint: disable=invalid-name





def pure_llm_ilist(source_tokens, target_tokens, api_key):
    """
    使用DeepSeek LLM生成i_list的解决方案（带重试机制）
    
    参数:
        source_tokens: 源文本分词结果
        target_tokens: 目标文本分词结果
        api_key: DeepSeek API密钥
    
    返回:
        list: 目标文本中需要增强的token位置索引列表
    """
    """ 添加线程安全措施 """
    # 每个线程创建独立client实例
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    
    # 重试配置参数
    max_retries = 3
    base_delay = 1  # 初始等待时间（秒）
    max_delay = 10  # 最大等待时间（秒）
    
    # 构建优化后的提示词
    prompt = f"""作为多模态语义分析专家，请执行以下任务：

[输入数据]
源分词序列：{source_tokens}
目标分词序列：{target_tokens}

[任务要求]
1. 对比分析两个token序列的语义差异
2. 识别以下类型的变化：
   - 对象/场景替换（如"▁desert→▁bed"）
   - 动作/关系变化（如"▁on→▁in"）
   - 属性修改（如"▁red→▁blue"）
3. 忽略：
   - 标点符号变化
   - 子词拆分差异（如"▁bedroom→▁bed+room"）
   - 停用词调整（如"a/an/the"变化）

[输出要求]
返回严格遵循此JSON结构：
{{
    "reasoning": [
        {{
            "original": "源token", 
            "target": "目标token",
            "target_indices": [索引],
            "change_type": "场景/动作/属性/对象" 
        }}
    ],
    "ilist": [增强位置索引]
}}

[示例1]
输入：
源: ['▁A', '▁desert', 'e', 'd', '▁highway', '▁near', '▁snow', '▁mountains', '▁under', '▁', 'a', '▁partly', '▁cloud', 'y', '▁sky']
目标: ['▁A', '▁red', '▁car', '▁on', '▁', 'a', '▁desert', 'e', 'd', '▁highway', '▁near', '▁snow', '▁mountains', '▁under', '▁', 'a', '▁partly', '▁cloud', 'y', '▁sky']
输出：
{{
    "ilist": [1,2,3],
    "reasoning": [
        {{"original": "''", "target": "'▁red', '▁car'", "target_indices": [1,2], "change_type": "对象"}},
        {{"original": "'▁on'", "target": "'▁in'", "target_indices": [3], "change_type": "动作"}}
    ]
}}

[示例2]
输入：
源: ['▁A', '▁person', '▁sit', 's', '▁on', '▁', 'a', '▁desert']
目标: ['▁A', '▁person', '▁sit', 's', '▁in', '▁', 'a','▁bedroom']
输出：
{{
    "ilist": [4,7],
    "reasoning": [
        {{"original": "'▁desert'", "target": "'▁bedroom'", "target_indices": [7], "change_type": "场景"}},
        {{"original": "'▁on'", "target": "'▁in'", "target_indices": [4], "change_type": "动作"}}
    ]
}}"""

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "..."},
                    {"role": "user", "content": prompt}
                ],
                temperature=1,
                response_format={"type": "json_object"},
                stream=False
            )
            
            # 解析响应
            result = json.loads(response.choices[0].message.content)
            # 修改打印部分为：
            print("\n" + "="*40 + " LLM完整响应 " + "="*40)
            print(json.dumps(result, 
                            indent=2, 
                            ensure_ascii=False, 
                            separators=(',', ': ')))
            print("=" * 90 + "\n")
            # 验证索引有效性
            max_valid_index = len(target_tokens) - 1
            validated_ilist = [
                min(idx, max_valid_index) 
                for idx in result.get("ilist", [])
            ]
            client.close()
            return list(set(validated_ilist))

        except Exception as e:
            if attempt == max_retries:  # 最后一次尝试仍然失败
                print(f"API调用失败，已达最大重试次数{max_retries}次。错误信息: {str(e)}")
                client.close()
                return []
            
            # 计算指数退避等待时间
            delay = min(base_delay * (2** attempt), max_delay)
            print(f"第{attempt+1}次尝试失败，{delay:.1f}秒后重试。错误信息: {str(e)}")
            time.sleep(delay)
    
    # 在返回前关闭连接
    client.close()
    return []  # 所有尝试失败后的最终返回


# 第一步：生成所有token数据并保存
def generate_and_save_tokens(dataloader, pipe, token_filename="token_data.json"):
    all_sources = []
    all_targets = []

    for img, source_prompt, target_prompt in dataloader:
        # 处理source提示
        source_inputs = pipe.tokenizer_2(
            source_prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        source_tokens = pipe.tokenizer_2.convert_ids_to_tokens(source_inputs.input_ids[0])
        try:
            end_idx = source_tokens.index('</s>')
            all_sources.append(source_tokens[:end_idx+1])
        except ValueError:
            all_sources.append(source_tokens)  # 处理没有</s>的情况

        # 处理target提示
        target_inputs = pipe.tokenizer_2(
            target_prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        target_tokens = pipe.tokenizer_2.convert_ids_to_tokens(target_inputs.input_ids[0])
        try:
            end_idx = target_tokens.index('</s>')
            all_targets.append(target_tokens[:end_idx+1])
        except ValueError:
            all_targets.append(target_tokens)

    # 保存到文件
    with open(token_filename, 'w') as f:
        json.dump({
            "sources": all_sources,
            "targets": all_targets
        }, f, indent=2)

# 修改后的第二步：多线程生成增强索引
def generate_and_save_ilist(token_filename="token_data.json", 
                           result_filename="ilist_data.json",
                           api_key="your_api_key",
                           max_workers=20):
    # 读取token数据
    with open(token_filename, 'r') as f:
        data = json.load(f)
    
    # 准备任务参数
    task_args = [
        (src[:-1], tgt[:-1], api_key) 
        for src, tgt in zip(data["sources"], data["targets"])
    ]
    
    # 线程安全的结果收集
    results = []
    lock = threading.Lock()
    
    # 单个任务处理函数
    def process_task(source, target, key):
        try:
            ilist = pure_llm_ilist(source, target, key)
            with lock:
                return {
                    "source_tokens": source + ['</s>'],  # 还原原始格式
                    "target_tokens": target + ['</s>'],
                    "ilist": ilist
                }
        except Exception as e:
            print(f"任务处理失败: {str(e)}")
            return None
    
    # 使用线程池处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for args in task_args:
            futures.append(executor.submit(process_task, *args))
        
        # 进度显示
        completed = 0
        total = len(futures)
        
        # 收集结果
        for future in futures:
            try:
                result = future.result()
                if result:
                    results.append(result)
                completed += 1
                print(f"进度: {completed}/{total} ({completed/total:.1%})", end='\r')
            except Exception as e:
                print(f"\n任务异常: {str(e)}")
    
    # 保存结果
    with open(result_filename, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

# 第三步：最终数据加载
def load_final_data(filename="ilist_data.json"):
    with open(filename, 'r') as f:
        return json.load(f)

@torch.inference_mode()
def main(args):

    if args.dtype == 'bfloat16':
        DTYPE = torch.bfloat16
    elif args.dtype == 'float16':
        DTYPE = torch.float16
    elif args.dtype == 'float32':
        DTYPE = torch.float32
    else:
        raise ValueError(f"不支持的数据类型: {args.dtype}")


    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"

    # ******** Loading pipeline **********
    pipe = RfSolverFluxPipeline.from_pretrained(args.model_path, torch_dtype=DTYPE)
    #pipe.to(device)
    # print(pipe.hf_device_map)
    pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()

    # ******** Input processing **********
    if args.eval_datasets == '':
        img = Image.open(args.image_path)
        train_transforms = transforms.Compose(
                    [
                        transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
                        transforms.CenterCrop(1024),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ]
                )

        img = train_transforms(img).unsqueeze(0)
        dataloader = [img, args.source_prompt, args.target_prompt]
    else:
        default_transform = transforms.Compose(
            [
                transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(1024),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        dataset = get_dataloader(args.eval_datasets,default_transform)
        dataloader = DataLoader(
            dataset,
            batch_size=1,          # 每批64个样本
            shuffle=False,           # 训练时打乱数据
            num_workers=8,          # 使用4个子进程加载数据
            pin_memory=True         # 如果使用GPU，可以加速数据传输
        )


        generate_and_save_tokens(dataloader, pipe)
        generate_and_save_ilist()
        final_dataset = load_final_data()
        
        # 验证数据加载
        print(f"加载到{len(final_dataset)}条数据")
        print("第一条数据示例:", final_dataset[0])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用不同参数测试 interpolated_denoise。')
    parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/Flux-dev', help='预训练模型的路径')
    parser.add_argument('--image_path', type=str, default='./example/image.png', help='输入图像的路径')
    parser.add_argument('--eval-datasets', type=str, default='', help='选择要编辑的数据集：EditEval_v1, PIE-Bench')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16', 'float32'], help='计算的数据类型')   
    args = parser.parse_args()
    main(args)
