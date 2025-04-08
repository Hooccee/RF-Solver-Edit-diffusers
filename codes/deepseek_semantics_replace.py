import json
import random
from openai import OpenAI
from typing import List, Dict
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def process_item(item: Dict, api_key: str, num_variations: int = 10) -> Dict:
    """
    多线程处理单个数据项的核心函数
    单次请求生成所有variation
    """
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    source_tokens = item['source_tokens']
    target_tokens = item['target_tokens']
    ilist = item['ilist']
    variations = {}

    if not ilist:
        return None

    # 构建一次性生成所有variation的prompt
    prompt = f"""# Multi-Variation Semantic Replacement Task

## Source Context
• Original token sequence: {target_tokens}
• Replacement positions: {ilist} (0-based index)
• Required variations: {num_variations}

## Generation Requirements
1. **Diversity First Principle**:
   - Generate {num_variations} DISTINCTLY DIFFERENT variations in a single response
   - Each variation must use DIFFERENT replacement strategies from the defined types

2. **Variation Types (MUST USE AT LEAST 8 DIFFERENT TYPES)**:
   [Object Addition] Add new objects to the scene (e.g., "→ ▁vase")
   [Object Replacement] Replace existing objects (e.g., "▁dog → ▁cat")
   [Object Removal] Remove objects (replace with "▁[REMOVED]")
   [Background Change] Alter scene setting (e.g., "▁office → ▁beach")
   [Style Change] Modify artistic style (e.g., "▁realistic → ▁cartoon")
   [Texture Change] Change material properties (e.g., "▁wooden → ▁metallic")
   [Action Change] Modify actions/verbs (e.g., "▁running → ▁jumping")
   [Color Change] Alter colors (e.g., "▁red → ▁blue")
   [Lighting Change] Adjust illumination (e.g., "▁daylight → ▁moonlight")
   [Perspective Change] Modify viewpoint (e.g., "▁close-up → ▁aerial")

3. **Replacement Constraints**:
   - For each variation, modify ALL specified positions ({ilist})
   - No two variations should use the same combination of replacement types
   - Avoid minor changes (e.g., synonyms or slight modifications)

4. **Special Cases**:
   - For Object Removal: Replace with special token "▁[REMOVED]"
   - For Object Addition: Use "▁[ADDED:<object>]" format
   - For Style/Texture changes: Clearly indicate the modification type

## Output Format Rules
Return STRICTLY in this JSON format:
{{
    "variations": [
        {{
            "variation_id": 1,
            "original_tokens": ["token1", "token2"],
            "replacement_tokens": ["new1", "new2"],
            "positions": [position_indices],
            "change_type": ["ObjectAddition/Replacement/Removal/etc"],
            "new_sequence": ["new", "token", "sequence"],
            "variation_description": "Brief description of the change"
        }},
        ... // {num_variations} variations total
    ]
}}

## Example Output for 4 Variations (positions [1,3]):
{{
    "variations": [
        {{
            "variation_id": 1,
            "original_tokens": ["▁black", "▁park"],
            "replacement_tokens": ["▁[REMOVED]", "▁garden"],
            "positions": [1,3],
            "change_type": ["ObjectRemoval", "BackgroundChange"],
            "new_sequence": ["▁A", "▁[REMOVED]", "▁dog", "▁garden"],
            "variation_description": "Removed the color attribute and changed setting to garden"
        }},
        {{
            "variation_id": 2,
            "original_tokens": ["▁black", "▁park"],
            "replacement_tokens": ["▁polka-dotted", "▁circus"],
            "positions": [1,3],
            "change_type": ["TextureChange", "BackgroundChange"],
            "new_sequence": ["▁A", "▁polka-dotted", "▁dog", "▁circus"],
            "variation_description": "Changed texture to polka-dotted and moved scene to circus"
        }},
        {{
            "variation_id": 3,
            "original_tokens": ["▁black", "▁park"],
            "replacement_tokens": ["▁golden", "▁[ADDED:▁fountain]"],
            "positions": [1,3],
            "change_type": ["ColorChange", "ObjectAddition"],
            "new_sequence": ["▁A", "▁golden", "▁dog", "▁[ADDED:▁fountain]"],
            "variation_description": "Changed color to golden and added fountain"
        }},
        {{
            "variation_id": 4,
            "original_tokens": ["▁black", "▁park"],
            "replacement_tokens": ["▁watercolor", "▁gallery"],
            "positions": [1,3],
            "change_type": ["StyleChange", "BackgroundChange"],
            "new_sequence": ["▁A", "▁watercolor", "▁dog", "▁gallery"],
            "variation_description": "Applied watercolor style and moved to art gallery"
        }}
    ]
}}"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Expert semantic variation generator"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,
            top_p=0.95,
            response_format={"type": "json_object"},
            stream=False
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Get original detokenized sentence
        original_detokenized = ' '.join([token.lstrip("▁") for token in target_tokens])
        
        # Process variations
        for var in result['variations']:
            var_id = var['variation_id']
            
            # Process new sequence for clean sentence
            clean_tokens = []
            for token in var['new_sequence']:
                if token == "▁[REMOVED]":
                    continue  # Skip removed tokens
                elif token.startswith("▁[ADDED:"):
                    # Extract added object and format
                    added_obj = token[8:-1].lstrip("▁")
                    clean_tokens.append(added_obj)
                else:
                    clean_tokens.append(token.lstrip("▁"))
                    
            variations[f'variation_{var_id}'] = {
                'modified_positions': var['positions'],
                'original_tokens': var['original_tokens'],
                'replacement_tokens': var['replacement_tokens'],
                'new_target_tokens': var['new_sequence'],
                'new_target_sentence': ' '.join(clean_tokens),
                'original_target_sentence': original_detokenized,
                'num_replaced': len(var['positions']),
                'change_type': var['change_type']
            }
            
    except Exception as e:
        print(f"Error processing item: {e}")
        return None
    finally:
        client.close()

    return {'original_data': item, 'variations': variations} if variations else None

def generate_semantic_variations(
    json_file: str,
    output_file: str,
    api_key: str,
    num_variations: int = 10,
    max_workers: int = 5
):
    """
    多线程版本的主函数
    每组target tokens生成10组随机语义修改
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 使用字典来保存结果，键为原始索引
    results = {}
    lock = threading.Lock()
    total_items = len(data)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务，并保留原始索引
        futures = {
            executor.submit(
                process_item,
                item=item,
                api_key=api_key,
                num_variations=num_variations
            ): idx for idx, item in enumerate(data)
        }

        # 进度跟踪
        def update_progress():
            nonlocal completed
            with lock:
                completed += 1
                print(f"进度: {completed}/{total_items} ({completed/total_items:.1%})", end='\r')

        # 处理完成的任务
        for future in as_completed(futures):
            try:
                result = future.result()
                original_idx = futures[future]  # 获取原始索引
                if result:
                    with lock:
                        results[original_idx] = result  # 使用原始索引作为键
                update_progress()
            except Exception as e:
                update_progress()
                continue

    # 按原始顺序排序并保存结果
    sorted_results = [results[idx] for idx in sorted(results.keys())]
    
    # 保存结果
    with open(output_file, 'w') as f:
        json.dump(sorted_results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='多线程语义增强生成器')
    parser.add_argument('--input', required=True, help='输入JSON文件路径')
    parser.add_argument('--output', default='output_variations.json', help='输出文件路径')
    parser.add_argument('--api_key', type=str, required=True, help='API访问密钥')
    parser.add_argument('--threads', type=int, default=8, help='最大并发线程数')
    parser.add_argument('--num_variations', type=int, default=10, help='每组生成多少种语义变化')
    
    args = parser.parse_args()
    
    generate_semantic_variations(
        json_file=args.input,
        output_file=args.output,
        api_key=args.api_key,
        num_variations=args.num_variations,
        max_workers=args.threads
    )


#     nohup python3 your_script.py \
#   --input input.json \
#   --output output_variations.json \
#   --api_key your-api-key-here \
#   --threads 8 \
#   --num_variations 10 \
#   > semantic_variations.log 2>&1 &