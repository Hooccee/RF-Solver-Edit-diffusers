from torch.utils.data import Dataset
import json
from PIL import Image
import os

default_rootpath = "/data/lyw/PIE-Benchmark/annotation_images"
default_jsonpath = "/data/lyw/PIE-Benchmark/mapping_file.json"

class PIE_Bench_dataset(Dataset):
    def __init__(self, transform=None,
                 jsonpath=default_jsonpath,
                 rootpath=default_rootpath):
        super(PIE_Bench_dataset, self).__init__()
        self.transform = transform
        self.jsonpath = jsonpath
        self.rootpath = rootpath
        
        # 加载JSON元数据文件
        with open(self.jsonpath, 'r') as f:
            metadata = json.load(f)
        
        self.samples = []
        # 遍历JSON中的每一项
        for idx, item in metadata.items():
            # 提取关键信息
            rel_image_path = item['image_path']  # 相对路径，如 "5_change_attribute_pose_40/.../xxx.jpg"
            source_prompt = item['original_prompt'].replace('[', '').replace(']', '')  # 清理方括号
            target_prompt = item['editing_prompt'].replace('[', '').replace(']', '')
            
            # 生成完整图像路径
            full_image_path = os.path.join(self.rootpath, rel_image_path)
            
            # 解析编辑类别（从路径中提取语义关键词）
            edit_class = self._parse_edit_class(rel_image_path)
            
            self.samples.append({
                "image_path": full_image_path,
                "source_prompt": source_prompt,
                "target_prompt": target_prompt,
                "edit_class": edit_class
            })
    
    def _parse_edit_class(self, rel_image_path):
        """解析路径生成编辑类别（如 'change_attribute_pose-artificial-indoor'）"""
        dirs = rel_image_path.split('/')[:-1]  # 去掉文件名，保留目录层级
        keywords = []
        for dir_part in dirs:
            # 示例目录格式: "5_change_attribute_pose_40" → 提取 "change_attribute_pose"
            parts = dir_part.split('_')
            # 跳过数字前缀（如 "5_"）和尾部数字（如 "_40"）
            semantic_parts = parts[1:-1] if parts[-1].isdigit() else parts[1:]
            keywords.append('_'.join(semantic_parts))
        return '-'.join(keywords)  # 连接关键词
    
    def __getitem__(self, index):
        sample = self.samples[index]
        # 处理可能的文件扩展名问题（如 .jpg 或 .jpeg）
        image_path = sample["image_path"]
        if not os.path.exists(image_path):
            image_path = image_path.replace(".jpg", ".jpeg")
        
        # 加载图像
        img = Image.open(image_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        #     print("if yes",img.shape)
        # print("img.shape",img.shape)
        return (
            img,                       # 图像张量
            sample["source_prompt"],   # 字符串
            sample["target_prompt"]    # 字符串
        )
        
    def __len__(self):
        return len(self.samples)