from torch.utils.data import Dataset
import json
from PIL import Image
import os

default_rootpath = "/data/lyw/PIE-Benchmark/annotation_images"
default_jsonpath = "/data/lyw/PIE-Benchmark/mapping_file.json"
default_ilistpath = "/home/chx/mySrc/diffusers-dev-Bob/codes/ilist_data_PIE-Bench.json"

class PIE_Bench_dataset_shauffle(Dataset):
    def __init__(self, transform=None,
                 jsonpath=default_jsonpath,
                 rootpath=default_rootpath,
                 ilistpath=default_ilistpath):
        super(PIE_Bench_dataset_shauffle, self).__init__()
        self.transform = transform
        self.jsonpath = jsonpath
        self.rootpath = rootpath
        self.ilistpath = ilistpath

        # 加载原始JSON元数据
        with open(self.jsonpath, 'r') as f:
            metadata = json.load(f)
        
        # 加载ilist数据
        with open(self.ilistpath, 'r') as f:
            ilist_data = json.load(f)

        self.samples = []
        # 添加数据长度一致性检查
        if len(metadata) != len(ilist_data):
            raise ValueError(f"数据长度不一致！metadata: {len(metadata)}, ilist_data: {len(ilist_data)}")

        for count, (idx, item) in enumerate(metadata.items()):
            # 严格检查ilist数据格式
            try:
                current_ilist = ilist_data[count]['ilist']
            except IndexError:
                raise IndexError(f"ilist_data索引越界，请检查第{count}项数据是否存在")
            except KeyError:
                raise KeyError(f"ilist_data第{count}项缺少'ilist'字段")
                
            # 解析原始数据
            rel_image_path = item['image_path']
            source_prompt = item['original_prompt'].replace('[', '').replace(']', '')
            target_prompt = item['editing_prompt'].replace('[', '').replace(']', '')
            
            full_image_path = os.path.join(self.rootpath, rel_image_path)
            edit_class = self._parse_edit_class(rel_image_path)
            
            self.samples.append({
                "image_path": full_image_path,
                "source_prompt": source_prompt,
                "target_prompt": target_prompt,
                "edit_class": edit_class,
                "ilist": current_ilist
            })

    def _parse_edit_class(self, rel_image_path):
        """解析路径生成编辑类别"""
        dirs = rel_image_path.split('/')[:-1]
        keywords = []
        for dir_part in dirs:
            parts = dir_part.split('_')
            semantic_parts = parts[1:-1] if parts[-1].isdigit() else parts[1:]
            keywords.append('_'.join(semantic_parts))
        return '-'.join(keywords)

    def __getitem__(self, index):
        sample = self.samples[index]
        
        # 处理图像路径
        image_path = sample["image_path"]
        if not os.path.exists(image_path):
            image_path = image_path.replace(".jpg", ".jpeg")

        # 加载图像
        img = Image.open(image_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return (
            img,
            sample["source_prompt"],
            sample["target_prompt"],
            sample["ilist"]
        )

    def __len__(self):
        return len(self.samples)