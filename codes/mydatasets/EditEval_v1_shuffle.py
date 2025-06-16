from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import json

default_rootpath = '/data/chx/EditEval_v1/Dataset'
default_csvpath = '/data/chx/EditEval_v1/Dataset/editing_prompts_collection.xlsx'
default_ilistpath = '/home/chx/mySrc/diffusers-dev-Bob/codes/ilist_data_EditEval_v1.json'  # 根据实际情况设置默认路径

class EditEval_v1_datase_shuffle(Dataset):
    def __init__(self, transform=None,
                 csvpath=default_csvpath,
                 rootpath=default_rootpath,
                 ilistpath=default_ilistpath):
        super(EditEval_v1_datase_shuffle, self).__init__()
        self.transform = transform
        self.csvpath = csvpath
        self.rootpath = rootpath
        self.ilistpath = ilistpath

        df = pd.read_excel(self.csvpath)
        
        # 读取数据列
        self.edit_class = df['Edit Class'].tolist()
        self.source_prompt = df['Source Prompt'].tolist()
        self.target_prompt = df['Target Prompt'].tolist()
        
        self.samples = []
        class_counter = {}
        current_class = None

        for item, s, t in zip(self.edit_class, self.source_prompt, self.target_prompt):
            if not pd.isna(item):
                formatted_class = item.strip().lower().replace(' ', '_')
                formatted_class = formatted_class.replace('object_removal', 'object_removel')
                current_class = formatted_class
                if current_class not in class_counter:
                    class_counter[current_class] = 1

            if current_class is None:
                raise ValueError("CSV文件中缺少初始类别定义")
            
            path = f"{current_class}/{class_counter[current_class]}.jpg"
            self.samples.append({
                "image_path": path,
                "source_prompt": s,
                "target_prompt": t,
                "edit_class": current_class
            })
            class_counter[current_class] += 1

        # 加载ilist数据
        with open(self.ilistpath, 'r') as f:
            ilist_data = json.load(f)

        if len(self.samples) != len(ilist_data):
            raise ValueError(f"数据长度不一致！Excel: {len(self.samples)}, ilist: {len(ilist_data)}")

        # 添加ilist到每个样本
        for idx, sample in enumerate(self.samples):
            sample['ilist'] = ilist_data[idx]['ilist']

    def __getitem__(self, index):
        sample = self.samples[index]
        impath = os.path.join(self.rootpath, sample['image_path'])
        
        if not os.path.exists(impath):
            impath = impath.replace('.jpg', '.jpeg')
        
        img = Image.open(impath).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return (
            img,
            sample['source_prompt'],
            sample['target_prompt'],
            sample['ilist']
        )

    def __len__(self):
        return len(self.samples)