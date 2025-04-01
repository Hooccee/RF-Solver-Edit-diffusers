from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch.nn as nn
import torch
from torchvision import transforms
import torchmetrics
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.regression import MeanSquaredError

class metircs:  # 输入图像值范围均为[-1,1]
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 初始化CLIP评分指标
        self.clip_metric_calculator = CLIPScore(model_name_or_path="/data/chx/clip-vit-base-patch32").to(self.device)
        
        # 初始化MSE评分指标
        self.mse_metric_calculator = MeanSquaredError().to(self.device)
        
        # 初始化PSNR评分指标
        self.psnr_metric_calculator = PeakSignalNoiseRatio(data_range=2.0).to(self.device)
        
        # 初始化LPIPS评分指标
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(self.device)

        # 初始化SSIM评分指标
        self.ssim_metric_calculator = StructuralSimilarityIndexMeasure(data_range=2.0).to(self.device)

        # 加载DINOv2模型和处理器
        model_folder = '/data/chx/dinov2-base'
        self.dino_processor = AutoImageProcessor.from_pretrained(model_folder)
        self.dino_model = AutoModel.from_pretrained(model_folder).to(self.device)
        self.dino_model.eval()

    def _tensor_to_pil(self, tensor):
        """将[-1,1]范围的Tensor转换为PIL图像"""
        # 处理可能的批量维度（当输入是4维时）
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)  # 从[B C H W] -> [C H W]
        
        # 转换为[0,1]范围
        image_01 = (tensor + 1) / 2
        # 调整维度顺序并转换为uint8
        image_uint8 = (image_01.permute(1, 2, 0) * 255).clamp(0, 255).to(torch.uint8)
        # 转换为PIL图像
        return Image.fromarray(image_uint8.cpu().numpy())

    def dino_scores(self, image1, image2):
        """计算两幅图像之间的DINO特征相似度"""
        # 转换Tensor到PIL图像
        image1_pil = self._tensor_to_pil(image1)
        image2_pil = self._tensor_to_pil(image2)
        
        # 处理图像并提取特征
        with torch.no_grad():
            inputs1 = self.dino_processor(images=image1_pil, return_tensors="pt").to(self.device)
            inputs2 = self.dino_processor(images=image2_pil, return_tensors="pt").to(self.device)
            
            outputs1 = self.dino_model(**inputs1)
            outputs2 = self.dino_model(**inputs2)
        
        # 提取并平均特征
        features1 = outputs1.last_hidden_state.mean(dim=1)
        features2 = outputs2.last_hidden_state.mean(dim=1)
        
        # 计算余弦相似度并归一化
        sim = torch.nn.functional.cosine_similarity(features1, features2, dim=1).item()
        return (sim + 1) / 2  # 归一化到[0,1]
    
    def clip_scores(self, image, txt):
        # 定义通用预处理流程（[-1,1] -> [0,255] uint8）
        clip_transform = transforms.Compose([
            transforms.Normalize(mean=[-1.0], std=[2.0]),      # 逆向标准化到[0,1]
            transforms.Lambda(lambda x: (x * 255).clamp(0,255)), # 转换为0-255范围
            transforms.Lambda(lambda x: x.type(torch.uint8)),   # 转换为uint8类型
        ])

        # 处理第一个图像参数（始终是Tensor）
        processed_img = clip_transform(image).to(self.device)

        # 根据第二个参数类型动态处理
        if isinstance(txt, torch.Tensor):  # 输入是图像对
            processed_txt = clip_transform(txt).to(self.device)
        else:                              # 输入是文本
            processed_txt = txt  # 保持字符串原样

        # 统一计算CLIP分数
        score = self.clip_metric_calculator(processed_img, processed_txt)
        return score.cpu().item()
        
    def mse_scores(self, image1, image2):

        score =  self.mse_metric_calculator(image1.contiguous(),image2.contiguous())
        score = score.cpu().item()
        return score

    def psnr_scores(self, image1, image2):

        score = self.psnr_metric_calculator(image1,image2)
        score = score.cpu().item()
        
        return score

    
    def lpips_scores(self, image1, image2):

        score =  self.lpips_metric_calculator(image1,image2)
        score = score.cpu().item()
        
        return score
    
    def ssim_scores(self, image1, image2):
        score = self.ssim_metric_calculator(image1,image2)
        score = score.cpu().item()
        
        return score