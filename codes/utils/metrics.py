import torch
from PIL import Image
from torchvision import transforms
import torchmetrics
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.regression import MeanSquaredError

class metircs:  #输入图像值范围均为[-1,1]
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 初始化CLIP评分指标，用于测量图像与文本的相似度
        self.clip_metric_calculator = CLIPScore(model_name_or_path="/data/chx/clip-vit-base-patch32").to(self.device)
        
        # 初始化MSE评分指标，用于测量像素级别的差异
        self.mse_metric_calculator = MeanSquaredError().to(self.device)
        
        # 初始化PSNR评分指标，用于测量图像质量
        self.psnr_metric_calculator = PeakSignalNoiseRatio(data_range=2.0).to(self.device)
        
        # 初始化LPIPS评分指标，用于测量感知相似度
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(self.device)
        

    def clip_scores(self,  image, txt):
        # 逆向标准化 + 恢复像素范围
        clip_transform = transforms.Compose([
            # 逆向标准化: [-1,1] → [0,1]
            transforms.Normalize(mean=[-1.0], std=[2.0]),
            # 转换为 [0,255] 并调整维度顺序
            transforms.Lambda(lambda x: (x * 255).type(torch.uint8)),
        ])

        image=clip_transform(image).to(self.device)

        score = self.clip_metric_calculator(image, txt)
        score = score.cpu().item()
        return score
    
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