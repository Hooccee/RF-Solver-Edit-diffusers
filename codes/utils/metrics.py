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
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(self.device)

        # 初始化SSIM评分指标，用于测量结构相似度
        self.ssim_metric_calculator = StructuralSimilarityIndexMeasure(data_range=2.0).to(self.device)

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