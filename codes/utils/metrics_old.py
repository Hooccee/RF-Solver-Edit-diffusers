import torch
import clip
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import lpips

class metircs:
    def __init__(self, ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # clip score
        clip_model, clip_preprocess = clip.load('ViT-B/32', device=self.device)
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        # mse score & psnr score
        self.to_tensor_transform = transforms.Compose([transforms.ToTensor()])
        self.mse_loss = nn.MSELoss()
        # lpips score
        self.loss_fn = lpips.LPIPS(net='vgg').to(self.device)

    def clip_scores(self, prompt, images):

        text_tokens = clip.tokenize(prompt).to(self.device)
        images = self.clip_preprocess(images.convert("RGB")).unsqueeze(0).to(self.device)  # 预处理图像并添加批次维度，移动到设备上

        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            text_features = self.clip_model.encode_text(text_tokens)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 归一化>图像特征   ?
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # 归一化文本特征

            similarity_scores = (image_features @ text_features.T).squeeze()  # 计算相似度得分（点积）

        return similarity_scores.mean().item()
    
    def mse_scores(self, image1, image2):
        if isinstance(image1, Image.Image):
            image1 = self.to_tensor_transform(image1).to(self.device)
        if isinstance(image2, Image.Image):
            image2 = self.to_tensor_transform(image2).to(self.device)

        mse = self.mse_loss(image1, image2).item()
        return mse

    def psnr_scores(self, image1, image2):
        max_value = 1.0
        if isinstance(image1, Image.Image):
            image1 = self.to_tensor_transform(image1).to(self.device)
        if isinstance(image2, Image.Image):
            image2 = self.to_tensor_transform(image2).to(self.device)
        
        mse = self.mse_loss(image1, image2)
        psnr = 10 * torch.log10(max_value**2 / mse).item()
        return psnr
    
    def lpips_scores(self, image1, image2):
        if isinstance(image1, Image.Image):
            image1_tensor = self.to_tensor_transform(image1)
            # 手动归一化
            mean = image1_tensor.mean(dim=(1, 2))
            std = image1_tensor.std(dim=(1, 2))
            image1_normalized = (image1_tensor - mean[:, None, None]) / std[:, None, None]
            image1 = image1_normalized.to(self.device)
        if isinstance(image2, Image.Image):
            image2_tensor = self.to_tensor_transform(image2)
            mean = image2_tensor.mean(dim=(1, 2))
            std = image2_tensor.std(dim=(1, 2))
            image2_normalized = (image2_tensor - mean[:, None, None]) / std[:, None, None]
            image2 = image2_normalized.to(self.device)
        # loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization
        loss = self.loss_fn(image1, image2).item()
        return loss