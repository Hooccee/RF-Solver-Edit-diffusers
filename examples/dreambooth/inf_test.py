import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

def main():
    # 加载输入图片并调整大小
    image = load_image(
        "/mmu-vcg-hdd/caohaoxiang/dataset/Single_Image_Reflection_Removal/OpenRR-1k/test_100/blended/test_0012.jpg"
        ).resize((1168, 880))
    image.save("test_0012.png")

    # 加载 pipeline
    pipeline = FluxKontextPipeline.from_pretrained(
        "/mmu-vcg-hdd/caohaoxiang/model/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16
    ).to('cuda')

    # 加载 LoRA 权重
    pipeline.load_lora_weights(
        "/mmu-vcg/caohaoxiang/mySrc/FLUX.1-Kontext/diffusers/examples/dreambooth/openrr1k_hf-i2i_RR4K_openrr/checkpoint-1200",
        adapter_name="lora"
    )
    pipeline.set_adapters(["lora"], adapter_weights=[1])

    # 推理
    result = pipeline(
        image=image,
        # prompt="reflection removal",
        prompt="reflection removal",
        height=880,
        width=1168,
        num_inference_steps=24,
        guidance_scale=1,
    ).images[0]
    result.save("test_0012_val_1200.png")
    print("推理完成，结果已保存为 test_0012_val_1200.png")

if __name__ == "__main__":
    main()