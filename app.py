from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO
import base64
from huggingface_hub import snapshot_download
import os

class InferlessPythonModel:
    def initialize(self):
        print("Hello World 7")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "stability/stable-diffusion-2-1",
            use_safetensors=True,
            torch_dtype=torch.float16,
            device_map='auto'
        )


    def infer(self, inputs):
        prompt = inputs["prompt"]
        image = self.pipe(prompt).images[0]
        buff = BytesIO()
        image.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue()).decode()
        return { "generated_image_base64" : img_str }
        
    def finalize(self):
        self.pipe = None
