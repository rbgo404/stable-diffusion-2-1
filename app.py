from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from io import BytesIO
import base64
import os

class InferlessPythonModel:
    def initialize(self):
        self.pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1",use_safetensors=True,torch_dtype=torch.float16).to("cuda")
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        
    def infer(self, inputs):
        prompt = inputs["prompt"]
        image = self.pipe(prompt,negative_prompt="low quality",num_inference_steps=9).images[0]
        buff = BytesIO()
        image.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue()).decode()
        
        return {"generated_image_base64" : img_str }
    
    def finalize(self):
        self.pipe = None
