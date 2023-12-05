from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO
import base64


class InferlessPythonModel:
    def initialize(self):
        print("New Changes are comming on 5 dec")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
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
