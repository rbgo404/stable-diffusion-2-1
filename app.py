from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO
import base64


class InferlessPythonModel:
    def initialize(self):
        print("New Changes are comming")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
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
        return { "label_1" : 0.398 , "label_2" : 0.563, "label_3" : 0.434 }
        
    def finalize(self):
        self.pipe = None
