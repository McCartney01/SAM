import torch
from PIL import Image
from sam.models import load_model_and_preprocess

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
model, vis_processors, _ = load_model_and_preprocess(name='sam', model_type='sam', is_eval=True, device=device)
model = model.to(device)

def gen(img_path, prompt):
    raw_image = [Image.open(p).convert("RGB") for p in img_path]
    image = [vis_processors["eval"](r).unsqueeze(1) for r in raw_image]
    image = torch.cat(image, dim=1).unsqueeze(0).to(device)
    output = model.generate({"image": image, "prompt": [prompt]})
    print(output)

img_path = ['images/1.png','images/2.png','images/3.png','images/4.png','images/5.png']
prompt = "Sum up the common factor in these five pictures in a single statement."
gen(img_path, prompt)

img_path = ['images/6.jpg','images/7.jpg','images/8.jpg']
prompt = "With the narratives paired with the initial images, how would you conclude the story using the last picture?{image#1}Caption#1:Even though they had been advised against it, Lizzy and Bobby went for a bike ride on the beach.{image#2}Caption#2:They were doing fine until they came upon some sand dunes, and Lizzy lost her balance and fell.{image#3}Caption#3:"
gen(img_path, prompt)