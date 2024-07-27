import os
import torch
import argparse
from torch.utils.data import Dataset
import json
from tqdm import tqdm 
from PIL import Image
from torch.utils.data.dataloader import default_collate
import torch
from sam.models import load_model_and_preprocess

class MDataset(Dataset):
    def __init__(
        self, vis_processor, annoation, vis_root
    ):
        self.vis_root = vis_root
        self.annotation = annoation
        self.vis_processor = vis_processor
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        images = []
        ann = self.annotation[index]

        for p in ann['img_list']:
            image = Image.open(os.path.join(self.vis_root, p)).convert("RGB")
            image = self.vis_processor(image)
            images.append(image)
        images = torch.stack(images, 1)
        return {
            "sample_id": ann['id'],
            "image": images,
            "prompt": str(ann['instruction']),
            "response": str(ann['output'])
        }        
    def collater(self, samples):
        return default_collate(samples)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', '-d', type=str)
parser.add_argument('--save_dir', '-s', type=str, default='result')

args = parser.parse_args()

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
model, vis_processors, _ = load_model_and_preprocess(name='sam', model_type='sam', is_eval=True, device=device)
model = model.to(device)

for dataset in ['AESOP', 'VIST', 'DM800K', 'Conceptual', 'Animal', 'Vehicle']:
    dataset_dir = os.path.join(args.data_dir, dataset)
    with open(os.path.join(dataset_dir, 'annotations.json')) as f:
        ann = json.load(f)
    E = MDataset(vis_processors['eval'], ann, dataset_dir)
    data_loader = torch.utils.data.DataLoader(dataset=E, batch_size=int(20/E[0]['image'].size(1)),shuffle=False,num_workers = 0)

    preds = []
    for i, samples in enumerate(tqdm(data_loader)):
        samples['image'] = samples['image'].to(device)
        pred_responses = model.generate({"image": samples['image'], "prompt": samples['prompt']})
        for sid, gt, p in zip(samples['sample_id'],samples['response'],pred_responses):
            preds.append({'sample_id':sid.item(),'pred_response':p, 'gt_response':gt})
        
        output_dir = os.path.join(args.save_dir, dataset)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir,'pred.json'),'w',encoding='utf8') as f:
            json.dump(preds,f,indent=4,ensure_ascii=False)