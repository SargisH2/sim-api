from fastapi import FastAPI

from pydantic import BaseModel


from transformers import AutoFeatureExtractor, AutoModel
from PIL import Image

import torchvision.transforms as T
from torch.nn import functional
from torch import no_grad

import requests


import pickle
with open('extractor.pkl', 'rb') as f:
    extractor =  pickle.load(f)

# model_ckpt = "google/vit-base-patch16-224"
# extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
hidden_dim = model.config.hidden_size

transformation_chain = T.Compose(
    [
        T.Resize(int((256 / 224) * extractor.size["height"])),
        T.CenterCrop(extractor.size["height"]),
        T.ToTensor(),
        T.Normalize(mean=extractor.image_mean, std=extractor.image_std),
    ]
)
device = "cpu"

def compute_scores(emb_one, emb_two):
    """Computes cosine similarity between two vectors."""
    scores = functional.cosine_similarity(emb_one, emb_two)
    return scores.numpy().tolist()

def get_scores_img(url1, url2):###################
    img1 = Image.open(requests.get(url1, stream=True).raw)
    img2 = Image.open(requests.get(url2, stream=True).raw)
    img1, img2 = transformation_chain(img1).unsqueeze(0), transformation_chain(img2).unsqueeze(0)
    new_batch1 = {"pixel_values": img1.to(device)}
    new_batch2 = {"pixel_values": img2.to(device)}
    with no_grad():
        emb1 = model(**new_batch1).last_hidden_state[:, 0].cpu()
        emb2 = model(**new_batch2).last_hidden_state[:, 0].cpu()
    
    return compute_scores(emb1, emb2)



class Details(BaseModel):
    url1: str
    url2: str
app = FastAPI()

@app.get('/')
def index():
    return {'message': "This is the home page of this API. Go to /apiv1/"}

@app.post('/apiv1/')
def api1(data: Details):
    return {
        'url1':  data.url1[:-5],
        'url2':  data.url2[:-5],
        'score': get_scores_img(data.url1, data.url2)
        }


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=80)