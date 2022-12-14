import sys
from pathlib import Path
import json
import torch
from PIL import Image
import pyrootutils
from transformers import CLIPProcessor, CLIPModel, CLIPTextModel, CLIPVisionModel
from typing import Dict, List, Tuple, Optional, Any
from tqdm.auto import tqdm
import pandas as pd

root = pyrootutils.setup_root(".", pythonpath=True, cwd=False)

from visualsem_dataset_nodes import VisualSemNodesDataset



def compute_node_embeddings(vs: VisualSemNodesDataset,
                            clip_processor: CLIPProcessor,
                            clip_model: CLIPModel,
                            approaches: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
    
    data = {a: dict() for a in approaches}
    
    for node_idx in tqdm(range(len(vs))):
        node_bnid = vs.bnids[node_idx]
        node = vs[node_idx]
        
        glosses = node['glosses']['en']
        images = [Image.open(img) for img in vs.get_node_images_by_bnid(node_bnid)]
        

        with torch.no_grad():
            # compute avg gloss embs
            gloss_prep = processor.tokenizer.batch_encode_plus(glosses,
                                                         add_special_tokens=True,
                                                         padding=True,
                                                         truncation=True, 
                                                         max_length=77,
                                                         return_tensors="pt").to("cuda:0")
            gloss_feats = model.get_text_features(input_ids=gloss_prep["input_ids"],
                                            attention_mask=gloss_prep["attention_mask"],
                                            output_attentions=False,
                                            output_hidden_states=False)
            avg_glosses_emb = torch.mean(gloss_feats, dim=0)
            data[approaches[0]][node_bnid] = avg_glosses_emb.cpu()

            
            # compute avg image embs
            img_prep = processor.feature_extractor(images, return_tensors="pt").to("cuda:0")
            img_feats = model.get_image_features(pixel_values=img_prep["pixel_values"],
                                                 output_attentions=False,
                                                 output_hidden_states=False)
            avg_images_emb = torch.mean(img_feats, dim=0)
            data[approaches[1]][node_bnid] = avg_images_emb.cpu()
            
            #compute avg glosses and images feats
            avg_glosses_and_images_emb = torch.mean(torch.concat((gloss_feats, img_feats)), dim=0)
            data[approaches[2]][node_bnid] = avg_glosses_and_images_emb.cpu()


    return data


if __name__ == '__main__':
    nodes_json   = root / "dataset" / "nodes.v2.json"
    glosses_json = root / "dataset" / "gloss_files" / "nodes.glosses.json"
    tuples_json  = root / "dataset" / "tuples.v2.json"
    path_to_images  = root / "dataset" / "images"

    assert nodes_json.exists()
    assert glosses_json.exists()
    assert tuples_json.exists()
    assert path_to_images.exists()

    vs = VisualSemNodesDataset(nodes_json, glosses_json, tuples_json, path_to_images)

    clip_model = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"

    processor = CLIPProcessor.from_pretrained(clip_model)
    model = CLIPModel.from_pretrained(clip_model)

    model.to("cuda:0")

    approaches = ["avg_glosses", "avg_images", "avg_glosses_and_images"]

    data = compute_node_embeddings(vs, processor, model, approaches)
    for approach in approaches:
        out = root / 'dataset' / 'embs' / f'{clip_model}_{approach}.pt'
        if not out.parent.exists():
            out.parent.mkdir(parents=True)

        torch.save(data[approach], out)
