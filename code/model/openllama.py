import numpy as np
import torch.nn as nn
import os
import logging
import random
from PIL import Image
import torchvision.transforms as transforms

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch.nn.functional as F
from .ImageBind import *
from .ImageBind import data
from .AnomalyGPT_models import LinearLayer, Adapter, MMCI
from utils.loss import FocalLoss, BinaryDiceLoss
import kornia as K

import torch
from .siamese_model_conf_gnn import GNNNet

CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper', 'object',
               'candle', 'cashew', 'chewinggum', 'fryum', 'macaroni', 'pcb', 'pipe fryum']

prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect',
                 '{} without damage']
prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']

prompt_state = [prompt_normal, prompt_abnormal]

prompt_templates = ['a photo of a {}.', 'a photo of the {}.']
# prompt_templates = ['a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.',
#                     'a bright photo of a {}.', 'a dark photo of the {}.', 'a photo of my {}.', 'a photo of the cool {}.',
#                     'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.',
#                     'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.',
#                     'a photo of one {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'a low resolution photo of a {}.',
#                     'a photo of a large {}.', 'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.',
#                     'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'a dark photo of a {}.',
#                     'a photo of a cool {}.', 'a photo of a small {}.', 'there is a {} in the scene.', 'there is the {} in the scene.',
#                     'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']
#
objs = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal nut', 'pill', 'screw', 'tile',
        'toothbrush', 'transistor', 'wood', 'zipper', 'object',
        'candle', 'cashew', 'chewinggum', 'fryum', 'macaroni', 'pcb', 'pipe fryum', 'macaroni1', 'macaroni2', 'pcb1',
        'pcb2', 'pcb3', 'pcb4', 'capsules']

prompt_sentences = {}

for obj in objs:
    prompt_sentence_obj = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(obj) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = data.load_and_transform_text(prompted_sentence, torch.cuda.current_device())
        prompt_sentence_obj.append(prompted_sentence)
    prompt_sentences[obj] = prompt_sentence_obj


def encode_text_with_prompt_ensemble(model, obj, device):
    global prompt_sentences
    normal_sentences = []
    abnormal_sentences = []
    for idx in range(len(obj)):
        sentence = prompt_sentences[obj[idx].replace('_', ' ')]
        normal_sentences.append(sentence[0])
        abnormal_sentences.append(sentence[1])

    normal_sentences = torch.cat(normal_sentences).to(device)
    abnormal_sentences = torch.cat(abnormal_sentences).to(device)

    class_embeddings_normal = model({ModalityType.TEXT: normal_sentences})[ModalityType.TEXT][0]
    class_embeddings_abnormal = model({ModalityType.TEXT: abnormal_sentences})[ModalityType.TEXT][0]
    # class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

    class_embeddings_normal = class_embeddings_normal.reshape(
        (len(obj), len(prompt_templates) * len(prompt_normal), 1024))
    class_embeddings_normal = class_embeddings_normal.mean(dim=1, keepdim=True)
    class_embeddings_normal = class_embeddings_normal / class_embeddings_normal.norm(dim=-1, keepdim=True)

    class_embeddings_abnormal = class_embeddings_abnormal.reshape(
        (len(obj), len(prompt_templates) * len(prompt_abnormal), 1024))
    class_embeddings_abnormal = class_embeddings_abnormal.mean(dim=1, keepdim=True)
    class_embeddings_abnormal = class_embeddings_abnormal / class_embeddings_abnormal.norm(dim=-1, keepdim=True)

    text_features = torch.cat([class_embeddings_normal, class_embeddings_abnormal], dim=1)

    return text_features


def find_first_file_in_directory(directory_path):
    try:
        file_list = os.listdir(directory_path)
        for item in file_list:
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                return item_path
        return None

    except OSError as e:
        print(f"Error while accessing directory: {e}")
        return None


class OpenLLAMAPEFTModel(nn.Module):

    def __init__(self, **args):
        super(OpenLLAMAPEFTModel, self).__init__()
        self.args = args
        imagebind_ckpt_path = args['imagebind_ckpt_path']
        stage = args['stage']

        print(f'Initializing visual encoder from {imagebind_ckpt_path} ...')

        self.visual_encoder, self.visual_hidden_size = imagebind_model.imagebind_huge(args)
        imagebind_ckpt = torch.load(imagebind_ckpt_path, map_location=torch.device('cpu'))
        self.visual_encoder.load_state_dict(imagebind_ckpt, strict=True)
        self.iter = 0

        self.image_decoder = LinearLayer(1280, 1024, 4)
        self.adapter = Adapter(1024, 1024)
        self.aGNN = GNNNet()
        self.mmci = MMCI()

        self.loss_focal = FocalLoss()
        self.loss_dice = BinaryDiceLoss()

        # free vision encoder
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder.eval()
        print('Visual encoder initialized.')

        self.device = torch.cuda.current_device()

    def rot90_img(self, x, k):
        # k is 0,1,2,3
        degreesarr = [0., 90., 180., 270., 360]
        degrees = torch.tensor(degreesarr[k]).to(torch.float32).to(self.device)
        x = K.geometry.transform.rotate(x, angle=degrees, padding_mode='reflection')
        return x

    def encode_image(self, image_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(torch.float32) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'][0]  # bsz x 1024
            patch_features = embeddings['vision'][1]  # bsz x h*w x 1280

        image_embeds = self.adapter(image_embeds)
        patch_tokens = self.image_decoder(patch_features)  # bsz x h*w x 1024

        return image_embeds, patch_tokens

    def encode_image_for_one_shot(self, image_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(torch.float32) for key in inputs}
        for key in inputs:
            images = inputs[key]
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            patch_features = embeddings['vision'][1]  # bsz x h*w x 1280
            for i in range(len(patch_features)):
                # print("patch_features[i] shape",patch_features[i].shape)#[257,B,1280]
                patch_features[i] = patch_features[i].transpose(0, 1)[:, 1:, :]
                # print("after", patch_features[i].shape)#[B,256,1280]
        return patch_features, images

    def encode_image_for_one_shot_from_tensor(self, image_tensors):
        if not isinstance(image_tensors, list):
            image_tensors = [image_tensors]
        inputs = {ModalityType.VISION: torch.stack(image_tensors, dim=0).to(self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(torch.float32) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            patch_features = embeddings['vision'][1]  # bsz x h*w x 1280
            for i in range(len(patch_features)):
                patch_features[i] = patch_features[i].transpose(0, 1)[:, 1:, :]

        return patch_features

    def encode_image_for_one_shot_with_aug(self, image_paths):
        image_tensors = data.load_and_transform_vision_data(image_paths, self.device).to(torch.float32)
        B, C, H, W = image_tensors.shape
        # print(B,C,H,W)

        rotated_images = torch.zeros((4, B, C, H, W)).to(torch.float32).to(self.device)

        for j, degree in enumerate([0, 1, 2, 3]):
            rotated_img = self.rot90_img(image_tensors, degree)
            # 存储旋转后的图像
            rotated_images[j] = rotated_img

        image_tensors = rotated_images.transpose(0, 1).reshape(B * 4, C, H, W)

        inputs = {ModalityType.VISION: image_tensors}
        # convert into visual dtype
        inputs = {key: inputs[key] for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            patch_features = embeddings['vision'][1]  # bsz x h*w x 1280
            for i in range(len(patch_features)):
                patch_features[i] = patch_features[i].transpose(0, 1)[:, 1:, :].reshape(B, 4, 256, 1280).reshape(B,
                                                                                                                 4 * 256,
                                                                                                                 1280)

        return patch_features

    def encode_image_from_tensor(self, image_tensors):
        if not isinstance(image_tensors, list):
            image_tensors = [image_tensors]
        inputs = {ModalityType.VISION: torch.stack(image_tensors, dim=0).to(self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(torch.float32) for key in inputs}

        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'][0]  # bsz x 1024
            # print("image_embeds",image_embeds.shape)
            patch_features = embeddings['vision'][1]  # bsz x h*w x 1024

        image_embeds = self.adapter(image_embeds)
        patch_tokens = self.image_decoder(patch_features)

        return image_embeds, patch_tokens

    

    def extract_multimodal_feature(self, inputs, web_demo):
        if inputs['image_paths']:
            prompt = inputs['prompt']
            c_name = 'object'
            for name in CLASS_NAMES:
                if name in prompt:
                    c_name = name
                    break

            if not web_demo:
                image_embeds, patch_tokens = self.encode_image(inputs['image_paths'])
                feats_text_tensor = encode_text_with_prompt_ensemble(self.visual_encoder, [c_name], self.device)


            B = patch_tokens[0].size(0)
            patch_tokens = [patch_tokens[i].transpose(1, 2).view(B, 1024, 16, 16) for i in range(4)]
            patch_tokens = self.mmci(patch_tokens)
            outputs = self.aGNN(*patch_tokens)
            outputs = list(outputs)
            patch_tokens = outputs + patch_tokens

            anomaly_maps = []
            for layer in range(len(patch_tokens)):
                patch_tokens[layer] = patch_tokens[layer].view(patch_tokens[layer].size(0), patch_tokens[layer].size(1),
                                                               -1).permute(0, 2, 1)
                patch_tokens[layer] = patch_tokens[layer] / patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * patch_tokens[layer] @ feats_text_tensor.transpose(-2, -1))
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                            size=224, mode='bilinear', align_corners=True)
                anomaly_map = torch.softmax(anomaly_map, dim=1)
                anomaly_maps.append(anomaly_map[:, 1, :, :])

            anomaly_map_ret = torch.mean(torch.stack(anomaly_maps, dim=0), dim=0).unsqueeze(1)
            if inputs['normal_img_paths']:
                query_patch_tokens, _ = self.encode_image_for_one_shot(inputs['image_paths'])
                if 'mvtec' in 'normal_img_paths':
                    normal_patch_tokens = self.encode_image_for_one_shot_with_aug(inputs['normal_img_paths'])

                else:
                    normal_patch_tokens, normal_images = self.encode_image_for_one_shot(inputs['normal_img_paths'])

                sims = []

                for i in range(len(query_patch_tokens)):
                    query_patch_tokens_reshaped = query_patch_tokens[i].view(256, 1, 1280)  # [256, 1, 1280]
                    normal_tokens_reshaped = normal_patch_tokens[i].reshape(1, -1, 1280)  # [1, 256, 1280]
                    cosine_similarity_matrix = F.cosine_similarity(query_patch_tokens_reshaped, normal_tokens_reshaped,
                                                                   dim=2)
                    sim_max, _ = torch.max(cosine_similarity_matrix, dim=1)
                    sims.append(sim_max)

                sim = torch.mean(torch.stack(sims, dim=0), dim=0).reshape(1, 1, 16, 16)
                sim = F.interpolate(sim, size=224, mode='bilinear', align_corners=True)
                anomaly_map = 1 - sim  # (anomaly_map_ret + 1 - sim) / 2
                anomaly_map = torch.cat([sim, anomaly_map], dim=1)
                anomaly_map = torch.softmax(anomaly_map, dim=1)
                r = inputs['r']
                anomaly_map_ret = r * anomaly_map_ret + (1 - r) * anomaly_map[:, 1, :, :]

            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            image_map = image_embeds.unsqueeze(1) @ feats_text_tensor.transpose(-2, -1)  # [B,1,2]
            global image_score
            image_score = image_map[0, 0, 1]

        return anomaly_map_ret, image_score

    def generate(self, inputs, web_demo=False):
        pixel_output, image_score = self.extract_multimodal_feature(inputs, web_demo)
        return pixel_output, image_score
