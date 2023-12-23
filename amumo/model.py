import clip
import open_clip
import numpy as np
import torch
import os
import json
from .CLOOB_local.clip import clip as cloob
from .CLOOB_local.clip.model import CLIPGeneral
from .CLOOB_local.cloob_training import model_pt, pretrained
from torchvision import transforms
import requests
from tqdm import tqdm
from transformers import AutoProcessor, BlipModel
import amumo.utils as ut

class CLIPModelInterface:
    available_models = []
    model_name = 'interface' 
    logit_scale = torch.tensor(0) # defines the temperature parameter used for scaling the contrastive loss

    def __init__(self, name, device) -> None:
        assert name in self.available_models, 'choose one of ' + str(self.available_models)
        self.name=name
        self.device = device
    
    def encode_image(self, images):
        """Encode a batch of images to a CLIP embedding space"""
        """images: a list of PIL images"""
        pass

    def encode_text(self, texts):
        """Encode a batch of texts to a CLIP embedding space"""
        """texts: a list of strings"""
        pass



def checkpoint_download_helper(url, name):
    checkpoint_dir = 'checkpoints/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint = checkpoint_dir + name
    if not os.path.exists(checkpoint):
        print('model checkpoint not found. downloading...')

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(checkpoint, 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True) as progress_bar:
                for data in response.iter_content(chunk_size=4096):
                    file.write(data)
                    progress_bar.update(len(data))

    return checkpoint


class PrecalculatedModel(CLIPModelInterface):
    model_name = 'precalculated'

    def __init__(self, name, dataset_name, modality1_features, modality2_features, logit_scale=torch.tensor(0)) -> None:
        # this class is a workaround for precalculated features
        # it just saves the features as cached files so that the "encode_image" and "encode_text" methods are not called
        self.available_models = [name]
        super().__init__(name, device='cpu')
        self.logit_scale = logit_scale
        self.modality1_features = modality1_features
        self.modality2_features = modality2_features
        self.process_precalculated_features(dataset_name)

    def process_precalculated_features(self, dataset_name):
        data_prefix = dataset_name + '_' + self.model_name + '_' + self.name
        data_prefix = data_prefix.replace('/', '-')
        np.savetxt(ut.data_checkpoint_dir + data_prefix + '_image-embedding.csv', self.modality1_features.cpu(),
                   delimiter=',')
        np.savetxt(ut.data_checkpoint_dir + data_prefix + '_text-embedding.csv', self.modality2_features.cpu(),
                   delimiter=',')

    def encode_image(self, images):
        raise NotImplementedError("this cannot be done for precalculated features -> use cached features")

    def encode_text(self, texts):
        raise NotImplementedError("this cannot be done for precalculated features -> use cached features")

class CLIPModel(CLIPModelInterface):
    available_models = clip.available_models()
    model_name = 'CLIP'

    def __init__(self, name='RN50', device='cpu') -> None:
        super().__init__(name, device)
        self.model, self.preprocess = clip.load(name, device=device)
        self.model.eval()
        self.logit_scale = self.model.logit_scale

    def encode_image(self, images):
        try:
            images = [self.preprocess(i) for i in images]
        except:
            print(images)
        image_input = torch.tensor(np.stack(images)).to(self.device)
        return self.model.encode_image(image_input).float().cpu()

    def encode_text(self, texts):
        text_tokens = clip.tokenize(texts, truncate = True).to(self.device)
        return self.model.encode_text(text_tokens).float().cpu()



class OpenCLIPModel(CLIPModelInterface):
    available_models = open_clip.list_openai_models()
    model_name = 'OpenCLIP'

    def __init__(self, name='RN50', device='cpu', dataset='openai') -> None:
        super().__init__(name, device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(name, pretrained=dataset, device=self.device)
        self.tokenize = open_clip.get_tokenizer(name)
        self.model.eval()
        self.logit_scale = self.model.logit_scale

    def encode_image(self, images):
        images = [self.preprocess(i) for i in images]
        image_input = torch.tensor(np.stack(images)).to(self.device)
        return self.model.encode_image(image_input).float().cpu()

    def encode_text(self, texts):
        text_tokens = self.tokenize(texts).to(self.device)
        return self.model.encode_text(text_tokens).float().cpu()


class BLIPModel(CLIPModelInterface):
    available_models = ['Vit-B'] # option for 'Vit-L'?
    model_name = 'BLIP'

    def __init__(self, name='Vit-B', device='cpu') -> None:
        super().__init__(name, device)
        self.model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        self.preprocess = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        self.model.eval()
        self.logit_scale = torch.tensor(0) # TODO: check if temperature is really 0

    def encode_image(self, images):
        input = self.preprocess(images=list(images), return_tensors="pt").to(self.device)
        return self.model.get_image_features(**input).float().cpu()

    def encode_text(self, texts):
        tokens = self.preprocess(text=list(texts), padding=True, truncation=True, return_tensors="pt").to(self.device)
        return self.model.get_text_features(**tokens).float().cpu()


class CyCLIPModel(CLIPModel):
    available_models = ["RN50"]
    model_name = 'CyCLIP'
    checkpoints = {
        'cyclip-3M.pt': 'https://drive.google.com/uc?id=1nF33F3yjtiWr3bgllBXk5Wf07Uo7Uv9G&export=download&confirm=9_s_'
    }

    def __init__(self, name='RN50', device='cpu') -> None:
        super().__init__(name, device)

        checkpoint = checkpoint_download_helper(self.checkpoints['cyclip-3M.pt'],'cyclip-3M.pt') # 'i-cyclip.pt', 'c-cyclip.pt'

        state_dict = torch.load(checkpoint, map_location = device)["state_dict"]
        if(next(iter(state_dict.items()))[0].startswith("module")):
            state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
        self.model.load_state_dict(state_dict)

        self.logit_scale = self.model.logit_scale

    def encode_image(self, images):
        images = [self.preprocess(i) for i in images]
        image_input = torch.tensor(np.stack(images)).to(self.device)
        return self.model.encode_image(image_input).float().cpu()

    def encode_text(self, texts):
        text_tokens = clip.tokenize(texts, truncate = True).to(self.device)
        return self.model.encode_text(text_tokens).float().cpu()


class PyramidCLIP(CLIPModelInterface): # TODO
    available_models = ['RN50', 'ViT-B-32', 'ViT-B-16']
    model_name = 'PyramidCLIP'

    def __init__(self, name='RN50', device='cpu', checkpoints_dir='checkpoints') -> None:
        print("not implemented")

class CLOOB_Model(CLIPModelInterface):
    available_models = ['RN50', 'RN50x4']
    model_name = 'CLOOB'
    checkpoints = {
        'cloob_rn50_yfcc_epoch_28.pt': 'https://ml.jku.at/research/CLOOB/downloads/checkpoints/cloob_rn50_yfcc_epoch_28.pt',
        'cloob_rn50x4_yfcc_epoch_28.pt': 'https://ml.jku.at/research/CLOOB/downloads/checkpoints/cloob_rn50x4_yfcc_epoch_28.pt'
    }

    def __init__(self, name='RN50', device='cpu') -> None:
        super().__init__(name, device)

        ckpt_name = 'cloob_' + name.lower() + '_yfcc_epoch_28.pt' # cloob_rn50x4_yfcc_epoch_28.pt
        checkpoint_path = checkpoint_download_helper(self.checkpoints[ckpt_name], ckpt_name)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_config_file = os.path.join(os.path.dirname(__file__), 'CLOOB_local/training/model_configs/', checkpoint['model_config_file'])

        print('Loading model from', model_config_file)
        assert os.path.exists(model_config_file), 'config file does not exist'
        with open(model_config_file, 'r') as f:
            model_info = json.load(f)
        self.model = CLIPGeneral(**model_info)
        self.model.eval()

        sd = checkpoint["state_dict"]
        sd = {k[len('module.'):]: v for k, v in sd.items()}
        if 'logit_scale_hopfield' in sd:
            sd.pop('logit_scale_hopfield', None)
        self.model.load_state_dict(sd)

        self.preprocess = cloob._transform(self.model.visual.input_resolution, is_train=False)

        self.logit_scale = self.model.logit_inv_tau


    def encode_image(self, images):
        images = [self.preprocess(i) for i in images]
        image_input = torch.tensor(np.stack(images)).to(self.device)
        return self.model.encode_image(image_input).float().cpu()

    def encode_text(self, texts):
        text_tokens = cloob.tokenize(texts).to(self.device)
        return self.model.encode_text(text_tokens).float().cpu()

class CLOOB_LAION400M_Model(CLIPModelInterface):
    available_models = ['ViT-B/16']
    model_name = 'CLOOB-LAION400M'

    def __init__(self, name='ViT-B/16', device='cpu') -> None:
        super().__init__(name, device)
        
        # ['cloob_laion_400m_vit_b_16_16_epochs', 'cloob_laion_400m_vit_b_16_32_epochs']
        config = pretrained.get_config('cloob_laion_400m_vit_b_16_16_epochs')
        self.model = model_pt.get_pt_model(config)
        checkpoint = pretrained.download_checkpoint(config)
        self.model.load_state_dict(model_pt.get_pt_params(config, checkpoint))
        self.model.eval().requires_grad_(False).to(self.device)

        # see: https://github.com/crowsonkb/cloob-training/blob/master/train.py#L215
        self.preprocess = transforms.Compose([
            transforms.Resize(self.model.config['image_encoder']['image_size'], interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(self.model.config['image_encoder']['image_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.model.config['image_encoder']['normalize']['mean'],
                            std=self.model.config['image_encoder']['normalize']['std'])
            ])
        
        self.logit_scale = torch.tensor(3.4012)

    def encode_image(self, images):
        images = [self.preprocess(i) for i in images]
        image_input = torch.tensor(np.stack(images)).to(self.device)
        return self.model.image_encoder(image_input).float().cpu()

    def encode_text(self, texts):
        text_tokens = self.model.tokenize(texts, truncate = True).to(self.device)
        return self.model.text_encoder(text_tokens).float().cpu()


available_CLIP_models = {
        'CLIP': CLIPModel,
        'OpenCLIP': OpenCLIPModel,
        # 'BLIP': BLIPModel,
        'CyCLIP': CyCLIPModel,
        # 'PyramidCLIP': PyramidCLIP,
        'CLOOB': CLOOB_Model,
        'CLOOB_LAION400M': CLOOB_LAION400M_Model
    }

def get_model(clip_name='CLIP', image_encoder_name=None, device="cpu"):
    assert clip_name in available_CLIP_models, 'choose one of ' + str(list(available_CLIP_models))
    if image_encoder_name is None:
        image_encoder_name = available_CLIP_models[clip_name].available_models[0]
    assert image_encoder_name in available_CLIP_models[clip_name].available_models, 'choose one of ' + str(available_CLIP_models[clip_name].available_models)

    return available_CLIP_models[clip_name](image_encoder_name, device=device)
