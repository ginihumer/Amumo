import clip
import open_clip
import numpy as np
import torch
import os
import json
import ..CLOOB_local.clip.clip as cloob
from CLOOB_local.clip.model import CLIPGeneral
from CLOOB_local.cloob_training import model_pt, pretrained
from torchvision import transforms

class CLIPModelInterface:
    available_models = []
    model_name = 'interface'

    def __init__(self, name, device, checkpoints_dir=None) -> None:
        assert name in self.available_models, 'choose one of ' + str(self.available_models)
        self.name=name
        self.device = device
        self.checkpoints_dir = checkpoints_dir
    
    def encode_image(self):
        """Encode a batch of images to a CLIP embedding space"""
        pass

    def encode_text(self):
        """Encode a batch of texts to a CLIP embedding space"""
        pass



class CLIPModel(CLIPModelInterface):
    available_models = clip.available_models()
    model_name = 'CLIP'

    def __init__(self, name='RN50', device='cpu', checkpoints_dir=None) -> None:
        super().__init__(name, device, checkpoints_dir)
        self.model, self.preprocess = clip.load(name, device=device)
        self.model.eval()

    def encode_image(self, images):
        images = [self.preprocess(i) for i in images]
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

    def encode_image(self, images):
        images = [self.preprocess(i) for i in images]
        image_input = torch.tensor(np.stack(images)).to(self.device)
        return self.model.encode_image(image_input).float().cpu()

    def encode_text(self, texts):
        text_tokens = self.tokenize(texts).to(self.device)
        return self.model.encode_text(text_tokens).float().cpu()



class CyCLIPModel(CLIPModel):
    available_models = ["RN50"]
    model_name = 'CyCLIP'

    def __init__(self, name='RN50', device='cpu', checkpoints_dir='checkpoints') -> None:
        super().__init__(name, device, checkpoints_dir=checkpoints_dir)
        checkpoint = self.checkpoints_dir + '/cyclip-3M.pt' # 'i-cyclip.pt', 'c-cyclip.pt'

        state_dict = torch.load(checkpoint, map_location = device)["state_dict"]
        if(next(iter(state_dict.items()))[0].startswith("module")):
            state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
        self.model.load_state_dict(state_dict)

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

    def __init__(self, name='RN50', device='cpu', checkpoints_dir='checkpoints') -> None:
        super().__init__(name, device, checkpoints_dir)

        checkpoint_path = checkpoints_dir + '/cloob_' + name.lower() + '_yfcc_epoch_28.pt' # cloob_rn50x4_yfcc_epoch_28.pt
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

    def __init__(self, name='ViT-B/16', device='cpu', checkpoints_dir=None) -> None:
        super().__init__(name, device, checkpoints_dir)
        
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
