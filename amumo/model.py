
import numpy as np
import torch
import os
import json
import requests
from tqdm import tqdm

class CLIPModelInterface:
    available_models = []
    model_name = 'interface' 
    logit_scale = torch.tensor(0) # defines the temperature parameter used for scaling the contrastive loss

    def __init__(self, name, device) -> None:
        assert name in self.available_models, 'choose one of ' + str(self.available_models)
        self.name=name
        self.device = device
        self.encoding_functions = {
            "image": self.encode_image,
            "text": self.encode_text
        }
    
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
    
class CLIPModel(CLIPModelInterface):
    model_name = 'CLIP'

    def __init__(self, name='RN50', device='cpu') -> None:
        try:
            import clip
        except ImportError:
            print("To support CLIP model, please install 'clip': 'pip install git+https://github.com/openai/CLIP.git'.")
        self.available_models = clip.available_models()
        super().__init__(name, device)

        self.tokenize = clip.tokenize
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
        text_tokens = self.tokenize(texts, truncate = True).to(self.device)
        return self.model.encode_text(text_tokens).float().cpu()
        



class OpenCLIPModel(CLIPModelInterface):
    model_name = 'OpenCLIP'

    def __init__(self, name='RN50', device='cpu', dataset='openai') -> None:
        try:
            import open_clip
        except ImportError:
            print("To support OpenCLIP model, please install 'openclip': 'pip install open-clip-torch==2.20.0'.")
        self.available_models = open_clip.list_openai_models()
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
        
        try:
            from transformers import AutoProcessor, BlipModel
        except ImportError:
            print("To support BLIP model, please install 'transformers': 'pip install transformers==4.31.0'.")

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
        
        try:
            import clip
        except ImportError:
            print("To support CyCLIP model, please install 'clip': 'pip install git+https://github.com/openai/CLIP.git'.")

        self.tokenize = clip.tokenize
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
        text_tokens = self.tokenize(texts, truncate = True).to(self.device)
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

        try:
            from .CLOOB_local.clip import clip as cloob
            from .CLOOB_local.clip.model import CLIPGeneral
        except ImportError:
            print("To support CLOOB model, please install 'clip': 'pip install git+https://github.com/openai/CLIP.git'.")

        self.tokenize = cloob.tokenize
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
        text_tokens = self.tokenize(texts).to(self.device)
        return self.model.encode_text(text_tokens).float().cpu()



class CLOOB_LAION400M_Model(CLIPModelInterface):
    available_models = ['ViT-B/16']
    model_name = 'CLOOB-LAION400M'

    def __init__(self, name='ViT-B/16', device='cpu') -> None:
        super().__init__(name, device)
        try:
            from torchvision import transforms
            from .CLOOB_local.cloob_training import model_pt, pretrained
        except ImportError:
            print("To support CLOOB_LAION400M model, please install 'torchvision': 'pip install torchvision'.")
        
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



class ImageBind_Model(CLIPModelInterface):
    available_models = ['huge']
    model_name = 'ImageBind'

    def __init__(self, name='huge', device='cpu', bpe_path=".assets/bpe_simple_vocab_16e6.txt.gz") -> None:
        super().__init__(name, device)
        try:
            # pip install git+https://github.com/facebookresearch/ImageBind
            # pip install soundfile
            from imagebind.models import imagebind_model as ImageBindModel
            from imagebind import data as ImageBindData
            from torchvision import transforms
        except ImportError:
            print("To support ImageBind model, please install 'ImageBind', 'soundfile', and 'torchvision': 'pip install git+https://github.com/facebookresearch/ImageBind', 'pip install soundfile' and 'pip install torchvision'.")


        ImageBindData.BPE_PATH = bpe_path # download from: https://github.com/facebookresearch/ImageBind/tree/main/bpe

        # Instantiate model
        self.model = ImageBindModel.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(device)

        self.image_preprocess = transforms.Compose([
            transforms.Resize(
                224, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224),
        ])

        # self.logit_scale = self.model.logit_scale # TODO
        self.encoding_functions = {
            "image": self.encode_image,
            "text": self.encode_text,
            "audio": self.encode_audio,
            # "video": ...
            "thermal": self.encode_thermal,
            "depth": self.encode_depth,
            # "imu": ...
        }

    
    def encode_audio(self, audios):
        from imagebind.models.imagebind_model import ModalityType
        from imagebind import data as ib_data
        from torchvision import transforms
        
        num_mel_bins=128
        target_length=204
        sample_rate=16000
        clip_duration=2
        clips_per_video=3
        mean=-4.268
        std=9.138

        audio_outputs = []
        clip_sampler = ib_data.ConstantClipsPerVideoSampler(
            clip_duration=clip_duration, clips_per_video=clips_per_video
        )

        for waveform in audios:

            all_clips_timepoints = ib_data.get_clip_timepoints(
                clip_sampler, waveform.size(1) / sample_rate
            )
            all_clips = []
            for clip_timepoints in all_clips_timepoints:
                waveform_clip = waveform[
                    :,
                    int(clip_timepoints[0] * sample_rate) : int(
                        clip_timepoints[1] * sample_rate
                    ),
                ]
                waveform_melspec = ib_data.waveform2melspec(
                    waveform_clip, sample_rate, num_mel_bins, target_length
                )
                all_clips.append(waveform_melspec)

            normalize = transforms.Normalize(mean=mean, std=std)
            all_clips = [normalize(ac).to(self.device) for ac in all_clips]

            all_clips = torch.stack(all_clips, dim=0)
            audio_outputs.append(all_clips)

        audios = torch.tensor(np.stack(audio_outputs)).to(self.device)
    
        inputs = {
            ModalityType.AUDIO: audios,
        }
        return self.model(inputs)[ModalityType.AUDIO].float().cpu()
    
    def encode_depth(self, depths):
        from imagebind.models.imagebind_model import ModalityType
        # TODO: how to normalize?
        # normalize = transforms.Normalize(
        #             mean=(0.48145466),
        #             std=(0.26862954),
        #         )
        normalize = lambda x : x
    
        depths = [normalize(self.image_preprocess(depth)) for depth in depths]
        input = torch.tensor(np.stack(depths)).to(self.device)
    
        inputs = {
            ModalityType.DEPTH: input,
        }
        return self.model(inputs)[ModalityType.DEPTH].float().cpu()
    
    def encode_thermal(self, thermals):
        from imagebind.models.imagebind_model import ModalityType
        import transforms
        # TODO: how to normalize?
        transform_thermal = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.45),
                std=(0.26),
            )
        ])
    
        thermals = [transform_thermal(self.image_preprocess(thermal)) for thermal in thermals]
        input = torch.tensor(np.stack(thermals)).to(self.device)
    
        inputs = {
            ModalityType.THERMAL: input,
        }
        return self.model(inputs)[ModalityType.THERMAL].float().cpu()
        
    def encode_image(self, images):
        from imagebind.models.imagebind_model import ModalityType
        from torchvision import transforms
        
        transform_imgs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            )
        ])
        images = [transform_imgs(self.image_preprocess(i.convert("RGB"))) for i in images]
        image_input = torch.tensor(np.stack(images)).to(self.device)
        inputs = {
            ModalityType.VISION: image_input,
        }
        return self.model(inputs)[ModalityType.VISION].float().cpu()

    def encode_text(self, texts):
        from imagebind.models.imagebind_model import ModalityType
        from imagebind import data as ImageBindData
        text_tokens = ImageBindData.load_and_transform_text(texts, self.device)
        inputs = {
            ModalityType.TEXT: text_tokens,
        }
        return self.model(inputs)[ModalityType.TEXT].float().cpu()



available_CLIP_models = {
        'CLIP': CLIPModel,
        # 'OpenCLIP': OpenCLIPModel,
        # 'BLIP': BLIPModel,
        # 'CyCLIP': CyCLIPModel,
        # 'PyramidCLIP': PyramidCLIP,
        # 'CLOOB': CLOOB_Model,
        # 'CLOOB_LAION400M': CLOOB_LAION400M_Model
    }

def get_model(clip_name='CLIP', image_encoder_name=None, device="cpu"):
    assert clip_name in available_CLIP_models, 'choose one of ' + str(list(available_CLIP_models))
    return available_CLIP_models[clip_name](device=device)
