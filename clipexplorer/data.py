# from img2dataset import download
import os        
import webdataset as wds
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import COCO
import requests
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib


class DatasetInterface:
    name = 'DatasetInterface'

    def __init__(self, path, seed=31415, batch_size = 100) -> None:
        self.path = path
        self.seed = seed
        self.batch_size = batch_size
        pass

    def get_data(self):
        
        if self.batch_size is None:
            return self.all_images, self.all_prompts

        # create a random batch
        batch_idcs = self._get_random_subsample(len(self.all_images))

        return self.all_images[batch_idcs], self.all_prompts[batch_idcs]

    def _get_random_subsample(self, arr_len):
        if self.batch_size is None:
            self.batch_size = arr_len
        np.random.seed(self.seed)
        arr = np.random.rand(arr_len) 
        sorted_indices = arr.argsort()
        subsample_idcs = sorted_indices[:min(self.batch_size, len(sorted_indices))]
        return subsample_idcs
    
    def get_filtered_data(self, filter_list, method=any):
        # filter_list: a list of strings that are used for filtering
        # method: any -> any substring given in filter_list is present; all -> all substrings must be contained in the string
        if filter_list is None or len(filter_list) <= 0:
            return self.get_data()

        subset_ids = np.array([i for i in range(len(self.all_prompts)) if method(substring in self.all_prompts[i].lower() for substring in filter_list)])
        if len(subset_ids) <= 0:
            print("no filter matches found")
            return [], []
        
        # create a random batch
        batch_idcs = self._get_random_subsample(len(subset_ids))
        subset_ids = subset_ids[batch_idcs]
        return self.all_images[subset_ids], self.all_prompts[subset_ids]
    


class Conceptual12M_Dataset(DatasetInterface):

    # see https://huggingface.co/datasets/conceptual_12m
    name = 'Conceptual12M'

    def __init__(self, path='', seed=31415, batch_size = 100):
        super().__init__(path, seed, batch_size)
        self.dataset = load_dataset("conceptual_12m")['train']

    def get_data(self):
        if self.batch_size is None:
            return CustomConceptual12MMapper(self.dataset, 'image'), CustomConceptual12MMapper(self.dataset, 'caption')

        # create a random batch
        batch_idcs = self._get_random_subsample(self.dataset.num_rows)
        batched_dataset = self.dataset.select(batch_idcs)

        return CustomConceptual12MMapper(batched_dataset, 'image'), CustomConceptual12MMapper(batched_dataset, 'caption')

    def get_filtered_data(self, filter_list, method=any):
        # filter_list: a list of strings that are used for filtering
        # method: any -> any substring given in filter_list is present; all -> all substrings must be contained in the string
        if filter_list is None or len(filter_list) <= 0:
            return self.get_data()

        subset = self.dataset.filter(lambda example: method(substring in example['caption'].lower() for substring in filter_list))
        # create a random batch
        batch_idcs = self._get_random_subsample(subset.num_rows)
        batched_dataset = subset.select(batch_idcs)

        return CustomConceptual12MMapper(batched_dataset, 'image'), CustomConceptual12MMapper(batched_dataset, 'caption')




class MSCOCO_Val_Dataset(DatasetInterface):
    # download validation annotations from https://cocodataset.org/#download 
    # 2017 Train/Val annotations [241MB] -> captions_val2017.json
    name = 'MSCOCO-Val'

    def __init__(self, path, seed=31415, batch_size = 100):
        super().__init__(path, seed, batch_size)

        self.annotation_file = '%s/captions_val2017.json'%(path)
        self.img_folder = '%s/%s/'%(path, self.name)

        if not os.path.exists(self.img_folder):
            os.makedirs(self.img_folder)

        # init prompts and images
        self.coco_caps = COCO(self.annotation_file)
        img_ids = list(self.coco_caps.imgs.keys())

        all_prompts = []
        all_image_infos = []
        
        for id in img_ids:
            anns = self.coco_caps.loadAnns(self.coco_caps.getAnnIds([id]))
            all_prompts.append(anns[0]['caption']) # only take the first caption out of the 5 available ones

            coco_img = self.coco_caps.loadImgs([id])[0]
            all_image_infos.append(coco_img)

        self.all_prompts = np.array(all_prompts)
        self.all_images = CustomMSCOCOValImageMapper(all_image_infos, self.img_folder)




class MSCOCO_Dataset(DatasetInterface):
    name = 'MSCOCO'

    def __init__(self, path, seed=31415, batch_size = 100):
        super().__init__(path, seed, batch_size)
        self.output_name = 'bench'

        # self.download_dataset()

        # https://webdataset.github.io/webdataset/gettingstarted/
        url = "file:" + self.path + self.output_name + "/{00000..00591}.tar" # http://storage.googleapis.com/nvdata-openimages/openimages-train-000000.tar"
        dataset = wds.WebDataset(url).shuffle(batch_size).decode("pil").rename(image="jpg;png;jpeg;webp", text="txt", json="json").to_tuple("image", "text").batched(batch_size)#.map_dict(image=preprocess, text=lambda text: clip.tokenize(text, truncate=True)[0], json=lambda json: json).to_tuple("image", "text", "json")
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=None)
        self.all_images, self.all_prompts = next(iter(dataloader))
        self.all_images = np.array(self.all_images)
        self.all_prompts = np.array(self.all_prompts)
        
    # def download_dataset(self):
    #     # TODO: check if this works
    #     output_dir = os.path.abspath(self.path + self.output_name)

    #     download(
    #         processes_count=16,
    #         thread_count=32,
    #         url_list=self.path+"mscoco.parquet",
    #         image_size=256,
    #         output_folder=output_dir,
    #         output_format="webdataset",
    #         input_format="parquet",
    #         url_col="URL",
    #         caption_col="TEXT",
    #         enable_wandb=True,
    #         number_sample_per_shard=1000,
    #         distributor="multiprocessing",
    #     )
    



class DiffusionDB_Dataset(DatasetInterface):
    name='DiffusionDB'

    def __init__(self, path, seed=31415, batch_size = 100):
        super().__init__(path, seed, batch_size)
        dataset = load_dataset('poloclub/diffusiondb', path)

        self.all_images = CustomDiffusionDBMapper(dataset["train"], "image")
        self.all_prompts = CustomDiffusionDBMapper(dataset["train"], "prompt")
    
class RandomAugmentation_Dataset(DatasetInterface):
    name='Augmented'

    def __init__(self, image, prompt, transform=transforms.Compose([transforms.RandomRotation(degrees=90)]), seed=31415, batch_size=100) -> None:
        super().__init__("", seed, batch_size)

        images = []
        for i in range(self.batch_size):
            images.append(transform(image))
        
        self.all_images = np.array(images)
        self.all_prompts = np.array([prompt]*self.batch_size)


class Rotate_Dataset(DatasetInterface):
    name='Rotated'

    def __init__(self, image, prompt, id=0, batch_size=100) -> None:
        super().__init__("", None, None) # set batch_size to none to prevent randomization
        self.name = self.name + '-' + str(id)

        angle = 360/batch_size
        images = []
        for i in range(batch_size):
            images.append(image.rotate(angle*i))
        
        self.all_images = np.array(images)
        self.all_prompts = np.array([prompt]*batch_size)


class Noise_Dataset(DatasetInterface):
    name='Noisy'

    def __init__(self, image, prompt, id=0, seed=31415, batch_size=100) -> None:
        super().__init__('', seed, None)
        self.name = self.name + '-' + str(id)

        noise_level = 1/batch_size
        image_array = np.array(image)
        images = []
        for i in range(batch_size):
            
            noise = np.random.randint(low=0, high=256, size=image_array.shape,dtype='uint8')
            noisy_image_array = np.clip((1-i*noise_level)*image_array + i*noise_level*noise, 0, 255).astype('uint8')
            noisy_image = Image.fromarray(noisy_image_array)
            images.append(noisy_image)

        self.all_images = np.array(images)
        self.all_prompts = np.array([prompt]*batch_size)
        



available_datasets = {
    'MSCOCO': MSCOCO_Dataset,
    'DiffusionDB': DiffusionDB_Dataset,
}

def get_dataset(dataset_name, path):
    assert dataset_name in available_datasets, 'choose one of ' + str(list(available_datasets))

    return available_datasets[dataset_name](path)




# ---------Custom Data Mappers---------

def fetch_single_image(image_url, timeout=None, retries=0):
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": get_datasets_user_agent()},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = Image.open(io.BytesIO(req.read()))
            break
        except Exception:
            # image = None
            image = Image.new('RGB', (100, 100))
    return image

def fetch_images(batch, num_threads=8, timeout=None, retries=0):
    fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(executor.map(fetch_single_image_with_args, batch["image_url"]))
    return batch

class CustomConceptual12MMapper:
    def __init__(self, dataset, selector=None, num_threads=8):
        # selector... defines whether to return "image", "caption", "image_url"
        self.dataset = dataset
        self.selector = selector
        self.num_threads = num_threads
        
    def __len__(self):
        return self.dataset.num_rows
        
    def __getitem__(self, indices):
        ### indices given as parameter do not directly translate to the indices of the wordnet dataframe because we load n_imgs_per_synset in this loader 
        ### virtual indices can be in the range of [0, len(self.dataframe) * self.n_imgs_per_synset - 1]
        ### -> we need to translate these virtual indices to the indices that we need for the wordnet dataframe
        if type(indices) == int:
            indices = [indices]
            
        if type(indices) == slice:
            def ifnone(val, default):
                if val is None:
                    return default
                return val
            indices = list(range(ifnone(indices.start, 0), ifnone(indices.stop, len(self)), ifnone(indices.step, 1)))
        
        indices = np.array(indices)

        subset = self.dataset.select(indices)
        if self.selector == 'image':
            subset = subset.map(fetch_images, batched=True, batch_size=100, fn_kwargs={"num_threads": self.num_threads})
        if self.selector is not None and self.selector in subset.features.keys():
            subset = subset[self.selector]

        if len(subset) == 1:
            return subset[0]
        return subset
    

    
class CustomMSCOCOValImageMapper:
    def __init__(self, img_infos, img_folder):
        self.img_infos = img_infos
        self.img_folder = img_folder
        
    def __len__(self):
        return len(self.img_infos)
        
    def __getitem__(self, indices):
        all_images = []
        for i in indices:
            coco_img = self.img_infos[i]
            if not os.path.exists(self.img_folder + coco_img['file_name']):
                print('downloading file', coco_img['file_name'])
                response = requests.get(coco_img['coco_url'])
                with open(self.img_folder + coco_img['file_name'], "wb") as file:
                    file.write(response.content)

            all_images.append(Image.open(self.img_folder + coco_img['file_name']).convert("RGB"))

        return all_images
    

class CustomDiffusionDBMapper:
    def __init__(self, dataset, selector=None):
        # selector... defines whether to return "image", "prompt", "seed", "step", "cfg", "sampler", "width", "height", "user_name", "timestamp", "image_nsfw", "prompt_nsfw", or all (None) properties
        self.dataset = dataset
        self.selector = selector
        
    def __len__(self):
        return self.dataset.num_rows
        
    def __getitem__(self, indices):
        subset = self.dataset[indices]
        if self.selector is not None and self.selector in subset:
            return subset[self.selector]
        return subset