from img2dataset import download
import os        
import webdataset as wds
from datasets import load_dataset
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


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

        subset_ids = np.array([i for i in range(len(self.all_prompts)) if method(substring in self.all_prompts[i] for substring in filter_list)])
        if len(subset_ids) <= 0:
            print("no filter matches found")
            return [], []
        
        # create a random batch
        batch_idcs = self._get_random_subsample(len(subset_ids))
        subset_ids = subset_ids[batch_idcs]
        return self.all_images[subset_ids], self.all_prompts[subset_ids]


class MSCOCO_Dataset(DatasetInterface):
    name = 'MSCOCO'

    def __init__(self, path, seed=31415, batch_size = 100):
        super().__init__(path, seed, batch_size)
        self.output_name = 'bench'

        # self.download_dataset()

        # https://webdataset.github.io/webdataset/gettingstarted/
        url = "file:" + self.path + self.output_name + "/{00000..00591}.tar" # http://storage.googleapis.com/nvdata-openimages/openimages-train-000000.tar"
        dataset = wds.WebDataset(url).shuffle(1000).decode("pil").rename(image="jpg;png;jpeg;webp", text="txt", json="json").to_tuple("image", "text").batched(1000)#.map_dict(image=preprocess, text=lambda text: clip.tokenize(text, truncate=True)[0], json=lambda json: json).to_tuple("image", "text", "json")
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=None)
        self.all_images, self.all_prompts = next(iter(dataloader))
        self.all_images = np.array(self.all_images)
        self.all_prompts = np.array(self.all_prompts)
        
    def download_dataset(self):
        # TODO: check if this works
        output_dir = os.path.abspath(self.path + self.output_name)

        download(
            processes_count=16,
            thread_count=32,
            url_list=self.path+"mscoco.parquet",
            image_size=256,
            output_folder=output_dir,
            output_format="webdataset",
            input_format="parquet",
            url_col="URL",
            caption_col="TEXT",
            enable_wandb=True,
            number_sample_per_shard=1000,
            distributor="multiprocessing",
        )
    


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

    def __init__(self, image, prompt, batch_size=100) -> None:
        super().__init__("", None, None) # set batch_size to none to prevent randomization

        angle = 360/batch_size
        images = []
        for i in range(batch_size):
            images.append(image.rotate(angle*i))
        
        self.all_images = np.array(images)
        self.all_prompts = np.array([prompt]*batch_size)


class Noise_Dataset(DatasetInterface):
    name='Noisy'

    def __init__(self, image, prompt, seed=31415, batch_size=100) -> None:
        super().__init__('', seed, None)

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

