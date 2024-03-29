# from img2dataset import download
import os
import glob
import webdataset as wds
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from pycocotools.coco import COCO
import requests
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import urllib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFMCS
import io

def reshape_image(arr):
    c, h, w = arr.shape
    reshaped_image = np.empty((h, w, c))

    reshaped_image[:, :, 0] = arr[0]
    reshaped_image[:, :, 1] = arr[1]
    reshaped_image[:, :, 2] = arr[2]

    reshaped_pil = Image.fromarray(reshaped_image.astype("uint8"))

    return reshaped_pil


def cellpainting_to_rgb(arr):
    arr0 = arr[:, :, 0].astype(np.float32)
    arr3 = arr[:, :, 3].astype(np.float32)
    arr4 = arr[:, :, 4].astype(np.float32)

    rgb_arr = np.dstack((arr0, arr3, arr4))
    
    image = Image.fromarray(rgb_arr.astype("uint8"))
    
    return image

# ---------Data Types (Modalities)---------

class DataTypeInterface:
    name = "DataTypeInterface"

    def __init__(self, data) -> None:
        self.data = data

    def getMinSummary(self, ids):
        # returns a minimal summary of a set of instances; e.g. text: top 2 n-grams
        return None

    def getSummary(self, ids):
        # returns a summary of a set of instances; e.g. text: all n-grams with count; image: n prototypes
        return None

    def getVisItem(self, id):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, indices):
        # returns the selected item(s)
        return self.data[indices]


class ImageType(DataTypeInterface):
    name = "Image"

    def __init__(self, data) -> None:
        super().__init__(data)

    def getVisItem(self, idx):
        output_img = io.BytesIO()
        self.data[idx].resize((300, 300)).save(output_img, format='JPEG')
        return output_img


class TextType(DataTypeInterface):
    name = "Text"

    def __init__(self, data) -> None:
        super().__init__(data)

    def getMinSummary(self, ids):
        # retrieve top 2 n-grams
        return get_textual_label_for_cluster(ids, self.data)

    def getVisItem(self, idx):
        return self.data[idx]


class MoleculeType(TextType):
    name = "Molecule"

    def __init__(self, data) -> None:
        # data is a list of SMILES
        super().__init__(data)

    def getMinSummary(self, ids):
        # retrieve MCS of mols
        if len(ids) == 1:
            return self.data[ids[0]]

        mols = [Chem.MolFromSmiles(smiles) for smiles in self.data[ids]]
        mcs = rdFMCS.FindMCS(mols)
        mcs_smiles = Chem.MolToSmiles(Chem.MolFromSmarts(mcs.smartsString))
        return mcs_smiles

    def getVisItem(self, idx):
        output_img = io.BytesIO()
        img = Chem.Draw.MolToImage(Chem.MolFromSmiles(self.data[idx]))
        img.resize((300, 300)).save(output_img, format='JPEG')
        return output_img


class BioImageType(ImageType):
    name = "Bio Image"

    def __init__(self, data) -> None:
        super().__init__(data)


# ---------Datasets---------
class DatasetInterface:
    name = 'DatasetInterface'
    MODE1_Type = ImageType
    MODE2_Type = TextType

    def __init__(self, path, seed=31415, batch_size=100) -> None:
        self.path = path
        self.seed = seed
        self.batch_size = batch_size

    def get_data(self):
        dataset_name = self.name + '_size-%i' % len(self.all_images)

        if self.batch_size is None:
            return self.all_images, self.all_prompts, dataset_name

        # create a random batch
        batch_idcs = self._get_random_subsample(len(self.all_images))

        images = self.MODE1_Type(self.all_images[batch_idcs])
        texts = self.MODE2_Type(self.all_prompts[batch_idcs])

        return images, texts, dataset_name

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

        subset_ids = np.array([i for i in range(len(self.all_prompts)) if
                               method(substring in self.all_prompts[i].lower() for substring in filter_list)])
        if len(subset_ids) <= 0:
            print("no filter matches found")
            return [], []

        # create a random batch
        batch_idcs = self._get_random_subsample(len(subset_ids))
        subset_ids = subset_ids[batch_idcs]

        images = self.MODE1_Type(self.all_images[subset_ids])
        texts = self.MODE2_Type(self.all_prompts[subset_ids])
        dataset_name = self.name + '_filter-' + method.__name__ + '_' + '-'.join(filter_list)

        return images, texts, dataset_name


class CLOOMDataset_Dataset(DatasetInterface):
    name = 'CLOOMDataset'

    def __init__(self, path, seed=31415, batch_size=100):
        super().__init__(path, seed, batch_size)

        self.MODE1_Type = BioImageType
        self.MODE2_Type = MoleculeType

        mol_index_file = '/.../cellpainting-split-test-imgpermol.csv'
        img_index_file = mol_index_file

        # molecule smiles
        all_molecules = pd.read_csv(mol_index_file)
        all_molecules.rename(columns={"SAMPLE_KEY": "SAMPLE_KEY_mol"}, inplace=True)
        # microscopy images
        all_microscopies = pd.read_csv(img_index_file)
        all_microscopies.rename(columns={"SAMPLE_KEY": "SAMPLE_KEY_img"}, inplace=True)
        # join the two dataframes
        cloome_data = pd.merge(all_molecules[["SAMPLE_KEY_mol", "SMILES"]],
                               all_microscopies[["SAMPLE_KEY_img", "SMILES"]], on="SMILES", how="inner")

        # subsample data
        self.subset_idcs = self._get_random_subsample(len(cloome_data))
        self.dataset = cloome_data.iloc[self.subset_idcs]

        self.all_prompts = self.dataset["SMILES"].values

        # microscopy images TODO... load images on demand with a custom image loader
        all_microscopies = pd.read_csv(img_index_file)

        all_images = []
        for img_id in self.dataset["SAMPLE_KEY_img"]:

            image = os.path.join('/.../cellpainting/npzs/chembl24/', f'{img_id}.npz')
            image = np.load(image, allow_pickle=True)['sample']
            image = cellpainting_to_rgb(image)

            all_images.append(image)

        self.all_images = np.empty(len(all_images), dtype=object)
        self.all_images[:] = all_images


class Conceptual12M_Dataset(DatasetInterface):
    # see https://huggingface.co/datasets/conceptual_12m
    name = 'Conceptual12M'

    def __init__(self, path='', seed=31415, batch_size=100):
        super().__init__(path, seed, batch_size)
        self.dataset = load_dataset("conceptual_12m")['train']

    def get_data(self):
        if self.batch_size is None:
            return CustomConceptual12MMapper(self.dataset, 'image'), CustomConceptual12MMapper(self.dataset, 'caption')

        # create a random batch
        batch_idcs = self._get_random_subsample(self.dataset.num_rows)
        batched_dataset = self.dataset.select(batch_idcs)

        images = ImageType(CustomConceptual12MMapper(batched_dataset, 'image'))
        texts = TextType(CustomConceptual12MMapper(batched_dataset, 'caption'))
        return images, texts

    def get_filtered_data(self, filter_list, method=any):
        # filter_list: a list of strings that are used for filtering
        # method: any -> any substring given in filter_list is present; all -> all substrings must be contained in the string
        if filter_list is None or len(filter_list) <= 0:
            return self.get_data()

        subset = self.dataset.filter(
            lambda example: method(substring in example['caption'].lower() for substring in filter_list))
        # create a random batch
        batch_idcs = self._get_random_subsample(subset.num_rows)
        batched_dataset = subset.select(batch_idcs)

        images = ImageType(CustomConceptual12MMapper(batched_dataset, 'image'))
        texts = TextType(CustomConceptual12MMapper(batched_dataset, 'caption'))
        return images, texts


class MSCOCO_Val_Dataset(DatasetInterface):
    # download validation annotations from https://cocodataset.org/#download 
    # 2017 Train/Val annotations [241MB] -> captions_val2017.json
    name = 'MSCOCO-Val'

    def __init__(self, path, seed=31415, batch_size=100):
        super().__init__(path, seed, batch_size)

        self.annotation_file = '%s/captions_val2017.json' % (path)
        self.img_folder = '%s/%s/' % (path, self.name)

        if not os.path.exists(self.img_folder):
            os.makedirs(self.img_folder)

        # init prompts and images
        self.coco_caps = COCO(self.annotation_file)
        img_ids = list(self.coco_caps.imgs.keys())

        all_prompts = []
        all_image_infos = []

        for id in img_ids:
            anns = self.coco_caps.loadAnns(self.coco_caps.getAnnIds([id]))
            all_prompts.append(anns[0]['caption'])  # only take the first caption out of the 5 available ones

            coco_img = self.coco_caps.loadImgs([id])[0]
            all_image_infos.append(coco_img)

        self.all_prompts = np.array(all_prompts)
        self.all_images = CustomMSCOCOValImageMapper(all_image_infos, self.img_folder)


class MSCOCO_Dataset(DatasetInterface):
    name = 'MSCOCO'

    def __init__(self, path, seed=31415, batch_size=100):
        super().__init__(path, seed, batch_size)
        self.output_name = 'bench'

        # self.download_dataset()

        # https://webdataset.github.io/webdataset/gettingstarted/
        url = "file:" + self.path + self.output_name + "/{00000..00591}.tar"  # http://storage.googleapis.com/nvdata-openimages/openimages-train-000000.tar"
        dataset = wds.WebDataset(url).shuffle(batch_size).decode("pil").rename(image="jpg;png;jpeg;webp", text="txt",
                                                                               json="json").to_tuple("image",
                                                                                                     "text").batched(
            batch_size)  # .map_dict(image=preprocess, text=lambda text: clip.tokenize(text, truncate=True)[0], json=lambda json: json).to_tuple("image", "text", "json")
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
    name = 'DiffusionDB'

    def __init__(self, path, seed=31415, batch_size=100):
        super().__init__(path, seed, batch_size)
        dataset = load_dataset('poloclub/diffusiondb', path)

        self.all_images = CustomDiffusionDBMapper(dataset["train"], "image")
        self.all_prompts = CustomDiffusionDBMapper(dataset["train"], "prompt")


class RandomAugmentation_Dataset(DatasetInterface):
    name = 'Augmented'

    def __init__(self, image, prompt, transform=transforms.Compose([transforms.RandomRotation(degrees=90)]), seed=31415,
                 batch_size=100) -> None:
        super().__init__("", seed, batch_size)

        images = []
        for i in range(self.batch_size):
            images.append(transform(image))

        self.all_images = np.array(images)
        self.all_prompts = np.array([prompt] * self.batch_size)


class Rotate_Dataset(DatasetInterface):
    name = 'Rotated'

    def __init__(self, image, prompt, id=0, batch_size=100) -> None:
        super().__init__("", None, None)  # set batch_size to none to prevent randomization
        self.name = self.name + '-' + str(id)

        angle = 360 / batch_size
        images = []
        for i in range(batch_size):
            images.append(image.rotate(angle * i))

        self.all_images = np.array(images)
        self.all_prompts = np.array([prompt] * batch_size)


class Blur_Dataset(DatasetInterface):
    name = 'Blurred'

    def __init__(self, image, prompt, id=0, batch_size=100) -> None:
        super().__init__("", None, None)  # set batch_size to none to prevent randomization
        self.name = self.name + '-' + str(id)

        max_radius = image.width / 8
        blur_radius = max_radius / batch_size
        images = []
        for i in range(batch_size):
            images.append(image.filter(ImageFilter.GaussianBlur(radius=blur_radius * i)))

        self.all_images = np.array(images)
        self.all_prompts = np.array([prompt] * batch_size)


class HShift_Dataset(DatasetInterface):
    name = 'HShift'

    def __init__(self, image, prompt, id=0, batch_size=100) -> None:
        super().__init__("", None, None)  # set batch_size to none to prevent randomization
        self.name = self.name + '-' + str(id)

        shift_min = -image.width
        shift_max = image.width
        images = []
        for shift_x in np.linspace(shift_min, shift_max, batch_size):
            images.append(image.transform(image.size, Image.AFFINE, (1, 0, shift_x, 0, 1, 0)))

        self.all_images = np.array(images)
        self.all_prompts = np.array([prompt] * batch_size)


class VShift_Dataset(DatasetInterface):
    name = 'VShift'

    def __init__(self, image, prompt, id=0, batch_size=100) -> None:
        super().__init__("", None, None)  # set batch_size to none to prevent randomization
        self.name = self.name + '-' + str(id)

        shift_min = -image.height
        shift_max = image.height
        images = []
        for shift_y in np.linspace(shift_min, shift_max, batch_size):
            images.append(image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, shift_y)))

        self.all_images = np.array(images)
        self.all_prompts = np.array([prompt] * batch_size)


class Noise_Dataset(DatasetInterface):
    name = 'Noisy'

    def __init__(self, image, prompt, id=0, seed=31415, batch_size=100) -> None:
        super().__init__('', seed, None)
        self.name = self.name + '-' + str(id)

        noise_level = 1 / batch_size
        image_array = np.array(image)
        images = []
        # noise = np.zeros_like(image_array.shape)
        for i in range(batch_size):
            # noise += noise_level*np.random.randint(low=0, high=256, size=image_array.shape,dtype='uint8')
            # noisy_image_array = np.clip((1-i*noise_level)*image_array + noise, 0, 255).astype('uint8')
            noise = np.random.randint(low=0, high=256, size=image_array.shape, dtype='uint8')
            noisy_image_array = np.clip((1 - i * noise_level) * image_array + i * noise_level * noise, 0, 255).astype(
                'uint8')
            noisy_image = Image.fromarray(noisy_image_array)
            images.append(noisy_image)

        self.all_images = np.array(images)
        self.all_prompts = np.array([prompt] * batch_size)


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
        all_images = []
        for i in indices:
            coco_img = self.img_infos[i]
            if not os.path.exists(self.img_folder + coco_img['file_name']):
                print('downloading file', coco_img['file_name'])
                response = requests.get(coco_img['coco_url'])
                with open(self.img_folder + coco_img['file_name'], "wb") as file:
                    file.write(response.content)

            all_images.append(Image.open(self.img_folder + coco_img['file_name']).convert("RGB"))

        if len(all_images) == 1:
            return all_images[0]
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
