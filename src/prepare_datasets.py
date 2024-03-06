# Author: Atharva Kulkarni
# Base code taken from https://github.com/facebookresearch/BalancingGroups/blob/main/datasets.py

import os
import torch
import pandas as pd
import numpy as np

from PIL import Image
from torchvision import transforms
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    AutoImageProcessor,
)
from datasets import (
    load_dataset,
    Dataset as hf_Dataset,
    Image as hf_Image
)
from torch.utils.data import DataLoader
from sklearn.datasets import make_blobs
import pandas as pd
import copy



class GroupDataset:
    def __init__(
        self, split, root, metadata, transform, subsample_what=None, duplicates=None
    ):
        self.transform_ = transform
        df = pd.read_csv(metadata)
        df = df[df["split"] == ({"tr": 0, "va": 1, "te": 2}[split])]
        if split == 'tr':
            self.index = df.index.tolist()
            print(f"\nSize of Training data: {len(self.index)}\n\n")
        
        self.i = list(range(len(df)))
        if "waterbirds" in metadata or "celeba" in metadata:
            self.x = copy.deepcopy(self.i)
        else:
            self.x = df["filename"].astype(str).map(lambda x: os.path.join(root, x)).tolist()
        self.y = df["y"].tolist()
        self.g = df["a"].tolist()

        self.count_groups()
        
        if subsample_what is not None:
            self.subsample_(subsample_what)

        if duplicates is not None:
            self.duplicate_(duplicates)

    def count_groups(self):
        self.wg, self.wy = [], []

        self.nb_groups = len(set(self.g))
        self.nb_labels = len(set(self.y))
        self.group_sizes = [0] * self.nb_groups * self.nb_labels
        self.class_sizes = [0] * self.nb_labels

        for i in self.i:
            self.group_sizes[self.nb_groups * self.y[i] + self.g[i]] += 1
            self.class_sizes[self.y[i]] += 1

        for i in self.i:
            self.wg.append(
                len(self) / self.group_sizes[self.nb_groups * self.y[i] + self.g[i]]
            )
            self.wy.append(len(self) / self.class_sizes[self.y[i]])

    def subsample_(self, subsample_what):
        perm = torch.randperm(len(self)).tolist()

        if subsample_what == "groups":
            min_size = min(list(self.group_sizes))
        else:
            min_size = min(list(self.class_sizes))

        counts_g = [0] * self.nb_groups * self.nb_labels
        counts_y = [0] * self.nb_labels
        new_i = []
        for p in perm:
            y, g = self.y[self.i[p]], self.g[self.i[p]]

            if (
                subsample_what == "groups"
                and counts_g[self.nb_groups * int(y) + int(g)] < min_size
            ) or (subsample_what == "classes" and counts_y[int(y)] < min_size):
                counts_g[self.nb_groups * int(y) + int(g)] += 1
                counts_y[int(y)] += 1
                new_i.append(self.i[p])

        self.i = new_i
        self.count_groups()

    def duplicate_(self, duplicates):
        new_i = []
        for i, duplicate in zip(self.i, duplicates):
            new_i += [i] * duplicate
        self.i = new_i
        self.count_groups()

    def __getitem__(self, i):
        j = self.i[i]
        x = self.transform(self.x[j])
        y = torch.tensor(self.y[j], dtype=torch.long)
        g = torch.tensor(self.g[j], dtype=torch.long)
        return torch.tensor(i, dtype=torch.long), x, y, g

    def __len__(self):
        return len(self.i)



class MaskGenerator:
    """
    A class to generate boolean masks for the pretraining task.

    A mask is a 1D tensor of shape (model_patch_size**2,) where the value is either 0 or 1,
    where 1 indicates "masked".
    """

    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        if self.input_size % self.mask_patch_size != 0:
            raise ValueError("Input size must be divisible by mask patch size")
        if self.mask_patch_size % self.model_patch_size != 0:
            raise ValueError("Mask patch size must be divisible by model patch size")

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        return torch.tensor(mask.flatten())


class CustomGaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = torch.nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = torch.nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img
    

    
# ----------------------------------------------------- Waterbirds Dataset -----------------------------------------------------

class Waterbirds(GroupDataset):
    def __init__(
        self, 
        data_path, 
        split, 
        subsample_what=None, 
        duplicates=None,
        method="erm",
        seed=0,
    ):
        self.method = method
        root = os.path.join(data_path, "waterbird_complete95_forest2water2/")
        metadata = os.path.join(data_path,"metadata_waterbirds.csv")
        image_df = pd.read_csv(metadata).rename(columns={'filename': 'image'})
        image_df = image_df[image_df["split"] == ({"tr": 0, "va": 1, "te": 2}[split])]
        image_df['image'] = image_df['image'].apply(lambda x: os.path.join(root, x))
        self.images = hf_Dataset.from_pandas(
            image_df
        ).cast_column("image", hf_Image())
        
        self.image_processor  = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        normalize = transforms.Normalize(mean=self.image_processor.image_mean, std=self.image_processor.image_std)
        if split == "tr":
            self.transformations = transforms.Compose([
                transforms.RandomResizedCrop((384, 384), scale=(0.05, 1.0)),
                transforms.ToTensor(),
                normalize,
            ])      
                
            self.mask_generator = None  
            if "mt" in self.method:
                
                if "mt2" in self.method:
                    color_jitter_strength = 1
                    color_jitter = transforms.ColorJitter(
                        0.8 * color_jitter_strength, 
                        0.8 * color_jitter_strength, 
                        0.8 * color_jitter_strength, 
                        0.2 * color_jitter_strength
                    )
                    self.mt_transformations = transforms.Compose([
                        transforms.Lambda(lambda img: img.convert("RGB")),
                        transforms.RandomResizedCrop((384, 384), scale=(0.67, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomApply([color_jitter], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        CustomGaussianBlur(kernel_size=int(0.1 * 384)),
                        transforms.ToTensor()
                    ])
                    
                else:
                    self.mt_transformations = transforms.Compose([
                        transforms.Lambda(lambda img: img.convert("RGB")),
                        transforms.RandomResizedCrop((384, 384), scale=(0.67, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ])
                    self.mask_generator = MaskGenerator(
                        input_size=384,
                        mask_patch_size=32,
                        model_patch_size=16,
                        mask_ratio=0.6,
                    )
        else:
            self.transformations = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                normalize,
            ])
            
        print(f"\n{split} data transformations: \n{self.transformations}\n")
        self.processed_image = self.images.with_transform(self.preprocess_function)
         
        self.data_type = "images"
        super().__init__(
            split, "", metadata, self.transform, subsample_what, duplicates
        )

    
    def preprocess_function(self, examples):
        examples["pixel_values"] = [
            self.transformations(image.convert("RGB")) for image in examples["image"]
        ]
        return examples
     
    
    def mt_preprocess_function(self, examples):
        """Preprocess a batch of images by applying transforms + creating a corresponding mask, indicating
        which patches to mask."""
        examples["pixel_values"] = [self.mt_transformations(image) for image in examples["image"]]
        if self.method != "erm_mt2_l1":
            examples["mask"] = [self.mask_generator() for i in range(len(examples["image"]))]
        return examples
    
    
    def mt_collate_fn(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        if self.method != "erm_mt2_l1":
            mask = torch.stack([example["mask"] for example in examples])
            return {"pixel_values": pixel_values, "bool_masked_pos": mask}
        return {"pixel_values": pixel_values}


    def transform(self, idx):
        return self.processed_image[int(idx)]['pixel_values']


    def get_mt_data_loader(self, mt_bs, shuffle, sampler):
        self.mt_processed_image = self.images.with_transform(self.mt_preprocess_function)
        return DataLoader(
            dataset=self.mt_processed_image,
            batch_size=mt_bs,
            collate_fn=self.mt_collate_fn,
            shuffle=shuffle,
            sampler=sampler,
            # num_workers=4,
            pin_memory=True,
        )



# ----------------------------------------------------- CelebA Dataset -----------------------------------------------------

class CelebA(GroupDataset):
    def __init__(
        self, 
        data_path, 
        split, 
        subsample_what=None, 
        duplicates=None,
        method="erm",
        seed=0,
    ):
        self.method = method
        root = os.path.join(data_path, "img_align_celeba/")
        metadata = os.path.join(data_path,"metadata_celeba.csv")
        image_df = pd.read_csv(metadata).rename(columns={'filename': 'image'})
        image_df = image_df[image_df["split"] == ({"tr": 0, "va": 1, "te": 2}[split])]
        image_df['image'] = image_df['image'].apply(lambda x: os.path.join(root, x))
        self.images = hf_Dataset.from_pandas(
            image_df
        ).cast_column("image", hf_Image())
        
        self.image_processor  = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        normalize = transforms.Normalize(mean=self.image_processor.image_mean, std=self.image_processor.image_std)
        if split == "tr":
            self.transformations = transforms.Compose([
                transforms.RandomResizedCrop((384, 384), scale=(0.05, 1.0)),
                transforms.ToTensor(),
                normalize,
            ])      
                  
            self.mask_generator = None
            if "mt" in self.method:
                
                if "mt2" in self.method:
                    color_jitter_strength = 1
                    color_jitter = transforms.ColorJitter(
                        0.8 * color_jitter_strength, 
                        0.8 * color_jitter_strength, 
                        0.8 * color_jitter_strength, 
                        0.2 * color_jitter_strength
                    )
                    self.mt_transformations = transforms.Compose([
                        transforms.Lambda(lambda img: img.convert("RGB")),
                        transforms.RandomResizedCrop((384, 384), scale=(0.67, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomApply([color_jitter], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        CustomGaussianBlur(kernel_size=int(0.1 * 384)),
                        transforms.ToTensor()
                    ])
                    
                else:
                    self.mt_transformations = transforms.Compose([
                        transforms.Lambda(lambda img: img.convert("RGB")),
                        transforms.RandomResizedCrop((384, 384), scale=(0.67, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ])
                    self.mask_generator = MaskGenerator(
                        input_size=384,
                        mask_patch_size=32,
                        model_patch_size=16,
                        mask_ratio=0.6,
                    )
        else:
            self.transformations = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                normalize,
            ])
            
        print(f"\n{split} data transformations: \n{self.transformations}\n")
        self.processed_image = self.images.with_transform(self.preprocess_function)
         
        self.data_type = "images"
        super().__init__(
            split, "", metadata, self.transform, subsample_what, duplicates
        )

    
    def preprocess_function(self, examples):
        examples["pixel_values"] = [
            self.transformations(image.convert("RGB")) for image in examples["image"]
        ]
        return examples
     
    
    def mt_preprocess_function(self, examples):
        """Preprocess a batch of images by applying transforms + creating a corresponding mask, indicating
        which patches to mask."""
        examples["pixel_values"] = [self.mt_transformations(image) for image in examples["image"]]
        if self.method != "erm_mt2_l1":
            examples["mask"] = [self.mask_generator() for i in range(len(examples["image"]))]
        return examples
    
    
    def mt_collate_fn(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        if self.method != "erm_mt2_l1":
            mask = torch.stack([example["mask"] for example in examples])
            return {"pixel_values": pixel_values, "bool_masked_pos": mask}
        return {"pixel_values": pixel_values}


    def transform(self, idx):
        return self.processed_image[int(idx)]['pixel_values']


    def get_mt_data_loader(self, mt_bs, shuffle, sampler):
        self.mt_processed_image = self.images.with_transform(self.mt_preprocess_function)
        return DataLoader(
            dataset=self.mt_processed_image,
            batch_size=mt_bs,
            collate_fn=self.mt_collate_fn,
            shuffle=shuffle,
            sampler=sampler,
            # num_workers=4,
            pin_memory=True,
        )



# ----------------------------------------------------- MultiNLI Dataset -----------------------------------------------------

class MultiNLI(GroupDataset):
    def __init__(
        self, 
        data_path, 
        split, 
        subsample_what=None, 
        duplicates=None, 
        method="erm",
        seed=0,
        mlm_probability=0.15
    ):
        root = os.path.join(data_path, "glue_data", "MNLI")
        metadata = os.path.join(data_path, "metadata", "metadata_multinli.csv")

        self.features_array = []
        for feature_file in [
            "cached_train_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli-mm",
        ]:
            features = torch.load(os.path.join(root, feature_file))
            self.features_array += features

        self.all_input_ids = torch.tensor(
            [f.input_ids for f in self.features_array], dtype=torch.long
        )
        self.all_input_masks = torch.tensor(
            [f.input_mask for f in self.features_array], dtype=torch.long
        )
        self.all_segment_ids = torch.tensor(
            [f.segment_ids for f in self.features_array], dtype=torch.long
        )
        self.all_label_ids = torch.tensor(
            [f.label_id for f in self.features_array], dtype=torch.long
        )

        self.x_array = torch.stack(
            (self.all_input_ids, self.all_input_masks, self.all_segment_ids), dim=2
        )
        
        if split == "tr":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
                   
            self.data_collator = None         
            if "mt" in method:
                
                if "mt2" in method:
                    print("\nUsing CLM...\n")
                    self.data_collator = DataCollatorForLanguageModeling(
                        tokenizer=self.tokenizer,
                        mlm=False,
                        return_tensors='pt'
                    )
                    
                else:
                    print("\nUsing MLM...\n")
                    self.data_collator = DataCollatorForLanguageModeling(
                        tokenizer=self.tokenizer,
                        mlm=True,
                        mlm_probability=mlm_probability,
                        return_tensors='pt'
                    )
                                    
            self.tokenized_text = hf_Dataset.from_dict(
                {
                    'input_ids': self.all_input_ids,
                    'attention_mask': self.all_input_masks,
                    'token_type_ids': self.all_segment_ids,
                    
                }
            )
            print(self.tokenized_text)

        self.data_type = "text"
        super().__init__(
            split, "", metadata, self.transform, subsample_what, duplicates
        )


    def transform(self, i):
        return self.x_array[int(i)]
    
    
    def get_mt_data_loader(self, mt_bs, shuffle, sampler):
        mt_data = hf_Dataset.from_dict(self.tokenized_text[self.index])
        return DataLoader(
            dataset=mt_data,
            batch_size=mt_bs,
            collate_fn=self.data_collator,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=True,
        )

        

# ----------------------------------------------------- CivilComments Dataset -----------------------------------------------------

class CivilComments(GroupDataset):
    def __init__(
        self,
        data_path,
        split,
        subsample_what=None,
        duplicates=None,
        method="erm",
        seed=0,
        mlm_probability=0.15,
        granularity="coarse",
    ):
        
        if "small" in data_path:
            metadata = os.path.join(data_path, "metadata_civilcomments_small_{}.csv".format(granularity))
            self.text = load_dataset(
                "csv", 
                data_files=os.path.join(data_path, "civilcomments_small_{}.csv".format(granularity)),
                split='train',
                # cache_dir="/scratch/"
            )
            
        else:
            metadata = os.path.join(data_path, "metadata_civilcomments_{}.csv".format(granularity))
            self.text = load_dataset(
                "csv", 
                data_files=os.path.join(data_path, "civilcomments_{}.csv".format(granularity)),
                split='train',
                # cache_dir="/scratch/"
            )
            
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
        self.tokenized_text = self.text.map(
            self.preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=list(self.text.features.keys()),
        )
        
        self.data_collator = None
        if "mt" in method:
            
            if "mt2" in method:
                print("\nUsing CLM...\n")
                self.data_collator = DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=False,
                    return_tensors='pt'
                )
            else:
                print("\nUsing MLM...\n")
                self.data_collator = DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=True,
                    mlm_probability=mlm_probability,
                    return_tensors='pt'
                )
            
        self.data_type = "text"
        
        super().__init__(
            split, "", metadata, self.transform, subsample_what, duplicates
        )

    def preprocess_function(self, examples):
        # Remove empty lines
        examples['comment_text'] = [
            line for line in examples['comment_text'] if len(line) > 0 and not line.isspace()
        ]
        return self.tokenizer(
            examples['comment_text'],
            padding="max_length",
            truncation=True,
            max_length=220,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        
    def transform(self, idx):
        tokens = self.tokenized_text[int(idx)]
        return torch.squeeze(
            torch.stack(
                (
                    torch.tensor(tokens["input_ids"], dtype=torch.long).reshape(1, 220),
                    torch.tensor(tokens["attention_mask"], dtype=torch.long).reshape(1, 220),
                    torch.tensor(tokens["token_type_ids"], dtype=torch.long).reshape(1, 220)
                ),
                dim=2,
            ),
            dim=0,
        )
        
    def get_mt_data_loader(self, mt_bs, shuffle, sampler):
        mt_data = hf_Dataset.from_dict(self.tokenized_text[self.index])
        return DataLoader(
            dataset=mt_data,
            batch_size=mt_bs,
            collate_fn=self.data_collator,
            shuffle=shuffle,
            sampler=sampler,
            # num_workers=4,
            pin_memory=True,
        )
            

class CivilCommentsFine(CivilComments):
    def __init__(self, data_path, split, subsample_what=None, duplicates=None, seed=0):
        super().__init__(data_path, split, subsample_what, duplicates, seed, mlm_probability=0.15, granularity="fine")



# ----------------------------------------------------- Loader -----------------------------------------------------

def get_loaders(data_path, dataset_name, bs, method="erm", duplicates=None, seed=0):
    Dataset = {
        "multinli": MultiNLI,
        "civilcomments": CivilComments,
        "civilcomments_small": CivilComments,
        "waterbirds": Waterbirds,
        "celeba": CelebA
    }[dataset_name]

    def dl(dataset, bs, shuffle, weights, method):
        print(f"\nbs: {bs}\tshuffle: {shuffle}\tmethod: {method}\n")
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
        else:
            sampler = None
            
        if any(substring in method for substring in ["mt", "l1"]):
            return [
                DataLoader(
                    dataset,
                    batch_size=bs[0],
                    shuffle=shuffle,
                    sampler=sampler,
                    pin_memory=True,
                ),
                dataset.get_mt_data_loader(bs[1], shuffle, sampler)
            ]
            
        else:
            return DataLoader(
                dataset,
                batch_size=bs,
                shuffle=shuffle,
                sampler=sampler,
                pin_memory=True,
            )
            

    if method == "subg":
        subsample_what = "groups"
    elif method == "suby":
        subsample_what = "classes"
    else:
        subsample_what = None

    dataset_tr = Dataset(data_path, "tr", subsample_what, duplicates, method=method, seed=seed)

    if method == "rwg" or method == "dro":
        weights_tr = dataset_tr.wg
    elif method == "rwy":
        weights_tr = dataset_tr.wy
    else:
        weights_tr = None

    if method is not None and "mt" in method:
        eval_bs = bs[0]
    else:
        eval_bs = bs
        
    return {
        "tr": dl(dataset_tr, bs, weights_tr is None, weights_tr, method),
        "va": dl(Dataset(data_path, "va", None), eval_bs, False, None, None),
        "te": dl(Dataset(data_path, "te", None), eval_bs, False, None, None),
    }