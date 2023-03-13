import json
import os
from collections import Counter

import nltk as nltk
import torch
import torch.utils.data as data
from PIL import Image
from typing import List, MutableSet
from evaluation.build_test_sets import types as ts_types
import os


class VG:
    def __init__(self, vg_json: str, data_size=None, ts1_json=None, ts2_json=None):
        self.ids = []  # [  img_id + "_" + region_id  ]
        self.captions = {}  # { id: annotation }
        self.box = {}  # { id: list of four ints}
        self.size = data_size
        self._read_captions(vg_json)
        self._reserved_image_ids = self._get_reserved_image_ids(
            ts1_json, ts2_json)

        if data_size is not None:
            self.ids = self.ids[:data_size]
            cropped_captions = {}
            for id in self.ids:
                cropped_captions[id] = self.captions[id]
            self.captions = cropped_captions
            cropped_boxes = {}
            for id in self.ids:
                cropped_boxes[id] = self.box[id]
            self.box = cropped_boxes

    def _get_reserved_image_ids(self, ts1_json: str, ts2_json: str) -> List[str]:
        """
        If TS1 and TS2 have been generated, this function
        returns a list of image ids that were used in 
        either test set.
        """
        ts1: ts_types.TS1 = None
        ts2: ts_types.TS2 = None
        if os.path.exists(ts1_json):
            with open(ts1_json, "r") as f:
                ts1 = json.loads(f.read())
        if os.path.exists(ts2_json):
            with open(ts2_json, "r") as f:
                ts2 = json.loads(f.read())

        reserved_ids: MutableSet[int] = set()

        if ts1:
            for cluster in ts1:
                reserved_ids.add(cluster["target"]["original_image_id"])
                reserved_ids.update(item["original_image_id"]
                                    for item in cluster["distractors"])
        if ts2:
            for cluster in ts2:
                reserved_ids.add(cluster["target"]["original_image_id"])
                reserved_ids.update(item["original_image_id"]
                                    for item in cluster["distractors"])

        return [str(image_id) for image_id in reserved_ids]

    def _read_captions(self, vg_json: str):
        with open(vg_json, "r") as f:
            data = json.loads(f.read())
            for i, image in enumerate(data):
                for s in image["regions"]:
                    img_id = str(s["image_id"])

                    # If this image is used either in TS1 or TS2, skip it.
                    if img_id in self._reserved_image_ids:
                        continue

                    region_id = str(s["region_id"])
                    x_coordinate = s["x"]
                    y_coordinate = s["y"]
                    height = s["height"]
                    width = s["width"]
                    self.ids.append(img_id + "_" + region_id)
                    self.captions[img_id + "_" +
                                  region_id] = s["phrase"].lower()
                    one_box = self._edit_region(height, width, x_coordinate,
                                                y_coordinate)
                    self.box[img_id + "_" + region_id] = one_box

    def _edit_region(self, height: int, width: int, x_coordinate: int, y_coordinate: int) -> list[int] | list[int |
                                                                                                              float]:
        if width > height:
            # check if image recentering causes box to go off the image up
            if y_coordinate + (height / 2) - (width / 2) < 0.0:
                box = [
                    x_coordinate,
                    y_coordinate,
                    x_coordinate + max(width, height),
                    y_coordinate + max(width, height),
                ]
            else:
                box = [
                    x_coordinate,
                    y_coordinate + (height / 2) - (width / 2),
                    x_coordinate + max(width, height),
                    y_coordinate + (height / 2) - (width / 2) +
                    max(width, height),
                ]
        else:
            # check if image recentering causes box to go off the image to the left
            if x_coordinate + (width / 2) - (height / 2) < 0.0:
                box = [
                    x_coordinate,
                    y_coordinate,
                    x_coordinate + max(width, height),
                    y_coordinate + max(width, height),
                ]
            else:
                box = [
                    x_coordinate + (width / 2) - (height / 2),
                    y_coordinate,
                    x_coordinate + (width / 2) - (height / 2) +
                    max(width, height),
                    y_coordinate + max(width, height),
                ]
        return box

    def get_caption(self, id: str) -> str:
        return self.captions[id]

    def get_box(self, id: str) -> list:
        return self.box[id]

    def get_image(self, id: str) -> Image:
        img_id, region_id = id.split('_')
        # print('img_id, region_id: ', img_id, region_id)
        path1 = "./data/visual_genome_data/VG_100K/" + str(img_id) + ".jpg"
        path2 = "./data/visual_genome_data/VG_100K_2/" + str(img_id) + ".jpg"
        if os.path.isfile(path1):
            path = path1
        else:
            if os.path.isfile(path2):
                path = path2
            else:
                raise Exception("Mistake. This image is not in dataset.")
        img = Image.open(path)
        # img.show()
        return img  # This is only the default img, it is not the cropped/resized one

    def get_reesized_image(self, id: str) -> Image:
        img_id, region_id = id.split('_')
        path1 = "./data/visual_genome_data/VG_100K/" + str(img_id) + ".jpg"
        path2 = "./data/visual_genome_data/VG_100K_2/" + str(img_id) + ".jpg"
        if os.path.isfile(path1):
            path = path1
        else:
            if os.path.isfile(path2):
                path = path2
            else:
                raise Exception("Mistake. This image is not in dataset.")
        img = Image.open(path)
        # crop region from img
        box = self.box[id]
        cropped_img = img.crop(box)
        # resize into square
        resized_img = cropped_img.resize([224, 224], Image.LANCZOS)
        # resized_img.show()
        return resized_img

    def __len__(self):
        return len(self.ids)


class Coco:
    def __init__(self, coco_json: str, data_size=None):
        self.ids = []
        self.captions = {}  # { id: annotation }
        self.files = {}  # { id: filename }
        self.size = data_size
        self._read_captions(coco_json)

        if data_size is not None:
            self.ids = self.ids[:data_size]

    def _read_captions(self, coco_json: str):
        with open(coco_json) as f:
            data = json.load(f)
            for image in data["images"]:
                self.ids.append(image["id"])
                self.files[image["id"]] = image["file_name"]
                self.captions[image["id"]] = None

            for image in data["annotations"]:
                self.captions[image["image_id"]] = image["caption"]

    def get_image(self, id: str) -> str:
        return self.files[id]

    def get_caption(self, id: str) -> str:
        return self.captions[id]

    def __len__(self):
        return len(self.ids)


class Vocabulary:
    def __init__(self, captions_json: str, level: str, data_size=None, dataset="coco"):
        if dataset not in ["coco", "vg"]:
            raise Exception(
                "Dataset not found. Make sure it is either 'coco' or 'vg'")

        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.data_size = data_size
        self.dataset = dataset
        self.build_vocab(captions_json, level)

    def _build_vocab_vg(self, captions_json: str, level: str) -> List[str]:
        vg = VG(captions_json, self.data_size)
        counter = Counter()
        for i, id in enumerate(vg.ids):
            caption = str(vg.get_caption(id))
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)
        words = [word for word, cnt in counter.items()]
        if level == 'word':
            return words
        else:
            chars = set().union(*words)
            chars = list(chars)
            chars.sort()
            return chars

    def _build_vocab_coco(self, captions_json: str, level: str) -> List[str]:
        coco = Coco(captions_json, self.data_size)
        counter = Counter()
        for i, id in enumerate(coco.ids):
            caption = str(coco.get_caption(id))
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)
        words = [word for word, cnt in counter.items()]
        if level == 'word':
            return words
        else:
            chars = set().union(*words)
            chars = list(chars)
            chars.sort()
            # print(chars)
            return chars

    def build_vocab(self, captions_json: str, level: str):
        if self.dataset == "coco":
            words_chars = self._build_vocab_coco(captions_json, level)
        elif self.dataset == "vg":
            words_chars = self._build_vocab_vg(captions_json, level)

        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')

        for i, word in enumerate(words_chars):
            self.add_word(word)

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx["<unk>"])

    def __len__(self):
        return len(self.word2idx)


class CocoDataset(data.Dataset):
    """
    Custom Coco Dataset for compatibility with torch DataLoader
    """

    def __init__(self, image_dir: str, captions_json: str, vocab, transform=None, data_size=None, ):
        self.image_dir = image_dir
        self.coco = Coco(captions_json, data_size)
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, item):
        # print("item: ", item)
        coco_id = self.coco.ids[item]
        caption_str = self.coco.get_caption(coco_id)
        image_file = self.coco.get_image(coco_id)

        # load and preprocess image
        image = Image.open(os.path.join(
            self.image_dir, image_file)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # vectorize caption
        tokens = nltk.tokenize.word_tokenize(caption_str.lower())
        caption_vect = [self.vocab("<start>")]
        caption_vect.extend([self.vocab(token) for token in tokens])
        caption_vect.append(self.vocab("<end>"))
        caption_vect = torch.Tensor(caption_vect)
        return image, caption_vect

    def __len__(self):
        return len(self.coco)


class VGDataset(data.Dataset):
    """
    Custom VG Dataset for compatibility with torch DataLoader
    """

    def __init__(self, captions_json_vg: str, vocab, transform=None, data_size=None, ts1_json: str = None, ts2_json: str = None):
        self.vg = VG(captions_json_vg, data_size,
                     ts1_json=ts1_json, ts2_json=ts2_json)
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, item):
        vg_id = self.vg.ids[item]
        caption_str = self.vg.get_caption(vg_id)

        # load and preprocess image
        image = self.vg.get_reesized_image(vg_id).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # vectorize caption
        tokens = nltk.tokenize.word_tokenize(caption_str.lower())
        caption_vect = [self.vocab("<start>")]
        caption_vect.extend([self.vocab(token) for token in tokens])
        caption_vect.append(self.vocab("<end>"))
        caption_vect = torch.Tensor(caption_vect)
        return image, caption_vect

    def __len__(self):
        return len(self.vg)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


def get_loader(image_dir, captions_json, vocab, transform, batch_size, shuffle, num_workers, dataset='coco',
               data_size=20, ts1_json: str = None, ts2_json: str = None):
    """Returns torch.utils.data.DataLoader for custom coco/vg dataset."""
    if dataset == "coco":
        # COCO caption dataset
        coco = CocoDataset(image_dir=image_dir,
                           captions_json=captions_json,
                           vocab=vocab,
                           transform=transform,
                           data_size=20, ts1_json=ts1_json, ts2_json=ts2_json)
        # Data loader for COCO dataset
        # This will return (images, captions, lengths) for each iteration.
        # images: a tensor of shape (batch_size, 3, 224, 224).
        # captions: a tensor of shape (batch_size, padded_length).
        # lengths: a list indicating valid length for each caption. length is (batch_size).
        data_loader = torch.utils.data.DataLoader(dataset=coco,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  collate_fn=collate_fn)
    else:
        vg = VGDataset(captions_json_vg=captions_json,
                       vocab=vocab, transform=transform, data_size=20)
        # Data loader for VG dataset
        # This will return (images, captions, lengths) for each iteration.
        # images: a tensor of shape (batch_size, 3, 224, 224).
        # captions: a tensor of shape (batch_size, padded_length).
        # lengths: a list indicating valid length for each caption. length is (batch_size).
        data_loader = torch.utils.data.DataLoader(dataset=vg,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  collate_fn=collate_fn)

    return data_loader


if __name__ == '__main__':
    # nltk.download('punkt')

    # # for COCO
    # captions_json = "./data/captions_train2014.json"
    # image_dir_coco = "./data/train2014"
    # vocab = Vocabulary(captions_json, level='word')
    # dataset = CocoDataset(image_dir_coco, captions_json, vocab)
    # print(dataset[0])
    # data_loader = get_loader(image_dir_coco, captions_json, vocab, None, 128, shuffle=True, num_workers=1)
    # print(data_loader)

    # # for VG
    captions_json_vg = "./data/visual_genome_JSON/region_descriptions_sample.json"
    image_dir_vg = "../data/visual_genome_data"
    vg = VG(captions_json_vg)
    print(vg.captions['1_4091'])
    # print(vg.box)
    # vg.get_image('1_1382')
    vg.get_reesized_image('1_1382')
    vocab = Vocabulary(captions_json_vg, level='char', dataset='vg')
    dataset = VGDataset(captions_json_vg, vocab)
    print(dataset[0])
    data_loader = get_loader(image_dir_vg, captions_json_vg, vocab, None, 128, shuffle=True, num_workers=1,
                             dataset='vg')
    print(data_loader)
