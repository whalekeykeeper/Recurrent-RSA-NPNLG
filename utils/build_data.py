import math
import os.path
from utils.config import *
# from utils.image_and_text_utils import (
#     get_img_from_id,
#     vectorize_caption,
# )
import json
import pickle

import numpy as np
import pandas as pnd
from keras.preprocessing import image
from keras.applications import ResNet50


def make_id_to_caption():
    valids = 0
    invalids = 0
    id_to_caption = {}
    json_data = json.loads(
        open(
            "../data/visual_genome_JSON/region_descriptions.json",
            "r",
        ).read()
    )
    print("READ JSON, len:", len(json_data))

    max_sentence_length_vg, sym_set_vg = get_sym_set_for_VG()

    for i, image in enumerate(json_data):
        for s in image["regions"]:
            x_coordinate = s["x"]
            y_coordinate = s["y"]
            height = s["height"]
            width = s["width"]
            sentence = s["phrase"].lower()
            img_id = str(s["image_id"])
            region_id = str(s["region_id"])

            is_valid = valid_item(height, width, sentence, img_id, max_sentence_length_vg, sym_set_vg)

            if is_valid:
                valids += 1
                box = edit_region(height, width, x_coordinate, y_coordinate)
                id_to_caption[img_id + "_" + region_id] = (
                    vectorize_caption(sentence),
                    box
                )
            else:
                invalids += 1

        if i % 1000 == 0 and i > 0:
            print("PROGRESS:", i)

        # if i >6000:
        # 	break
        # print(len(id_to_caption))
        # print(id_to_caption)
    print(len(id_to_caption))
    print("num valid/ num invalid", valids, invalids)
    pickle.dump(
        id_to_caption, open("../data/id_to_caption", "wb")
    )


def get_sym_set_for_VG():
    with open("../data/visual_genome_JSON/region_descriptions.json", "r") as f:
        data = json.loads(f.read())
        id_cap = {}
        for i, image in enumerate(data):
            for s in image["regions"]:
                sentence = s["phrase"].lower()
                img_id = str(s["image_id"])
                region_id = str(s["region_id"])
                id_cap[img_id + "_" + region_id] = sentence
    # print(len(id_cap.items()))
    all_caps = [cap for cap in id_cap.values()]
    max_len = 0
    for cap in all_caps:
        if len(cap) > max_len:
            max_len = len(cap)
    # print(all_caps[0:10])
    sym_set_vg = set().union(*all_caps)
    sym_set_vg = list(sym_set_vg)
    sym_set_vg.sort()
    # print("max_sentence_length_vg:", max_len)
    # print(sym_set_vg)
    return max_len, sym_set_vg


# determine if a region and caption are suitable for inclusion in data
def valid_item(height, width, sentence, img_id, max_sentence_length_vg, sym_set_vg):
    ratio = (float(max(height, width))) / float(min(height, width))
    size = float(height)
    file_exists = os.path.isfile(os.path.isfile("/Users/qin/Documents/GitHub/Recurrent-RSA-NPNLG/data/visual_genome_data/VG_100K/" + str(img_id) + ".jpg"))
    if not file_exists:
        file_exists = os.path.isfile(os.path.isfile(
            "/Users/qin/Documents/GitHub/Recurrent-RSA-NPNLG/data/visual_genome_data/VG_100K_2/" + str(img_id) + ".jpg"))
    good_length = len(sentence) <= max_sentence_length_vg
    no_punctuation = all((char in sym_set_vg) for char in sentence)
    return (
        ratio < 1.25 and size > 100.0 and file_exists and good_length and no_punctuation
    )


def edit_region(height, width, x_coordinate, y_coordinate):
    if width > height:
        # check if image recentering causes box to go off the image up
        if y_coordinate + (height / 2) - (width / 2) < 0.0:
            box = (
                x_coordinate,
                y_coordinate,
                x_coordinate + max(width, height),
                y_coordinate + max(width, height),
            )
        else:
            box = (
                x_coordinate,
                y_coordinate + (height / 2) - (width / 2),
                x_coordinate + max(width, height),
                y_coordinate + (height / 2) - (width / 2) + max(width, height),
            )
    else:
        # check if image recentering causes box to go off the image to the left
        if x_coordinate + (width / 2) - (height / 2) < 0.0:
            box = (
                x_coordinate,
                y_coordinate,
                x_coordinate + max(width, height),
                y_coordinate + max(width, height),
            )
        else:
            box = (
                x_coordinate + (width / 2) - (height / 2),
                y_coordinate,
                x_coordinate + (width / 2) - (height / 2) + max(width, height),
                y_coordinate + max(width, height),
            )

    return box

#
# def store_image_reps():
#     id_to_caption = pickle.load(
#         open("data/id_to_caption", "rb")
#     )
#     print("len id_to_caption", len(id_to_caption))
#
#     size = 1000
#     num_images = len(id_to_caption)
#     full_output = np.random.randn(len(id_to_caption), rep_size)
#     print("rep_size", rep_size)
#     mod_num = num_images % size
#     r = math.ceil(num_images / size)
#     for j in range(math.ceil(len(sorted(list(id_to_caption))) / size)):
#         print(
#             "RUNNING IMAGES THROUGH RESNET: step",
#             j + 1,
#             "out of",
#             len(range(math.ceil(len(list(id_to_caption)) / size))),
#         )
#         if j == r - 1:
#             num = mod_num
#         else:
#             num = size
#         img_tensor = np.zeros((num, 224, 224, 3))
#
#         for i, item in enumerate(
#                 sorted(list(id_to_caption))[j * size: ((j * size) + num)]
#         ):
#             img = get_img_from_id(item, id_to_caption)
#             img_vector = image.img_to_array(img)
#             img_tensor[i] = img_vector
#
#         reps = fc_resnet.predict(img_tensor)
#         # print("check",reps.shape[0],len(list(id_to_caption)[j*size:((j*size)+num)]))
#         assert reps.shape[0] == len(
#             list(id_to_caption)[j * size: ((j * size) + num)]
#         )
#         full_output[j * size: j * size + num] = reps[:num]
#
#         df = pnd.DataFrame(full_output, index=sorted(list(id_to_caption)))
#
#         assert df.shape == (len(id_to_caption), rep_size)
#
#         df.to_pickle(REP_DATA_PATH + "reps.pickle")
#         if j % 10 == 0:
#             df.to_pickle(REP_DATA_PATH + "reps.pickle_backup")
#

if __name__ == "__main__":
    # with open("data/visual_genome_JSON/region_descriptions.json", "r") as f:
    #     data = json.loads(f.read())
    #     text = data[0:2]
    #     jsonString = json.dumps(text)
    #     jsonFile = open("data/visual_genome_JSON/region_descriptions_sample.json", "w")
    #     jsonFile.write(jsonString)
    #     jsonFile.close()


    # e.g.: id_to_caption = {'10_1382': ('the boy with ice cream',(139,82,421,87)),'11_1382': ('the man with ice cream',(139,82,421,87))}...
    print("MAKING id_to_caption")
    make_id_to_caption()

    print("COMPUTING AND STORING image reps")

    # feed each image corresponding to a region in image from id in id_to_caption into a [rep_size] dim vector (or consider fewer)
    # save as pandas dataframe, with labelled columns

    # store_image_reps()`