import os.path
import numpy as np

from utils.config import *
import urllib.request
import pickle
import requests
from keras.applications import ResNet50
from keras.preprocessing import image
from PIL import Image as PIL_Image
import shutil
# from config import img_rep_layer

###IMAGE UTILS


def overlap(target_box, candidate_box):
    saved_x1 = target_box[0]
    saved_y1 = target_box[1]
    saved_x2 = target_box[2]
    saved_y2 = target_box[3]

    x1 = candidate_box[0]
    y1 = candidate_box[1]
    x2 = candidate_box[2]
    y2 = candidate_box[3]

    cond1 = saved_x1 < x1 and x1 < saved_x2
    cond2 = saved_y1 < y1 and y1 < saved_y2
    cond3 = saved_x1 < x2 and x2 < saved_x2
    cond4 = saved_y1 < y2 and y2 < saved_y2

    return cond1 or cond2 or cond3 or cond4


def get_img_from_id(item, id_to_caption):
    img_id, region_id = item.split("_")
    path = IMG_DATA_PATH + "VG_100K/" + img_id + ".jpg"
    img = PIL_Image.open(path)
    # crop region from img
    box = id_to_caption[item][region]
    # print("box",box)
    cropped_img = img.crop(box)
    # print("cropped_img", image.img_to_array(cropped_img))
    # resize into square
    resized_img = cropped_img.resize([224, 224], PIL_Image.LANCZOS)

    return resized_img


def get_rep_from_id(item, id_to_caption):
    img_id, region_id = item.split("_")
    path = IMG_DATA_PATH + "VG_100K/" + img_id + ".jpg"
    img = PIL_Image.open(path)
    # crop region from img
    box = id_to_caption[item][region]
    # print("box",box)
    cropped_img = img.crop(box)
    # print("cropped_img", image.img_to_array(cropped_img))
    # resize into square
    resized_img = cropped_img.resize([224, 224], PIL_Image.ANTIALIAS)

    display(resized_img)

    img = np.expand_dims(image.img_to_array(resized_img), 0)

    img = ResNet50(img_rep_layer).predict(img)
    return img


# nb: only the first part of the id, i.e. the image, not the region: doesn't crop
def get_rep_from_img_id(img_id):
    path = IMG_DATA_PATH + "VG_100K/" + img_id + ".jpg"
    img = PIL_Image.open(path)
    resized_img = img.resize([224, 224], PIL_Image.ANTIALIAS)

    img = np.expand_dims(image.img_to_array(resized_img), 0)

    img = ResNet50(img_rep_layer).predict(img)
    return img


# TODO use or remove


def get_img_from_url(url):
    response = requests.get(url, stream=True)
    with open("charpragcap/resources/img.jpg", "wb") as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response

    img = PIL_Image.open("charpragcap/resources/img.jpg")

    return img

    # model = ResNet50(img_rep_layer)

    # # file_name = "charpragcap/resources/local-filename.jpg"
    # # urllib.request.urlretrieve(url, file_name)
    # # img = PIL_Image.open(file_name)

    # img = img.resize([224,224],PIL_Image.ANTIALIAS)
    # display(img)
    # img = np.expand_dims(image.img_to_array(img),0)

    # rep = resnet(img_rep_layer).predict(img)
    # return rep


def get_rep_from_url(url, model):
    response = requests.get(url, stream=True)
    with open("charpragcap/resources/img.jpg", "wb") as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response

    img = PIL_Image.open("charpragcap/resources/img.jpg")

    # file_name = "charpragcap/resources/local-filename.jpg"
    # urllib.request.urlretrieve(url, file_name)
    # img = PIL_Image.open(file_name)

    img = img.resize([224, 224], PIL_Image.ANTIALIAS)
    # display(img)
    img = np.expand_dims(image.img_to_array(img), 0)

    rep = model.predict(img)
    return rep


# for ipython image displaying
def display_image(number):
    id_to_caption = pickle.load(open("charpragcap/resources/id_to_caption", "rb"))
    chosen_id = list(id_to_caption)[number]
    img_path = "data/VG_100K/" + str(chosen_id) + ".jpg"
    box = id_to_caption[chosen_id][1]
    img_id, region_id = chosen_id.split("_")
    img = PIL_Image.open(IMG_DATA_PATH + "VG_100K/" + str(img_id) + ".jpg")
    display(img)
    region = img.crop(box)
    region = region.resize([224, 224], PIL_Image.ANTIALIAS)
    display(region)


def display_img_from_url(url):
    response = requests.get(url, stream=True)
    with open("charpragcap/resources/img.jpg", "wb") as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response
    img = PIL_Image.open("charpragcap/resources/img.jpg")
    img = img.resize([224, 224], PIL_Image.ANTIALIAS)
    display(img)


def get_img(url):
    response = requests.get(url, stream=True)
    with open("charpragcap/resources/img.jpg", "wb") as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response
    img = PIL_Image.open("charpragcap/resources/img.jpg")
    img = img.resize([224, 224], PIL_Image.ANTIALIAS)
    display(img)
    rep = np.expand_dims(image.img_to_array(img), 0)

    rep = resnet(img_rep_layer).predict(img)
    return rep


def item_to_rep(item, id_to_caption):
    original_image = get_img_from_id(item, id_to_caption)
    original_image_vector = np.expand_dims(image.img_to_array(original_image), axis=0)
    input_image = ResNet50(img_rep_layer).predict(original_image_vector)
    return input_image

###TEXT UTILS


# convert caption into vector: SHAPE?
def vectorize_caption(sentence):
    if len(sentence) > 0 and sentence[-1] in list("!?."):
        sentence = sentence[:-1]
    sentence = start_token["char"] + sentence + stop_token["char"]
    sentence = list(sentence)
    while len(sentence) < max_sentence_length + 2:
        sentence.append(pad_token)

    caption_in = sentence[:-1]
    caption_out = sentence[1:]
    caption_in = np.asarray([char_to_index[x] for x in caption_in])
    caption_out = np.expand_dims(np.asarray([char_to_index[x] for x in caption_out]), 0)
    one_hot = np.zeros((caption_out.shape[1], len(sym_set)))
    one_hot[np.arange(caption_out.shape[1]), caption_out] = 1
    caption_out = one_hot
    return caption_in, caption_out


# takes (1,39,1) and returns string
def devectorize_caption(ary):
    # print("ARY",ary.shape,ary)
    return "".join([index_to_char[idx] for idx in np.squeeze(ary)])


# OTHER UTILS
def sentence_likelihood(img, sent):
    sentence = text_to_vecs(sent, words=True)
    # print(sentence.shape,img.shape)
    probs = s_zero.predict([img, sentence])
    probs = [x[word_to_index[sent[i + 1]]] for i, x in enumerate(probs[0][:-1])]
    # print(np.sum(np.log(probs)))


def largest_indices(self, ary, n):
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


# gives a train,val,test split, by way of three sets of id_to_caption keys


def split_dataset(trains=train_size, vals=val_size, tests=test_size):
    import pickle

    id_to_caption = pickle.load(open("charpragcap/resources/id_to_caption", "rb"))

    ids = sorted(list(id_to_caption))
    num_ids = len(ids)
    assert trains + vals + tests == 1.0

    num_train = int(num_ids * trains)
    num_val = num_train + int(num_ids * vals)
    num_test = num_val + int(num_ids * tests)

    trains, vals, tests = (
        ids[0:num_train],
        ids[num_train:num_val],
        ids[num_val:num_test],
    )
    return trains, vals, tests
