#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import libraries
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import json

from flask_restful import Resource, Api
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# import image processing
import sys
sys.path.insert(0, '../')
import image_utils
from image_utils import crop_image, normalize_image, convert_to_rgb

# import pytorch
import torch
from timm.data import create_loader, resolve_data_config, ImageDataset
from timm.models import create_model
import uuid
import os
import cv2


# Dictionary with label codes
label_dict = {0:'cannon',1:'eye', 2:'face', 3:'nail', 4:'pear',
              5:'piano',6:'radio', 7:'spider', 8:'star', 9:'sword',
              10:'airplane', 11:'axe', 12:'banana', 13:'basketball', 14:'cake', 
              15:'donut', 16:'flower', 17:'guitar', 18:'house', 19:'rainbow'}

# categories = [
#             'cannon', 'eye', 'face', 'nail', 'pear', 'piano', 'radio', 'spider', 'star', 'sword', 
#             'airplane', 'axe', 'banana', 'basketball', 'cake', 'donut', 'flower', 'guitar', 'house', 'rainbow'
#             ]

categories = ['aircraft carrier', 'airplane', 'alarm clock', 'ambulance', 'angel', 'animal migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday cake', 'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling fan', 'cello', 'cell phone', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise ship', 'cup', 'diamond', 'dishwasher', 'diving board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip flops', 'floor lamp', 'flower', 'flying saucer', 'foot', 'fork', 'frog', 'frying pan', 'garden', 'garden hose', 'giraffe', 'goatee', 'golf club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey puck', 'hockey stick', 'horse', 'hospital', 'hot air balloon', 'hot dog', 'hot tub', 'hourglass', 'house', 'house plant', 'hurricane', 'ice cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'light bulb', 'lighter', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paintbrush', 'paint can', 'palm tree', 'panda', 'pants', 'paper clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup truck', 'picture frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police car', 'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote control', 'rhinoceros', 'rifle', 'river', 'roller coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school bus', 'scissors', 'scorpion', 'screwdriver', 'sea turtle', 'see saw', 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping bag', 'smiley face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop sign', 'stove', 'strawberry', 'streetlight', 'string bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing set', 'sword', 'syringe', 'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis racquet', 'tent', 'The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 't-shirt', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing machine', 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine bottle', 'wine glass', 'wristwatch', 'yoga', 'zebra',
            'zigzag']

def load_model(architecture='mobilenetv3_large_100', filepath = '/root/quick-draw-image-recognition/checkpoint-161.pth.tar'):
    model = create_model(
                        architecture,
                        num_classes=345,
                        in_chans=3,
                        pretrained=True,
                        checkpoint_path=filepath
                            )
    model.eval()
    config = resolve_data_config({}, model=model)
    return model, config

def get_prediction(model, config, data_dir):

    loader = create_loader(
        ImageDataset(data_dir),
        input_size=[3, 28, 28],
        batch_size=32,
        use_prefetcher=True,
        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
        num_workers=2,
        crop_pct=config['crop_pct'])

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            input = input.cpu()
            target = target.cpu()
            labels = model(input)
            print(labels.topk(5))
            topk = labels.topk(5)[1]
    lbl_list = []
    for label in topk[0]:
        lbl_list.append(categories[label.item()])

    return lbl_list

app = Flask(__name__)

# load model
model, config = load_model()
main_dir = 'drawn_images'
# index webpage receives user input for the model
@app.route('/')
@app.route('/index')
def index():
    # render web page
    return render_template('index.html')

@app.route('/go/<dataURL>')
def pred(dataURL):
    """
    Render prediction result.
    """

    # decode base64  '._-' -> '+/='
    dataURL = dataURL.replace('.', '+')
    dataURL = dataURL.replace('_', '/')
    dataURL = dataURL.replace('-', '=')

    # get the base64 string
    image_b64_str = dataURL
    # convert string to bytes
    byte_data = base64.b64decode(image_b64_str)
    image_data = BytesIO(byte_data)
    # open Image with PIL
    img = Image.open(image_data)
    # img.save('test2.jpg')
    # save original image as png (for debugging)

    #img.save('image' + str(ts) + '.png', 'PNG')

    # convert image to RGBA
    img = img.convert("RGBA")

    # preprocess the image for the model
    image_cropped = crop_image(img) # crop the image and resize to 28x28
    image_normalized = normalize_image(image_cropped) # normalize color after crop
    
    # convert image from RGBA to RGB
    img_rgb = convert_to_rgb(image_normalized)
    # img_rgb.save('test2.jpg')
    pix = np.array(img_rgb)
    uid_main = str(uuid.uuid4())
    uid_sub = str(uuid.uuid4())
    data_dir = os.path.join(main_dir, uid_main, uid_sub)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    cv2.imwrite(os.path.join(data_dir, 'drawn_image.jpg'), np.invert(pix))

    lbl_list = get_prediction(model, config, data_dir)

    # render the hook.html passing prediction resuls
    return render_template(
        'hook.html',
        result = lbl_list, # predicted class label
        dataURL = dataURL # image to display with result
    )

def main():
    app.run(host='0.0.0.0', port=3002, debug=True)


if __name__ == '__main__':
    main()
