#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import libraries
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import json

from flask_restful import Resource, Api
from flask import Flask, request, jsonify
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

categories = [
            'cannon', 'eye', 'face', 'nail', 'pear', 'piano', 'radio', 'spider', 'star', 'sword', 
            'airplane', 'axe', 'banana', 'basketball', 'cake', 'donut', 'flower', 'guitar', 'house', 'rainbow'
            ]

def load_model(architecture='mobilenetv3_large_100', filepath = '/root/quick-draw-image-recognition/checkpoint-2.pth.tar'):
    model = create_model(
                        architecture,
                        num_classes=20,
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
            topk = labels.topk(5)[1]
    lbl_list = []
    for label in topk[0]:
        lbl_list.append(categories[label.item()])

    return lbl_list

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)

# load model
model, config = load_model()
main_dir = 'drawn_images'

class CApp(Resource):
    @staticmethod
    def get(dataURL):
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

        return jsonify({'predictions':lbl_list})

api.add_resource(CApp, '/<path:dataURL>')

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
