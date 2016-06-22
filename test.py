from PIL import Image
import cStringIO as StringIO
import urllib
import caffe

import numpy as np

image_dim=227

string_buffer = StringIO.StringIO(
    urllib.urlopen("https://media.giphy.com/media/BzHMb7KgFi4c8/giphy-facebook_s.jpg").read())
im = Image.open(string_buffer)

S = im.size
if S[0] > S[1]:
    ratio = float(S[1]) / image_dim
    im = im.resize((int(round(S[0] / ratio)), image_dim))
    largeDim = 0
else:
    ratio = float(S[0]) / image_dim
    im = im.resize((image_dim, int(round(S[1] / ratio))))
    largeDim = 1


if largeDim == 0:
    for coord1 in range(0, S[0], int(np.floor(image_dim / 2))):
        imCrop=im.crop((coord1,0,coord1+image_dim,image_dim))
        imCrop.save("tmp.png")
        image = caffe.io.load_image("tmp.png")
    imCrop = im.crop((S[0]-image_dim, 0, S[0], image_dim))
    imCrop.save("tmp.png")
    image = caffe.io.load_image("tmp.png")
else:
    for coord2 in range(0, S[1], int(np.floor(image_dim / 2))):
        imCrop=im.crop((0,coord2,image_dim,coord2+image_dim))
        imCrop.save("tmp.png")
        image = caffe.io.load_image("tmp.png")
    imCrop = im.crop((0, S[1] - image_dim, image_dim, S[1]))
    imCrop.save("tmp.png")
    image = caffe.io.load_image("tmp.png")

