
# example of URL http://localhost:5000/classify_url?imageurl=https://dl.dropboxusercontent.com/u/67895186/69.png&callback=test

import os
import time
import cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image
import cStringIO as StringIO
#import urllib2
import time


import json

import caffe

VIOLENT_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/models/violent')
WEATHER_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/models/weather')
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])
WEATHER_LABELS=['Cloudy','DawnDusk','Day','Foggy','Indoors','Night','Outdoors','Rainy','Sunny']

# Obtain the flask app object
app = flask.Flask(__name__)

image_dim=227

@app.route('/classify_violent', methods=['GET'])
def classify_violent():
    #imageurl = flask.request.args.get('imageurl', '')
    imagepath = flask.request.args.get('imagepath', '')
    print(imagepath)
    callback = flask.request.args.get('callback', '')
    try:
        start = time.time()
        #tmpDownloaded = urllib2.urlopen(imageurl).read()
        end = time.time()
        time1 = end - start
        #string_buffer = StringIO.StringIO(tmpDownloaded)
        im = Image.open(imagepath)
        #im.save("disp.png")

        S = im.size
        if S[1] > S[0]:
            ratio = float(S[1]) / image_dim
            im = im.resize((int(round(S[0] / ratio)), image_dim))
            SNew = im.size
            newIm = Image.new("RGB", (SNew[0], SNew[0]))
            newIm.paste(im, (0, (SNew[0] - SNew[1]) / 2))
        else:
            ratio = float(S[0]) / image_dim
            im = im.resize((image_dim, int(round(S[1] / ratio))))
            SNew = im.size
            newIm = Image.new("RGB", (SNew[1], SNew[1]))
            newIm.paste(im, ((SNew[1] - SNew[0]) / 2, 0))
        im=newIm

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('Image open error: %s', err)
        return str(-10000)
    #string_buffer.close()



    start = time.time()
    im.save("tmp.png")
    image = caffe.io.load_image("tmp.png")
    result = app.clf_v.classify_violent_image(image)
    output=str(result[1][1])
    end = time.time()
    time2=end-start

    d = json.dumps(dict(prediction=output,time1=time1,time2=time2))
    #return callback + '(' + d + ');'
    return output


@app.route('/classify_weather', methods=['GET'])
def classify_weather():
    imageurl = flask.request.args.get('imageurl', '')
    callback = flask.request.args.get('callback', '')
    try:
        string_buffer = StringIO.StringIO(
            urllib.urlopen(imageurl).read())
        im=Image.open(string_buffer)

        S = im.size
        if S[0] > S[1]:
            ratio = float(S[1]) / image_dim
            im = im.resize((int(round(S[0] / ratio)), image_dim))
            largeDim = 0
        else:
            ratio = float(S[0]) / image_dim
            im = im.resize((image_dim, int(round(S[1] / ratio))))
            largeDim = 1

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return callback + '(' + "-10000" + ');'

    string_buffer.close()

    probs=np.array(np.repeat(-float('inf'),len(WEATHER_LABELS)))
    confidences=np.array([])
    if largeDim == 0:
        for coord1 in range(0, S[0], int(np.floor(image_dim / 2))):
            imCrop = im.crop((coord1, 0, coord1 + image_dim, image_dim))
            imCrop.save("tmp.png")
            image = caffe.io.load_image("tmp.png")
            result = app.clf_w.classify_weather_image(image)
            probs=np.c_[probs,np.array(result[1])]
        imCrop = im.crop((S[0] - image_dim, 0, S[0], image_dim))
        imCrop.save("tmp.png")
        image = caffe.io.load_image("tmp.png")
        result = app.clf_w.classify_weather_image(image)
        probs = np.c_[probs, np.array(result[1])]
        output=np.amax(probs,1)
    else:
        for coord2 in range(0, S[1], int(np.floor(image_dim / 2))):
            imCrop = im.crop((0, coord2, image_dim, coord2 + image_dim))
            imCrop.save("tmp.png")
            image = caffe.io.load_image("tmp.png")
            result = app.clf_w.classify_weather_image(image)
            probs=np.c_[probs,np.array(result[1])]
        imCrop = im.crop((0, S[1] - image_dim, image_dim, S[1]))
        imCrop.save("tmp.png")
        image = caffe.io.load_image("tmp.png")
        result = app.clf_w.classify_weather_image(image)
        probs = np.c_[probs, np.array(result[1])]
        output = np.amax(probs, 1)
    #logging.info('Image: %s', imageurl)

    d = json.dumps(dict(zip(WEATHER_LABELS, output.T)))
    return callback + '(' + d + ');'


class ImagenetClassifier_Violent(object):
    default_args = {
        'model_def_file': (
            '{}/deploy.prototxt'.format(VIOLENT_DIRNAME)),
        'pretrained_model_file': (
            '{}/storedModel.caffemodel'.format(VIOLENT_DIRNAME)),
        'mean_file': (
            '{}/imageMean.npy'.format(VIOLENT_DIRNAME)),
    }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['image_dim'] = image_dim
    default_args['raw_scale'] = image_dim

    def __init__(self, model_def_file, pretrained_model_file, mean_file,
                 raw_scale, image_dim, gpu_mode):
        logging.info('Loading net and associated files...')
        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self.net = caffe.Classifier(
            model_def_file, pretrained_model_file,
            image_dims=(image_dim, image_dim), raw_scale=raw_scale,
            mean=np.load(mean_file).mean(1).mean(1), channel_swap=(2, 1, 0)
        )

    def classify_violent_image(self, image):
        try:
            starttime = time.time()
            scores = self.net.predict([image], oversample=True).flatten()
            endtime = time.time()

            indices = (-scores).argsort()[:5]

            return (True, scores, indices, '%.3f' % (endtime - starttime))

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')

class ImagenetClassifier_Weather(object):
    default_args = {
        'model_def_file': (
            '{}/deploy.prototxt'.format(WEATHER_DIRNAME)),
        'pretrained_model_file': (
            '{}/storedModel.caffemodel'.format(WEATHER_DIRNAME)),
        'mean_file': (
            '{}/imageMean.npy'.format(WEATHER_DIRNAME)),
    }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['image_dim'] = image_dim
    default_args['raw_scale'] = image_dim

    def __init__(self, model_def_file, pretrained_model_file, mean_file,
                 raw_scale, image_dim, gpu_mode):
        logging.info('Loading net and associated files...')
        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self.net = caffe.Classifier(
            model_def_file, pretrained_model_file,
            image_dims=(image_dim, image_dim), raw_scale=raw_scale,
            mean=np.load(mean_file).mean(1).mean(1), channel_swap=(2, 1, 0)
        )

    def classify_weather_image(self, image):
        try:
            starttime = time.time()
            scores = self.net.predict([image], oversample=True).flatten()
            endtime = time.time()


            return (True, scores, '%.3f' % (endtime - starttime))

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()
    ImagenetClassifier_Violent.default_args.update({'gpu_mode': opts.gpu})
    ImagenetClassifier_Weather.default_args.update({'gpu_mode': opts.gpu})

    # Initialize classifier + warm start by forward for allocation
    app.clf_v = ImagenetClassifier_Violent(**ImagenetClassifier_Violent.default_args)
    app.clf_v.net.forward()

    app.clf_w = ImagenetClassifier_Weather(**ImagenetClassifier_Weather.default_args)
    app.clf_w.net.forward()

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    start_from_terminal(app)

