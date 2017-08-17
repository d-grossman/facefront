import hashlib
import os
import pickle
import sys
from collections import defaultdict

import cv2
import numpy as np
from flask import Flask, request
from flask_restful import Api, Resource, abort, reqparse
from PIL import Image
from werkzeug.utils import secure_filename

from face import face

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/'
app.config['V1.0'] = '/api/1.0'
api = Api(app)

face_pickle = None
face_search_vectors = None
face_group_search = None

parser = reqparse.RequestParser()


def vec2str(v):
    ret_val = ''
    for x in v:
        ret_val += '{0}'.format(x)
    return ret_val


def vec2hash(v):
    v = vec2str(v)
    return hashlib.md5(v.encode("utf")).hexdigest()


class working(Resource):

    def get(self):
        return{'working': 'yes'}


def write_file(entity, prefix='static/'):
    img = Image.fromarray(np.roll(entity['pic'], 1, axis=-1))
    base_name = vec2hash(entity['face_vec'])
    filename = '{0}.{1}'.format(base_name, 'jpg')
    uri = os.path.join(prefix, filename)
    if not os.path.isfile(uri):
        img.save(uri)
    return uri


def handle_post_file():
    # get the image if it exists
    enc = None
    h = None
    if 'data' in request.files:
        # print('file found')
        # print('found,', request.files)
        # sys.stdout.flush()

        # get the filename
        file = request.files['data']
        # print('filename:', file.filename)
        # sys.stdout.flush()

        # keep the same extension on the file
        extension = file.filename.split('.')[-1]

        #@TODO add the time of the reference to keep things seperated
        filename_h = hashlib.md5(file.filename.encode("utf")).hexdigest()

        file.filename = '{0}.{1}'.format(filename_h, extension)
        print('filename:', file.filename)

        # save the file into the hash of the filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

        # read the file, find the face make the vector
        face_image = cv2.imread(os.path.join(
            app.config['UPLOAD_FOLDER'], file.filename))
        enc = face.face_encodings(face_image, None)[0]
        print('enc:', enc)
        sys.stdout.flush()

        # make a reference to the vector as a loose hash to the file
        h = vec2hash(enc)
    return enc, h


class compare_2_uploads(Resource):

    def get(self, search_vector_name1, search_vector_name2):
        try:
            vector1 = face_search_vectors[search_vector_name1]
        except Exception as e:
            abort(404, message='{0} vector ref {1} does not exist'.format(e,
                                                                          search_vector_name1))
        try:
            vector2 = face_search_vectors[search_vector_name2]
        except Exception as e:
            abort(404, message='{0} vector ref {1} does not exist'.format(e,
                                                                          search_vector_name2))
        distance = face.face_distance(
            [np.array(vector1)], np.array(vector2))[0]
        d = dict()
        d['vector_1'] = search_vector_name1
        d['vector_2'] = search_vector_name2
        d['distance_euclidean'] = '{0}'.format(distance)
        return d


class make_group(Resource):

    def post(self, group_name):
        ret_val = dict()
        ret_val['Found'] = 'False'
        ret_val['Name'] = 'None'
        args = parser.parse_args()
        print('args:', args)
        print('request:', request)
        sys.stdout.flush()

        enc, h = handle_post_file()

        if len(enc) == 128:
            # valid data update the return
            ret_val['Found'] = 'True'
            ret_val['Group'] = 'True'
            ret_val['Name'] = h
            ret_val['Vec'] = list(enc)
            face_group_search[group_name][h] = list(enc)
        # return back name of search vector
        return ret_val


class make_search_vector(Resource):

    def post(self):
        ret_val = dict()
        ret_val['Found'] = 'False'
        ret_val['Name'] = 'None'
        args = parser.parse_args()
        print('args:', args)
        print('request:', request)
        sys.stdout.flush()

        enc, h = handle_post_file()

        if len(enc) == 128:
            # valid data update the return
            ret_val['Found'] = 'True'
            ret_val['Name'] = h
            ret_val['Vec'] = list(enc)
            face_search_vectors[h] = list(enc)
        # return back name of search vector
        return ret_val


class find_group(Resource):

    def get(self, group_name, distance):
        ret_val = list()

        try:
            group_data = face_group_search[group_name]
        except Exception as e:
            abort(404, message='group {0} does not exist'.format(group_name))

        if distance > 1.0 or distance < 0:
            abort(
                404, message='distance {0} must be between [0,1]'.format(distance))

        print('group_name', group_name)
        print('distance', distance)
        sys.stdout.flush()

        for key in face_pickle:
            for group_key in group_data:
                entity = face_pickle[key]
                entity_vec = entity['face_vec']
                entity_pic = entity['pic']
                entity_times = entity['times']

                left = group_data[group_key]
                right = entity_vec

                vector_distance = face.face_distance([left], right)[0]
                if vector_distance < distance:
                    print('found:', key)
                    sys.stdout.flush()
                    d = dict()
                    d['Name'] = key
                    d['Uri'] = write_file(entity)
                    d['Distance'] = vector_distance
                    ret_val.append(d)
        ret_val.sort(key=lambda temp_d: temp_d['Distance'])
        return ret_val


class find_vectors(Resource):

    def get(self, search_vector_name, distance):

        ret_val = list()

        try:
            vector = face_search_vectors[search_vector_name]
        except Exception as e:
            abort(404, message='{0} work {1} does not exist'.format(e,
                                                                    search_vector_name))

        if distance > 1.0 or distance < 0:
            abort(404, message='distance {0} must be between [0,1]'.format(
                distance))

        for key in face_pickle:
            entity = face_pickle[key]
            entity_vec = entity['face_vec']
            entity_pic = entity['pic']
            entity_times = entity['times']
            left = face_search_vectors[search_vector_name]
            right = entity_vec

            vector_distance = face.face_distance([left], right)[0]
            if vector_distance < distance:
                print('found:', key)
                sys.stdout.flush()
                d = dict()
                d['Name'] = key
                d['Uri'] = write_file(entity)
                d['Distance'] = vector_distance
                ret_val.append(d)

        ret_val.sort(key=lambda temp_d: temp_d['Distance'])
        return ret_val

        # return {'vector': search_vector_name, 'distance': distance}

api.add_resource(working, app.config['V1.0'] + '/working')
api.add_resource(find_vectors, app.config[
                 'V1.0'] + '/find/<string:search_vector_name>/<float:distance>')
api.add_resource(compare_2_uploads, app.config[
                 'V1.0'] + '/compare2uploads/<string:search_vector_name1>/<string:search_vector_name2>')
api.add_resource(make_search_vector, app.config['V1.0'] + '/makevector')
api.add_resource(make_group, app.config[
                 'V1.0'] + '/makegroup/<string:group_name>')
api.add_resource(find_group, app.config[
                 'V1.0'] + '/findgroup/<string:group_name>/<float:distance>')

if __name__ == '__main__':
    # load the pickl file
    face_search_vectors = dict()
    face_group_search = defaultdict(dict)
    face_pickle = pickle.load(open('./data/faces.pickle', 'rb'))
    app.run(debug=False, host='0.0.0.0')
