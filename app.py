import hashlib
import os
import pickle
import sys
from collections import defaultdict

import cv2
import numpy as np
from flask import Flask, request
from flask_restful import Api, Resource, abort, reqparse

from face import face
from helpers import (file_digest, hash_files, vec2hash, vec2str, write_file,
                     write_frame)
from normalizeface import (align_face_to_template, get_face_landmarks,
                           normalize_faces)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/'
app.config['V1.0'] = '/api/1.0'
app.config['normalize'] = True
api = Api(app)

face_pickle = None
face_search_vectors = None
face_group_search = None
hash2file = None

parser = reqparse.RequestParser()


class return_frame(Resource):

    def get(self, file_hash, frame_number):
        ret_val = dict()

        print('return_frame', file_hash, frame_number)
        sys.stdout.flush()

        uri = None

        for i in hash2file:
            print(i, hash2file[i])
            sys.stdout.flush()

        try:
            uri = hash2file[file_hash]
        except Exception as e:
            abort(
                404,
                message='{0} file_hash {1} does not exist'.format(
                    e,
                    file_hash))

        if frame_number < 0:
            abort(404, message='frame_number {0} must be >0'.format(
                frame_number))

        video_file = cv2.VideoCapture(uri)
        video_length = int(video_file.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_number > video_length:
            abort(404, message='{0} > max length of vid {1}'.format(
                frame_number, video_length))

        video_file.set(1, frame_number)

        keep_going, img = video_file.read()

        if keep_going:

            meta = dict()
            meta['File_hash'] = file_hash
            meta['Frame_number'] = frame_number
            ret_val['Meta'] = meta
            ret_val['Frame'] = write_frame(file_hash, frame_number, img)
            # video_file.close()
            return ret_val

        else:
            # video_file.close()
            abort(404, message='frame decode error')


def handle_post_file():
    # get the image if it exists
    retval = []
    #loc = None
    #enc = None
    #h = None

    print('******************************')
    print('request.files:', request.files)
    print('request.files.keys:', list(request.files.keys()))
    sys.stdout.flush()

    for file_key in request.files.keys():

        # get the filename
        file = request.files[file_key]

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

        # get location of face so I can return it to gui.
        list_face_locs = face.face_locations(face_image, 2)
        enc = None

        if app.config['normalize']:
            # normalize
            list_face_encodings = normalize_faces(
                face_image, list_face_locs, 2)
            enc = list_face_encodings[0][0]
        else:
            #not normalize
            enc = face.face_encodings(face_image, list_face_locs)[0]

        loc = list_face_locs[0]
        print('enc:', enc)
        sys.stdout.flush()

        # make a reference to the vector as a loose hash to the file
        h = vec2hash(enc)
        temp = (loc, enc, h)
        retval.append(temp)
    return retval


class working(Resource):

    def get(self):
        return{'working': 'yes'}


class compare_2_uploads(Resource):

    def get(self, search_vector_name1, search_vector_name2):
        try:
            vector1 = face_search_vectors[search_vector_name1]
        except Exception as e:
            abort(
                404, message='{0} vector ref {1} does not exist'.format(
                    e, search_vector_name1))
        try:
            vector2 = face_search_vectors[search_vector_name2]
        except Exception as e:
            abort(
                404, message='{0} vector ref {1} does not exist'.format(
                    e, search_vector_name2))
        distance = face.face_distance(
            [np.array(vector1)], np.array(vector2))[0]
        d = dict()
        d['vector_1'] = search_vector_name1
        d['vector_2'] = search_vector_name2
        d['distance_euclidean'] = '{0}'.format(distance)
        return d


class make_group(Resource):

    def post(self, group_name):
        ret_val = list()
        args = parser.parse_args()
        print('args:', args)
        print('request:', request)
        sys.stdout.flush()

        loc_enc_h = handle_post_file()

        for loc, enc, h in loc_enc_h:
            cur_val = dict()
            cur_val['Found'] = 'False'
            cur_val['Name'] = 'None'
            cur_val['Group'] = 'False'

            if len is not None and len(enc) == 128:
                # valid data update the return
                cur_val['Found'] = 'True'
                cur_val['Group'] = 'True'
                cur_val['Name'] = h
                cur_val['Vec'] = list(enc)
                cur_val['Upload_coords'] = list(loc)
                face_group_search[group_name][h] = list(enc)
            ret_val.append(cur_val)
        # return back name of search vector
        return ret_val


class make_vector(Resource):

    def post(self):
        ret_val = list()
        args = parser.parse_args()
        print('args:', args)
        print('request:', request)
        sys.stdout.flush()

        loc_enc_h = handle_post_file()

        for loc, enc, h in loc_enc_h:
            cur_val = dict()
            cur_val['Found'] = 'False'
            cur_val['Name'] = 'None'

            if len is not None and len(enc) == 128:
                # valid data update the return
                cur_val['Found'] = 'True'
                cur_val['Name'] = h
                cur_val['Vec'] = list(enc)
                cur_val['Upload_coords'] = list(loc)
                face_search_vectors[h] = list(enc)
            ret_val.append(cur_val)
        # return back name of search vector
        return ret_val


class find_by_group(Resource):

    def get(self, group_name, distance):
        ret_val = list()

        try:
            group_data = face_group_search[group_name]
        except Exception as e:
            abort(404, message='{0} group {1} does not exist'.format(
                e, group_name))

        if distance > 1.0 or distance < 0:
            abort(
                404,
                message='distance {0} must be between [0,1]'.format(distance))

        print('group_name', group_name)
        print('distance', distance)
        sys.stdout.flush()

        for key in face_pickle:
            for group_key in group_data:
                entity = face_pickle[key]
                entity_vec = entity['face_vec']
                entity_pic = entity['face_pic']
                entity_videos = entity['videos']

                left = np.array(group_data[group_key])
                right = np.array(entity_vec)

                vector_distance = face.face_distance([left], right)[0]
                if vector_distance < distance:
                    print('found:', key)
                    sys.stdout.flush()
                    d = dict()
                    d['Name'] = key
                    d['Uri'] = write_file(entity)
                    d['Videos'] = entity_videos
                    d['Distance'] = vector_distance
                    ret_val.append(d)
        ret_val.sort(key=lambda temp_d: temp_d['Distance'])
        return ret_val


class find_by_vector(Resource):

    def get(self, search_vector_name, distance):

        ret_val = list()

        try:
            vector = face_search_vectors[search_vector_name]
        except Exception as e:
            abort(
                404, message='{0} searchvector {1} does not exist'.format(
                    e, search_vector_name))

        if distance > 1.0 or distance < 0:
            abort(
                404,
                message='distance {0} must be between [0,1]'.format(distance))

        for key in face_pickle:
            entity = face_pickle[key]
            entity_vec = entity['face_vec']
            entity_pic = entity['face_pic']
            entity_videos = entity['videos']
            left = np.array(face_search_vectors[search_vector_name])
            right = np.array(entity_vec)

            vector_distance = face.face_distance([left], right)[0]
            if vector_distance < distance:
                print('found:', key)
                sys.stdout.flush()
                d = dict()
                d['Name'] = key
                d['Uri'] = write_file(entity)
                d['Videos'] = entity_videos
                d['Distance'] = vector_distance
                ret_val.append(d)

        ret_val.sort(key=lambda temp_d: temp_d['Distance'])
        return ret_val


# TODO get rid of this one.
api.add_resource(
    find_by_vector,
    app.config['V1.0'] +
    '/find/<string:search_vector_name>/<float:distance>')

api.add_resource(working, app.config['V1.0'] + '/working')

# TODO use thisone soon
# api.add_resource(find_by_vector, app.config[
#                 'V1.0'] + '/findvector/<string:search_vector_name>/<float:distance>')

api.add_resource(
    compare_2_uploads,
    app.config['V1.0'] +
    '/compare2uploads/<string:search_vector_name1>/<string:search_vector_name2>')
api.add_resource(make_vector, app.config['V1.0'] + '/makevector')
api.add_resource(make_group, app.config[
                 'V1.0'] + '/makegroup/<string:group_name>')
api.add_resource(find_by_group, app.config[
                 'V1.0'] + '/findgroup/<string:group_name>/<float:distance>')
api.add_resource(return_frame, app.config['V1.0'] +
                 '/return_frame/<string:file_hash>/<int:frame_number>')

if __name__ == '__main__':
    # load the pickl file
    face_search_vectors = dict()
    face_group_search = defaultdict(dict)
    hash2file = hash_files('/mdata/*')
    face_pickle = pickle.load(open('./data/faces.pickle', 'rb'))
    app.run(debug=False, host='0.0.0.0')
