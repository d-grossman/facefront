import glob
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
from helpers import (file_digest, hash_files, vec2hash, vec2str, write_file,
                     write_frame)
from normalizeface import align_face_to_template, get_face_landmarks

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/'
app.config['V1.0'] = '/api/1.0'
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

        for i in hash2file:
            print(i, hash2file[i])
            sys.stdout.flush()

        try:
            uri = hash2file[file_hash]
        except Exception as e:
            abort(
                404, message='file_hash {0} does not exist'.format(file_hash))

        if frame_number < 0:
            abort(404, message='frame_number {0} must be >0'.format(
                frame_number))

        video_file = cv2.VideoCapture(hash2file[file_hash])
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
            return ret_val
            video_file.close()

        else:
            video_file.close()
            abort(404, message='frame decode error')


def normalize_face(pic, places, jitters):
    ret_val = list()

    for place in places:
        top, right, bottom, left = place
        landmarks = get_face_landmarks(
            face.pose_predictor, pic, dlib.rectangle(left, top, right, bottom))
        # TODO make sure that 150 is the right size..
        adjusted_face = align_face_to_template(pic, landmarks, 150)
        # print('place',place)
        # print('adjusted_face',adjusted_face.shape)
        # sys.stdout.flush()
        #encoding = np.array(face.face_encodings( adjusted_face, [(0,0,150,150)], jitters) )
        #encoding = np.array(face.face_encodings( adjusted_face, [(150,150,0,0)], jitters) )
        encoding = np.array(face.face_encodings(
            adjusted_face, [(0, 150, 150, 0)], jitters))
        ret_val.append(encoding)

    return ret_val


def handle_post_file():
    # get the image if it exists
    enc = None
    h = None
    if 'data' in request.files:

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

        # get location of face so I can return it to gui.
        list_face_locs = face.face_locations(face_image)

        #not normalize
        enc = face.face_encodings(face_image, list_face_locs)[0]

        # normalize
        #list_face_locations = face.face_locations(face_image)
        #list_face_encodings = normalize_faces(face_image,list_face_locations,2)
        #enc = list_face_encodings[0][0]

        loc = list_face_locs[0]
        print('enc:', enc)
        sys.stdout.flush()

        # make a reference to the vector as a loose hash to the file
        h = vec2hash(enc)
    return loc, enc, h


class working(Resource):

    def get(self):
        return{'working': 'yes'}


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
        ret_val['Group'] = 'False'
        args = parser.parse_args()
        print('args:', args)
        print('request:', request)
        sys.stdout.flush()

        loc, enc, h = handle_post_file()

        if len is not None and len(enc) == 128:
            # valid data update the return
            ret_val['Found'] = 'True'
            ret_val['Group'] = 'True'
            ret_val['Name'] = h
            ret_val['Vec'] = list(enc)
            ret_val['Upload_coords'] = list(loc)
            face_group_search[group_name][h] = list(enc)
        # return back name of search vector
        return ret_val


class make_vector(Resource):

    def post(self):
        ret_val = dict()
        ret_val['Found'] = 'False'
        ret_val['Name'] = 'None'
        args = parser.parse_args()
        print('args:', args)
        print('request:', request)
        sys.stdout.flush()

        loc, enc, h = handle_post_file()

        if len is not None and len(enc) == 128:
            # valid data update the return
            ret_val['Found'] = 'True'
            ret_val['Name'] = h
            ret_val['Vec'] = list(enc)
            ret_val['Upload_coords'] = list(loc)
            face_search_vectors[h] = list(enc)
        # return back name of search vector
        return ret_val


class find_by_group(Resource):

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
            abort(404, message='searchvector {1} does not exist'.format(
                search_vector_name))

        if distance > 1.0 or distance < 0:
            abort(
                404, message='distance {0} must be between [0,1]'.format(distance))

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
api.add_resource(find_by_vector, app.config[
                 'V1.0'] + '/find/<string:search_vector_name>/<float:distance>')

api.add_resource(working, app.config['V1.0'] + '/working')

# TODO use thisone soon
# api.add_resource(find_by_vector, app.config[
#                 'V1.0'] + '/findvector/<string:search_vector_name>/<float:distance>')

api.add_resource(compare_2_uploads, app.config[
                 'V1.0'] + '/compare2uploads/<string:search_vector_name1>/<string:search_vector_name2>')
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
