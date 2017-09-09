import hashlib
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
from flask import Flask, request

import cv2
from face import face
from flask_restful import Api, Resource, abort, reqparse
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

        if file_hash in hash2file:
            uri = hash2file[file_hash]['Location']
        else:
            abort(
                404,
                message='file_hash {1} does not exist'.format(file_hash))

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
            meta['file_hash'] = file_hash
            meta['frame_number'] = frame_number
            ret_val['meta'] = meta
            ret_val['frame'] = write_frame(file_hash, frame_number, img)
            # video_file.close()
            return ret_val

        else:
            # video_file.close()
            abort(404, message='frame decode error')


def handle_post_file():
    # get the image if it exists
    retval = []

    # print('******************************')
    #print('request.files:', request.files)
    #print('request.files.keys:', list(request.files.keys()))
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
        print('enc_len:', len(enc))
        sys.stdout.flush()

        # make a reference to the vector as a loose hash to the file
        h = vec2hash(enc)
        temp = (loc, enc, h)
        retval.append(temp)
    return retval


class return_feeds(Resource):

    def get(self):
        ret_val = dict()
        meta = dict()
        results = list()
        m_count = dict()

        m_count['count'] = len(hash2file.keys())
        meta['result_set'] = m_count

        ret_val['meta'] = meta
        for h in hash2file:
            val = dict()
            val['name'] = hash2file[h]['Name']
            val['hash'] = hash2file[h]['Hash']
            val['location'] = hash2file[h]['Location']
            val['uri'] = '/static/' + val['name']
            results.append(val)
        ret_val['results'] = results
        return ret_val


class working(Resource):

    def get(self):
        return{'working': 'yes'}


class make_results_comparisons(Resource):

    def post(self):
        ret_val = {}

        loc_enc_h = handle_post_file()
        if len(loc_enc_h) != 2:
            abort(404, message='expected exactly 2 faces, found {0}'.format(
                len(loc_enc_h)))

        ret_val['meta'] = {}
        ret_val['meta']['vector_set'] = {}
        ret_val['meta']['vector_set']['count'] = len(loc_enc_h)
        ret_val['meta']['vector_set']['vectors'] = []
        for loc, enc, h in loc_enc_h:
            d = {}
            d['hash'] = h
            d['face_coordinates'] = list(loc)
            d['vector'] = list(enc)
            ret_val['meta']['vector_set']['vectors'].append(d)

        left = np.array(loc_enc_h[0][1])
        right = np.array(loc_enc_h[1][1])
        ret_val['results'] = {}
        ret_val['results']['distance'] = face.face_distance([left], right)[0]

        return ret_val


class make_results_matches(Resource):

    def post(self):
        ret_val = {}
        query = {}
        meta = {}

        dist_name = 'threshold'

        #args = parser.parse_args()
        #print('args:', args)
        #print('request:', request)
        #print('request.data:', request.data)
        #print('request.form:', request.form)

        if dist_name not in request.form:
            abort(404, message='threshold not specified')

        distance = float(request.form[dist_name])

        if distance > 1.0 or distance < 0:
            abort(
                404,
                message='threshold {0} must be between [0,1]'.format(distance))
        sys.stdout.flush()

        query['feeds'] = self.make_feeds()
        query['threshold'] = distance
        meta['query'] = query

        loc_enc_h = handle_post_file()
        meta['vector_set'] = self.make_vector_set(loc_enc_h)

        ret_val['meta'] = meta
        ret_val['results'] = self.make_result_array(meta)

        meta['result_set'] = self.make_result_set(ret_val['results'])

        #print('len:', len(loc_enc_h))
        #print('threshold:', distance)
        sys.stdout.flush()

        return ret_val

    @classmethod
    def make_result_set(cls, results):
        result_set = {}

        # print('*********************')
        # print(results)
        # print('*********************')

        frame_count = 0
        result_count = 0
        for result in results:
            #print('result:', result)
            #print('results[\'video\']', result['videos'])
            for video in result['videos']:
                result_count += 1
                frame_count += len(video['frames'])
                # print('result_count:',result_count)
                # sys.stdout.flush()
        result_set['matches'] = frame_count
        result_set['count'] = result_count

        return result_set

    @classmethod
    def make_feeds(cls):
        feeds = {}
        for h in hash2file:
            val = {}
            val['name'] = hash2file[h]['Name']
            feeds[hash2file[h]['Hash']] = val
        return feeds

    @classmethod
    def make_vector_set(cls, loc_enc_h):
        vector_set = {}
        vectors = []

        for loc, enc, h in loc_enc_h:
            if enc is not None and len(enc) == 128:
                entity = {}
                entity['hash'] = h
                entity['face_coordinates'] = loc
                entity['vector'] = list(enc)
                vectors.append(entity)
        vector_set['count'] = len(vectors)
        vector_set['vectors'] = vectors
        return vector_set

    @classmethod
    def make_result_array(cls, meta):

        distance = meta['query']['threshold']
        #print('meta.keys:', meta.keys())
        # sys.stdout.flush()
        vectors = meta['vector_set']['vectors']

        video_array = []

        for entity in vectors:
            query_src_hash = entity['hash']
            query_src_enc = np.array(entity['vector'])

            for key in face_pickle:
                s_entity = face_pickle[key]
                s_vec = np.array(s_entity['face_vec'])
                #s_pic = s_entity['face_pic']
                s_videos = s_entity['videos']

                left = query_src_enc
                right = s_vec

                vector_distance = face.face_distance([left], right)[0]
                if vector_distance < distance:
                    entry = {}
                    entry['distance'] = vector_distance
                    entry['src'] = query_src_hash
                    entry['uri'] = write_file(s_entity)
                    entry['hash'] = entry['uri'].split('/')[-1].split('.')[0]
                    entry['videos'] = cls.proc_videos(s_videos)
                    video_array.append(entry)

            video_array.sort(key=lambda temp_d: temp_d['distance'])

            # print('videoArray*********************************')
            # print(video_array)
            # print('videoArray*********************************')

            return video_array

    @classmethod
    def proc_videos(cls, videos):
        video_list = []
        # print('videos:',videos)
        sys.stdout.flush()
        for source in videos:
            vid_d = {}
            vid_d['hash'] = source
            frames = []

            # print('videos[src]:',videos[source])
            for frame, face_box in videos[source]:
                frame_d = {}
                frame_d['id'] = frame
                frame_d['face_coordinates'] = list(face_box)
                frames.append(frame_d)
            vid_d['frames'] = frames
            video_list.append(vid_d)

        return video_list


api.add_resource(return_frame, app.config['V1.0'] +
                 '/frames/<string:file_hash>/<int:frame_number>')
api.add_resource(return_feeds, app.config['V1.0'] + '/feeds')
api.add_resource(working, app.config['V1.0'] + '/working')
api.add_resource(make_results_matches, app.config['V1.0'] + '/results/matches')
api.add_resource(make_results_comparisons,
                 app.config['V1.0'] + '/results/comparisons')

if __name__ == '__main__':
    # load the pickl file
    face_search_vectors = dict()
    face_group_search = defaultdict(dict)
    hash2file = hash_files('/mdata/*')
    face_pickle = pickle.load(open('./data/faces.pickle', 'rb'))
    app.run(debug=False, host='0.0.0.0')
