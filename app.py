import hashlib
import pickle
import sys
import os

import cv2
from flask import Flask, request
from flask_restful import Api, Resource, abort, reqparse
from werkzeug.utils import secure_filename

from face import face

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/'
app.config['V1.0'] = '/api/1.0'
api = Api(app)

face_pickle = None
face_search_vectors = None

parser = reqparse.RequestParser()


def vec2str(v):
    ret_val = ''
    for x in v:
        ret_val += '{0}'.format(x)
    return ret_val


class working(Resource):

    def get(self):
        return{'working': 'yes'}


class make_search_vector(Resource):

    def post(self):
        ret_val = dict()
        ret_val['Found'] = 'False'
        ret_val['Name'] = 'None'
        args = parser.parse_args()
        print('args:', args)
        print('request:', request)
        sys.stdout.flush()

        #get the image if it exists
        if 'data' in request.files:
            #print('file found')
            #print('found,', request.files)
            #sys.stdout.flush()

            #get the filename
            file = request.files['data']
            #print('filename:', file.filename)
            #sys.stdout.flush()

            #keep the same extension on the file
            extension = file.filename.split('.')[-1]

            #@TODO add the time of the reference to keep things seperated
            filename_h= hashlib.md5(file.filename.encode("utf")).hexdigest()

            file.filename = '{0}.{1}'.format(filename_h,extension)
            print('filename:', file.filename)

            #save the file into the hash of the filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],file.filename))

            #read the file, find the face make the vector
            face_image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'],file.filename))
            enc = face.face_encodings(face_image, None)[0]
            print('enc:', enc)
            sys.stdout.flush()
            vec_str = vec2str(enc)

            #make a reference to the vector as a loose hash to the file
            h = hashlib.md5(vec_str.encode("utf")).hexdigest()

            if len(enc) == 128:
                #valid data update the return
                ret_val['Found'] = 'True'
                ret_val['Name'] = h
                ret_val['vec'] = list(enc)
                face_search_vectors[h] = list(enc)
        # return back name of search vector
        return ret_val


class find_vectors(Resource):

    def get(self, search_vector_name, distance):

        ret_val = list()

        try:
            vector = face_search_vectors[search_vector_name]
        except Exception as e:
            abort(404, message='work {0} does not exist'.format(
                search_vector_name))

        if distance > 1.0 or distance < 0:
            abort(404, message='distance {0} must be between [0,1]'.format(
                search_vector_name))

        for key in face_pickle:
            entity = face_pickle[key]
            entity_vec = entity['face_vec']
            entity_pic = entity['pic']
            entity_times = entity['times']
            left = face_search_vectors[search_vector_name]
            right = entity_vec
            if (face.face_distance([left],right)) < distance:
                print('found:',key)
                sys.stdout.flush()
                ret_val.append(key)

        return ret_val

        #return {'vector': search_vector_name, 'distance': distance}

api.add_resource(working, app.config['V1.0'] + '/working')
api.add_resource(find_vectors, app.config[
                 'V1.0'] + '/find/<string:search_vector_name>/<float:distance>')
api.add_resource(make_search_vector, app.config['V1.0'] + '/makevector')

if __name__ == '__main__':
    # load the pickl file
    face_search_vectors = dict()
    face_pickle = pickle.load(open('./data/faces.pickle', 'rb'))
    app.run(debug=True, host='0.0.0.0')

