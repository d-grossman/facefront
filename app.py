from flask import Flask
from flask_restful import Resource, Api, abort
import pickle

app = Flask(__name__)
api = Api(app)

face_pickle = None
face_search_vectors = None

class working(Resource):
    def get(self):
        return{'working':'yes'}

class make_search_vector(Resource):
    def post(self):
        ret_val = dict()
        ret_val['Found']='False'
        ret_val['Name'] = 'None'
        args = parser.parse_args()
        #todo get image
        #todo process image into features
        #return back name of search vector
        return ret_val

class find_vectors(Resource):
    def get(self,search_vector_name,distance):
        try:
            vector = face_search_vectors[search_vector_name]        
        except Exception as e:
            abort(404, message='work {0} does not exist'.format(search_vector_name))

        if distance > 1.0 or distance < 0:
            abort(404, message='distance {0} must be between [0,1]'.format(search_vector_name))

        return {'vector':search_vector_name, 'distance':distance} 

api.add_resource(working,'/api/1.0/working')
api.add_resource(find_vectors,'/api/1.0/find/<string:search_vector_name>/<float:distance>')

if __name__ == '__main__':
    #load the pickl file
    face_search_vectors = dict()
    face_search_vectors['something'] = 'done'
    face_pickle = pickle.load(open('./data/faces.pickle','rb'))
    app.run(debug=True,host='0.0.0.0')
