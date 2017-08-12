from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class working(Resource):
    def get(self):
        return{'working':'yes'}

api.add_resource(working,'/')

if __name__ == '__main__':
    #TODO load the pickl file
    app.run(debug=True,host='0.0.0.0')
