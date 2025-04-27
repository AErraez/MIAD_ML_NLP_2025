#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from model_deployment import predict_api


app = Flask(__name__)

# Definición API Flask
api = Api(
    app, 
    version='1.0', 
    title='Song Popularity Prediction API',
    description='Song Popularity Prediction API')

ns = api.namespace('predict', 
     description='Popularity predictor')

# Definición argumentos o parámetros de la API
parser = ns.parser()
parser.add_argument(
    'dance', 
    type=float, 
    required=True, 
    help='Danceability score (0 a 1)', 
    location='args')

parser.add_argument(
    'energy', 
    type=float, 
    required=True, 
    help='Energy score (0 a 1)', 
    location='args')

parser.add_argument(
    'speech', 
    type=float, 
    required=True, 
    help='Speechness score (0 a 1)', 
    location='args')

parser.add_argument(
    'loud', 
    type=float, 
    required=True, 
    help='Loudness in Decibels', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})



# Definición de la clase para disponibilización
@ns.route('/')
class PopularityApi(Resource):

    @ns.doc(parser=parser)
    @ns.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_api(args['dance'],args['energy'],args['speech'],args['loud'])
        }, 200


    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
