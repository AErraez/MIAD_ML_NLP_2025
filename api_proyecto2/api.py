from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from model_deployment import predict_api
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

app = Flask(__name__)

# Definición API Flask
api = Api(
    app, 
    version='1.0', 
    title='Movie Gender Classification API',
    description='Movie Gender Classification  API')

ns = api.namespace('predict', 
     description='Gender Classificator')

# Definición argumentos o parámetros de la API
parser = ns.parser()
parser.add_argument(
    'title', 
    type=str, 
    required=True, 
    help='Title of the movie', 
    location='args')

parser.add_argument(
    'plot', 
    type=str, 
    required=True, 
    help='Plot of the movie', 
    location='args')


resource_fields = api.model('Resource', {
    'result': fields.Raw,
})

# Definición de la clase para disponibilización
@ns.route('/')
class PopularityApi(Resource):

    @ns.doc(parser=parser)
    @ns.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_api(args['title'],args['plot'])
        }, 200
    
app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)