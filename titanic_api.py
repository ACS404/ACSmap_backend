## titanic_api.py — Titanic Flask REST API
 
from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from model.titanic import TitanicModel   # adjust import path if needed
 
titanic_api = Blueprint('titanic_api', __name__, url_prefix='/api/titanic')
api = Api(titanic_api)
 
 
class TitanicAPI:
 
    class _Predict(Resource):
        def post(self):
            """
            POST /api/titanic/predict
 
            Accepts a JSON body with passenger fields and returns survival probabilities.
 
            Request body (JSON):
            {
                "pclass":   2,          // 1, 2, or 3
                "sex":      "female",   // "male" or "female"
                "age":      28,         // numeric age
                "sibsp":    0,          // siblings/spouses aboard
                "parch":    0,          // parents/children aboard
                "fare":     30.00,      // ticket fare
                "embarked": "S",        // "C", "Q", or "S"
                "alone":    true        // boolean
            }
 
            Response (JSON):
            {
                "die":     0.2341,
                "survive": 0.7659
            }
            """
            body = request.get_json()
            if not body:
                return {'error': 'No JSON body provided'}, 400
 
            # Validate required fields
            required = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'alone']
            missing = [f for f in required if f not in body]
            if missing:
                return {'error': f'Missing fields: {missing}'}, 400
 
            try:
                titanicModel = TitanicModel.get_instance()
                response = titanicModel.predict(body)
                return jsonify(response)
            except Exception as e:
                return {'error': str(e)}, 500
 
    class _FeatureWeights(Resource):
        def get(self):
            """
            GET /api/titanic/weights
 
            Returns the feature importance scores from the Decision Tree model.
 
            Response (JSON):
            {
                "pclass":      0.08,
                "sex":         0.35,
                "age":         0.21,
                ...
            }
            """
            try:
                titanicModel = TitanicModel.get_instance()
                weights = titanicModel.feature_weights()
                return jsonify(weights)
            except Exception as e:
                return {'error': str(e)}, 500
 
    api.add_resource(_Predict, '/predict')
    api.add_resource(_FeatureWeights, '/weights')