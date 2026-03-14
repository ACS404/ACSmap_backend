"""
Cancer Risk Prediction API
Flask REST API endpoints for ML-based cancer risk predictions
"""

from flask import Blueprint, request
from flask_restful import Api, Resource

from model.cancer_risk import CancerRiskModel

cancer_risk_api = Blueprint('cancer_risk_api', __name__, url_prefix='/api/cancer-risk')
api = Api(cancer_risk_api)


class CancerRiskAPI:
    """API endpoints for cancer risk prediction."""

    class _Predict(Resource):
        """Predict cancer risk based on patient demographics and lifestyle factors."""

        def post(self):
            try:
                patient_data = request.get_json()

                if not patient_data:
                    return {'error': 'No patient data provided'}, 400

                required_fields = [
                    'age', 'sex', 'race', 'smoking_status', 'bmi_category',
                    'alcohol_consumption', 'physical_activity', 'diet_quality',
                    'family_history', 'diabetes', 'hepatitis'
                ]

                missing_fields = [f for f in required_fields if f not in patient_data]
                if missing_fields:
                    return {'error': f'Missing required fields: {", ".join(missing_fields)}'}, 400

                model = CancerRiskModel.get_instance()
                prediction = model.predict(patient_data)
                risk_factors = model.get_risk_factors(patient_data)
                importances = model.feature_importances()

                return {
                    **prediction,
                    'risk_factors': risk_factors,
                    'feature_importances': importances
                }, 200

            except KeyError as e:
                return {'error': f'Invalid value for field: {str(e)}'}, 400
            except Exception as e:
                return {'error': f'Prediction failed: {str(e)}'}, 500

    class _FeatureImportances(Resource):
        """Get feature importance scores from the model."""

        def get(self):
            try:
                model = CancerRiskModel.get_instance()
                importances = model.feature_importances()

                return {
                    'feature_importances': importances,
                    'sorted_features': sorted(
                        importances.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                }, 200

            except Exception as e:
                return {'error': f'Failed to get importances: {str(e)}'}, 500

    class _ModelInfo(Resource):
        """Get information about the trained model."""

        def get(self):
            try:
                model = CancerRiskModel.get_instance()

                return {
                    'model_type': 'Ensemble (Logistic Regression + Random Forest)',
                    'features': model.features,
                    'training_samples': len(model.training_data),
                    'data_source': 'ACS Cancer Facts & Figures 2026',
                    'description': 'ML model for predicting cancer risk based on demographic and lifestyle factors',
                    'risk_categories': ['low', 'high'],
                    'supported_values': {
                        'sex': ['male', 'female'],
                        'race': ['white', 'black', 'hispanic', 'aian', 'aapi'],
                        'smoking_status': ['never', 'former', 'current'],
                        'bmi_category': ['normal', 'overweight', 'obese', 'severely-obese'],
                        'alcohol_consumption': ['none', 'light', 'moderate', 'heavy'],
                        'physical_activity': ['sedentary', 'moderate', 'active'],
                        'diet_quality': ['poor', 'average', 'healthy']
                    }
                }, 200

            except Exception as e:
                return {'error': f'Failed to get model info: {str(e)}'}, 500


# Register API endpoints
api.add_resource(CancerRiskAPI._Predict, '/predict')
api.add_resource(CancerRiskAPI._FeatureImportances, '/feature-importances')
api.add_resource(CancerRiskAPI._ModelInfo, '/model-info')