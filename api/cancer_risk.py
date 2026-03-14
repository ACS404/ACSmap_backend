"""
Cancer Risk Prediction API
Flask REST API endpoints for ML-based cancer risk predictions
"""

from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource

# Import will be uncommented when integrated into main app
from model.cancer_risk import CancerRiskModel

cancer_risk_api = Blueprint('cancer_risk_api', __name__, url_prefix='/api/cancer-risk')
api = Api(cancer_risk_api)


class CancerRiskAPI:
    """API endpoints for cancer risk prediction."""
    
    class _Predict(Resource):
        """Predict cancer risk based on patient demographics and lifestyle factors."""
        
        def post(self):
            """POST endpoint for cancer risk prediction.
            
            Expects JSON body with patient data:
            {
                "age": int,
                "sex": "male" | "female",
                "race": "white" | "black" | "hispanic" | "aian" | "aapi",
                "smoking_status": "never" | "former" | "current",
                "pack_years": float,
                "bmi_category": "normal" | "overweight" | "obese" | "severely-obese",
                "alcohol_consumption": "none" | "light" | "moderate" | "heavy",
                "physical_activity": "sedentary" | "moderate" | "active",
                "diet_quality": "poor" | "average" | "healthy",
                "family_history": boolean,
                "diabetes": boolean,
                "hepatitis": boolean
            }
            
            Returns JSON with:
            {
                "low_risk_probability": float,
                "high_risk_probability": float,
                "risk_category": "low" | "high",
                "model_confidence": float,
                "risk_factors": [...],
                "feature_importances": {...}
            }
            """
            try:
                # Get patient data from request
                patient_data = request.get_json()
                
                if not patient_data:
                    return jsonify({'error': 'No patient data provided'}), 400
                
                # Validate required fields
                required_fields = [
                    'age', 'sex', 'race', 'smoking_status', 'bmi_category',
                    'alcohol_consumption', 'physical_activity', 'diet_quality',
                    'family_history', 'diabetes', 'hepatitis'
                ]
                
                missing_fields = [field for field in required_fields if field not in patient_data]
                if missing_fields:
                    return jsonify({
                        'error': f'Missing required fields: {", ".join(missing_fields)}'
                    }), 400
                
                # Get model instance
                model = CancerRiskModel.get_instance()
                
                # Make prediction
                prediction = model.predict(patient_data)
                
                # Get risk factors analysis
                risk_factors = model.get_risk_factors(patient_data)
                
                # Get feature importances
                importances = model.feature_importances()
                
                # Combine results
                response = {
                    **prediction,
                    'risk_factors': risk_factors,
                    'feature_importances': importances
                }
                
                return jsonify(response), 200
                
            except KeyError as e:
                return jsonify({'error': f'Invalid value for field: {str(e)}'}), 400
            except Exception as e:
                return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    class _FeatureImportances(Resource):
        """Get feature importance scores from the model."""
        
        def get(self):
            """GET endpoint to retrieve feature importances.
            
            Returns JSON with feature importance scores.
            """
            try:
                model = CancerRiskModel.get_instance()
                importances = model.feature_importances()
                
                return jsonify({
                    'feature_importances': importances,
                    'sorted_features': sorted(
                        importances.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                }), 200
                
            except Exception as e:
                return jsonify({'error': f'Failed to get importances: {str(e)}'}), 500
    
    class _ModelInfo(Resource):
        """Get information about the trained model."""
        
        def get(self):
            """GET endpoint to retrieve model information.
            
            Returns model metadata and statistics.
            """
            try:
                model = CancerRiskModel.get_instance()
                
                return jsonify({
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
                }), 200
                
            except Exception as e:
                return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500


# Register API endpoints
api.add_resource(CancerRiskAPI._Predict, '/predict')
api.add_resource(CancerRiskAPI._FeatureImportances, '/feature-importances')
api.add_resource(CancerRiskAPI._ModelInfo, '/model-info')