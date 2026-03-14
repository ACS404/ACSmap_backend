"""
Cancer Risk Prediction Model - Enhanced Version
Uses machine learning to predict cancer risk based on demographic and lifestyle factors
Based on ACS Cancer Facts & Figures 2026 data with expanded medical history factors
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


class CancerRiskModel:
    """ML model for predicting cancer risk based on patient demographics and lifestyle factors.
    
    Uses epidemiological data from ACS Cancer Facts & Figures 2026 to train models
    that predict relative cancer risk compared to population baseline.
    """
    
    _instance = None
    
    def __init__(self):
        """Initialize the cancer risk prediction models."""
        self.logreg_model = None
        self.rf_model = None
        self.dt_model = None
        self.scaler = StandardScaler()
        
        # Define feature columns - EXPANDED
        self.features = [
            'age', 'sex', 'race', 'smoking_status', 'pack_years',
            'bmi_category', 'alcohol_consumption', 'physical_activity',
            'diet_quality', 'family_history', 'diabetes', 'hepatitis',
            'hpv', 'h_pylori', 'ibd', 'radiation_history',
            'immunosuppression', 'precancerous_lesions',
            'occupational_exposure', 'uv_exposure'
        ]
        
        # Generate synthetic training data based on ACS 2026 statistics
        self.training_data = self._generate_training_data()
        
    def _generate_training_data(self, n_samples=10000):
        """Generate synthetic training data based on ACS 2026 epidemiological data.
        
        Uses population statistics and relative risk multipliers from the ACS report
        to create realistic training examples.
        """
        np.random.seed(42)
        
        # Age distribution (weighted toward older adults as cancer risk increases with age)
        ages = np.random.choice(
            [25, 35, 45, 55, 65, 75, 85],
            n_samples,
            p=[0.10, 0.15, 0.20, 0.20, 0.20, 0.10, 0.05]
        )
        
        # Sex (0=female, 1=male)
        sex = np.random.binomial(1, 0.5, n_samples)
        
        # Race (0=White, 1=Black, 2=Hispanic, 3=AIAN, 4=AAPI)
        race = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.60, 0.13, 0.18, 0.02, 0.07])
        
        # Smoking status (0=never, 1=former, 2=current)
        smoking_status = np.random.choice([0, 1, 2], n_samples, p=[0.60, 0.30, 0.10])
        
        # Pack-years (for smokers only)
        pack_years = np.where(
            smoking_status > 0,
            np.random.gamma(2, 10, n_samples),  # Gamma distribution for pack-years
            0
        )
        
        # BMI category (0=normal, 1=overweight, 2=obese, 3=severely obese)
        bmi_category = np.random.choice([0, 1, 2, 3], n_samples, p=[0.28, 0.32, 0.30, 0.10])
        
        # Alcohol consumption (0=none, 1=light, 2=moderate, 3=heavy)
        alcohol = np.random.choice([0, 1, 2, 3], n_samples, p=[0.31, 0.40, 0.23, 0.06])
        
        # Physical activity (0=sedentary, 1=moderate, 2=active)
        activity = np.random.choice([0, 1, 2], n_samples, p=[0.26, 0.26, 0.48])
        
        # Diet quality (0=poor, 1=average, 2=healthy)
        diet = np.random.choice([0, 1, 2], n_samples, p=[0.20, 0.60, 0.20])
        
        # Family history (0=no, 1=yes) - 10-15% prevalence
        family_history = np.random.binomial(1, 0.13, n_samples)
        
        # Diabetes (0=no, 1=yes) - ~11% US prevalence
        diabetes = np.random.binomial(1, 0.11, n_samples)
        
        # Hepatitis (0=no, 1=yes) - ~2% prevalence
        hepatitis = np.random.binomial(1, 0.02, n_samples)
        
        # HPV infection (0=no, 1=yes) - ~43% prevalence in adults
        hpv = np.random.binomial(1, 0.12, n_samples)
        
        # H. pylori infection (0=no, 1=yes) - ~35% global prevalence
        h_pylori = np.random.binomial(1, 0.15, n_samples)
        
        # Inflammatory bowel disease (0=no, 1=yes) - ~1.3% prevalence
        ibd = np.random.binomial(1, 0.013, n_samples)
        
        # History of radiation therapy (0=no, 1=yes) - ~5% prevalence
        radiation_history = np.random.binomial(1, 0.05, n_samples)
        
        # Immunosuppression (0=no, 1=yes) - ~3% prevalence
        immunosuppression = np.random.binomial(1, 0.03, n_samples)
        
        # Precancerous lesions (0=no, 1=yes) - ~8% prevalence
        precancerous_lesions = np.random.binomial(1, 0.08, n_samples)
        
        # Occupational chemical exposure (0=no, 1=yes) - ~10% prevalence
        occupational_exposure = np.random.binomial(1, 0.10, n_samples)
        
        # High UV exposure (0=no, 1=yes) - ~20% prevalence
        uv_exposure = np.random.binomial(1, 0.20, n_samples)
        
        # Calculate risk score based on ACS 2026 relative risk factors
        risk_score = self._calculate_synthetic_risk(
            ages, sex, race, smoking_status, pack_years, bmi_category,
            alcohol, activity, diet, family_history, diabetes, hepatitis,
            hpv, h_pylori, ibd, radiation_history, immunosuppression,
            precancerous_lesions, occupational_exposure, uv_exposure
        )
        
        # Convert risk score to binary high/low risk (threshold at 75th percentile)
        high_risk_threshold = np.percentile(risk_score, 75)
        high_risk = (risk_score >= high_risk_threshold).astype(int)
        
        # Create DataFrame
        data = pd.DataFrame({
            'age': ages,
            'sex': sex,
            'race': race,
            'smoking_status': smoking_status,
            'pack_years': pack_years,
            'bmi_category': bmi_category,
            'alcohol_consumption': alcohol,
            'physical_activity': activity,
            'diet_quality': diet,
            'family_history': family_history,
            'diabetes': diabetes,
            'hepatitis': hepatitis,
            'hpv': hpv,
            'h_pylori': h_pylori,
            'ibd': ibd,
            'radiation_history': radiation_history,
            'immunosuppression': immunosuppression,
            'precancerous_lesions': precancerous_lesions,
            'occupational_exposure': occupational_exposure,
            'uv_exposure': uv_exposure,
            'risk_score': risk_score,
            'high_risk': high_risk
        })
        
        return data
    
    def _calculate_synthetic_risk(self, age, sex, race, smoking, pack_years, bmi,
                                   alcohol, activity, diet, family_hist, diabetes, hepatitis,
                                   hpv, h_pylori, ibd, radiation_hist, immunosupp,
                                   precancer, occup_exp, uv_exp):
        """Calculate synthetic risk score based on ACS 2026 relative risk multipliers."""
        
        # Base risk increases exponentially with age
        base_risk = np.exp((age - 50) / 20)
        
        # Smoking multiplier (19% of cancers attributable to smoking - ACS 2026)
        smoke_mult = np.where(smoking == 2, 5.0,  # Current smoker
                     np.where(smoking == 1, 2.0,  # Former smoker
                             1.0))  # Never smoked
        smoke_mult = smoke_mult * (1 + pack_years / 100)  # Dose-response
        
        # BMI multiplier (8% of cancers attributable to excess body weight - ACS 2026)
        bmi_mult = np.where(bmi == 3, 2.5,  # Severely obese
                   np.where(bmi == 2, 1.8,  # Obese
                   np.where(bmi == 1, 1.3,  # Overweight
                           1.0)))  # Normal
        
        # Alcohol multiplier (5% of cancers attributable to alcohol - ACS 2026)
        alc_mult = np.where(alcohol == 3, 1.8,  # Heavy
                   np.where(alcohol == 2, 1.3,  # Moderate
                   np.where(alcohol == 1, 1.1,  # Light
                           1.0)))  # None
        
        # Physical activity multiplier
        activity_mult = np.where(activity == 2, 0.85,  # Active
                        np.where(activity == 1, 0.95,  # Moderate
                                1.15))  # Sedentary
        
        # Diet quality multiplier
        diet_mult = np.where(diet == 2, 0.9,  # Healthy
                    np.where(diet == 1, 1.0,  # Average
                            1.2))  # Poor
        
        # Family history multiplier (1.5-2x risk for most cancers)
        family_mult = np.where(family_hist == 1, 1.7, 1.0)
        
        # Diabetes multiplier (increased risk for pancreas, liver, colorectal, uterine)
        diabetes_mult = np.where(diabetes == 1, 1.3, 1.0)
        
        # Hepatitis multiplier (major risk for liver cancer - ~75% of liver cancers)
        hepatitis_mult = np.where(hepatitis == 1, 3.0, 1.0)
        
        # HPV multiplier (causes cervical, oropharyngeal, anal cancers)
        hpv_mult = np.where(hpv == 1, 2.0, 1.0)
        
        # H. pylori multiplier (increases gastric cancer risk 2-6x)
        h_pylori_mult = np.where(h_pylori == 1, 2.5, 1.0)
        
        # IBD multiplier (increases colorectal cancer risk 2-3x)
        ibd_mult = np.where(ibd == 1, 2.5, 1.0)
        
        # Radiation history multiplier (increases leukemia, thyroid, breast cancer)
        radiation_mult = np.where(radiation_hist == 1, 2.0, 1.0)
        
        # Immunosuppression multiplier (increases lymphoma, skin cancer risk)
        immunosupp_mult = np.where(immunosupp == 1, 2.2, 1.0)
        
        # Precancerous lesions multiplier (5-20x risk depending on type)
        precancer_mult = np.where(precancer == 1, 3.5, 1.0)
        
        # Occupational exposure multiplier (asbestos, benzene, etc.)
        occup_mult = np.where(occup_exp == 1, 1.5, 1.0)
        
        # UV exposure multiplier (increases skin cancer risk 2-3x)
        uv_mult = np.where(uv_exp == 1, 1.8, 1.0)
        
        # Sex-based baseline (males have higher overall cancer incidence)
        sex_mult = np.where(sex == 1, 1.15, 1.0)
        
        # Race-based multipliers (from ACS Table 9)
        race_mult = np.where(race == 1, 1.14,  # Black
                    np.where(race == 3, 1.20,  # AIAN
                    np.where(race == 4, 0.90,  # AAPI
                    np.where(race == 2, 0.95,  # Hispanic
                            1.0))))  # White (baseline)
        
        # Combine all multipliers
        risk_score = (base_risk * smoke_mult * bmi_mult * alc_mult * 
                     activity_mult * diet_mult * family_mult * 
                     diabetes_mult * hepatitis_mult * hpv_mult *
                     h_pylori_mult * ibd_mult * radiation_mult *
                     immunosupp_mult * precancer_mult * occup_mult *
                     uv_mult * sex_mult * race_mult)
        
        # Add some random noise
        noise = np.random.normal(1.0, 0.1, len(age))
        risk_score = risk_score * noise
        
        return risk_score
    
    def _train(self):
        """Train machine learning models on the synthetic data."""
        
        # Prepare features and target
        X = self.training_data[self.features]
        y = self.training_data['high_risk']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Logistic Regression (main prediction model)
        self.logreg_model = LogisticRegression(max_iter=1000, random_state=42)
        self.logreg_model.fit(X_train_scaled, y_train)
        
        # Train Random Forest (for comparison)
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train, y_train)
        
        # Train Decision Tree (for feature importance)
        self.dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
        self.dt_model.fit(X_train, y_train)
        
        # Calculate accuracies
        logreg_acc = self.logreg_model.score(X_test_scaled, y_test)
        rf_acc = self.rf_model.score(X_test, y_test)
        dt_acc = self.dt_model.score(X_test, y_test)
        
        print(f"Model Training Complete:")
        print(f"  Logistic Regression Accuracy: {logreg_acc:.2%}")
        print(f"  Random Forest Accuracy: {rf_acc:.2%}")
        print(f"  Decision Tree Accuracy: {dt_acc:.2%}")
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance of the model (trained once, used many times)."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._train()
        return cls._instance
    
    def predict(self, patient_data):
        """Predict cancer risk for a patient.
        
        Args:
            patient_data (dict): Dictionary with patient information (see features list)
        
        Returns:
            dict: Prediction results with risk probabilities and explanations
        """
        
        # Convert categorical variables to numeric
        sex_map = {'female': 0, 'male': 1}
        race_map = {'white': 0, 'black': 1, 'hispanic': 2, 'aian': 3, 'aapi': 4}
        smoking_map = {'never': 0, 'former': 1, 'current': 2}
        bmi_map = {'normal': 0, 'overweight': 1, 'obese': 2, 'severely-obese': 3}
        alcohol_map = {'none': 0, 'light': 1, 'moderate': 2, 'heavy': 3}
        activity_map = {'sedentary': 0, 'moderate': 1, 'active': 2}
        diet_map = {'poor': 0, 'average': 1, 'healthy': 2}
        
        # Create feature vector
        features = pd.DataFrame([{
            'age': patient_data['age'],
            'sex': sex_map[patient_data['sex']],
            'race': race_map[patient_data['race']],
            'smoking_status': smoking_map[patient_data['smoking_status']],
            'pack_years': patient_data.get('pack_years', 0),
            'bmi_category': bmi_map[patient_data['bmi_category']],
            'alcohol_consumption': alcohol_map[patient_data['alcohol_consumption']],
            'physical_activity': activity_map[patient_data['physical_activity']],
            'diet_quality': diet_map[patient_data['diet_quality']],
            'family_history': int(patient_data.get('family_history', False)),
            'diabetes': int(patient_data.get('diabetes', False)),
            'hepatitis': int(patient_data.get('hepatitis', False)),
            'hpv': int(patient_data.get('hpv', False)),
            'h_pylori': int(patient_data.get('h_pylori', False)),
            'ibd': int(patient_data.get('ibd', False)),
            'radiation_history': int(patient_data.get('radiation_history', False)),
            'immunosuppression': int(patient_data.get('immunosuppression', False)),
            'precancerous_lesions': int(patient_data.get('precancerous_lesions', False)),
            'occupational_exposure': int(patient_data.get('occupational_exposure', False)),
            'uv_exposure': int(patient_data.get('uv_exposure', False))
        }])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get predictions from all models
        logreg_proba = self.logreg_model.predict_proba(features_scaled)[0]
        rf_proba = self.rf_model.predict_proba(features)[0]
        
        # Ensemble prediction (average of models)
        low_risk_prob = (logreg_proba[0] + rf_proba[0]) / 2
        high_risk_prob = (logreg_proba[1] + rf_proba[1]) / 2
        
        return {
            'low_risk_probability': float(low_risk_prob),
            'high_risk_probability': float(high_risk_prob),
            'risk_category': 'high' if high_risk_prob > 0.5 else 'low',
            'model_confidence': float(max(low_risk_prob, high_risk_prob))
        }
    
    def feature_importances(self):
        """Get feature importance scores from the decision tree model.
        
        Returns:
            dict: Feature names mapped to importance scores
        """
        importances = self.dt_model.feature_importances_
        return {feature: float(importance) 
                for feature, importance in zip(self.features, importances)}
    
    def get_risk_factors(self, patient_data):
        """Analyze which factors contribute most to patient's risk.
        
        Returns:
            list: Sorted list of risk factors with their contributions
        """
        risk_factors = []
        
        # Analyze each risk factor
        if patient_data.get('smoking_status') == 'current':
            risk_factors.append({
                'factor': 'Current smoking',
                'impact': 'high',
                'detail': f"{patient_data.get('pack_years', 0):.0f} pack-years. Smoking causes 19% of all cancers (ACS 2026)."
            })
        elif patient_data.get('smoking_status') == 'former':
            risk_factors.append({
                'factor': 'Former smoking',
                'impact': 'moderate',
                'detail': f"{patient_data.get('pack_years', 0):.0f} pack-years. Risk decreases after quitting but remains elevated."
            })
        
        if patient_data.get('bmi_category') in ['obese', 'severely-obese']:
            risk_factors.append({
                'factor': 'Excess body weight',
                'impact': 'high' if patient_data.get('bmi_category') == 'severely-obese' else 'moderate',
                'detail': 'Excess body weight is attributable to 8% of cancer cases (ACS 2026).'
            })
        
        if patient_data.get('alcohol_consumption') in ['moderate', 'heavy']:
            risk_factors.append({
                'factor': 'Alcohol consumption',
                'impact': 'high' if patient_data.get('alcohol_consumption') == 'heavy' else 'moderate',
                'detail': 'Alcohol consumption is attributable to 5% of cancer cases (ACS 2026).'
            })
        
        if patient_data.get('physical_activity') == 'sedentary':
            risk_factors.append({
                'factor': 'Physical inactivity',
                'impact': 'moderate',
                'detail': 'Regular physical activity reduces risk for multiple cancer types.'
            })
        
        if patient_data.get('diet_quality') == 'poor':
            risk_factors.append({
                'factor': 'Poor diet quality',
                'impact': 'moderate',
                'detail': 'High red/processed meat intake and low fruit/vegetable consumption increase risk.'
            })
        
        if patient_data.get('family_history'):
            risk_factors.append({
                'factor': 'Family history of cancer',
                'impact': 'high',
                'detail': 'First-degree relatives with cancer significantly increase risk (1.5-2x).'
            })
        
        if patient_data.get('diabetes'):
            risk_factors.append({
                'factor': 'Type 2 diabetes',
                'impact': 'moderate',
                'detail': 'Diabetes increases risk for pancreatic, liver, colorectal, and uterine cancers.'
            })
        
        if patient_data.get('hepatitis'):
            risk_factors.append({
                'factor': 'Hepatitis B or C infection',
                'impact': 'high',
                'detail': '~75% of liver cancers are attributable to HBV/HCV infection (ACS 2026).'
            })
        
        if patient_data.get('hpv'):
            risk_factors.append({
                'factor': 'HPV infection',
                'impact': 'high',
                'detail': 'HPV causes cervical, oropharyngeal, and anal cancers. Vaccination recommended.'
            })
        
        if patient_data.get('h_pylori'):
            risk_factors.append({
                'factor': 'H. pylori infection',
                'impact': 'high',
                'detail': 'H. pylori increases gastric cancer risk 2-6x. Treatment available.'
            })
        
        if patient_data.get('ibd'):
            risk_factors.append({
                'factor': 'Inflammatory bowel disease',
                'impact': 'high',
                'detail': 'IBD increases colorectal cancer risk 2-3x. Regular screening essential.'
            })
        
        if patient_data.get('radiation_history'):
            risk_factors.append({
                'factor': 'Prior radiation therapy',
                'impact': 'high',
                'detail': 'Radiation therapy increases risk for leukemia, thyroid, and breast cancer.'
            })
        
        if patient_data.get('immunosuppression'):
            risk_factors.append({
                'factor': 'Immunosuppression',
                'impact': 'high',
                'detail': 'Immunosuppressive medications increase lymphoma and skin cancer risk.'
            })
        
        if patient_data.get('precancerous_lesions'):
            risk_factors.append({
                'factor': 'History of precancerous lesions',
                'impact': 'high',
                'detail': 'Precancerous lesions (polyps, dysplasia) indicate 5-20x increased risk.'
            })
        
        if patient_data.get('occupational_exposure'):
            risk_factors.append({
                'factor': 'Occupational chemical exposure',
                'impact': 'moderate',
                'detail': 'Exposure to asbestos, benzene, or other carcinogens increases risk.'
            })
        
        if patient_data.get('uv_exposure'):
            risk_factors.append({
                'factor': 'High UV exposure history',
                'impact': 'moderate',
                'detail': 'Chronic sun exposure increases melanoma and non-melanoma skin cancer risk.'
            })
        
        # Sort by impact
        impact_order = {'high': 0, 'moderate': 1, 'low': 2}
        risk_factors.sort(key=lambda x: impact_order[x['impact']])
        
        return risk_factors


def initCancerRisk():
    """Initialize the cancer risk model singleton."""
    CancerRiskModel.get_instance()


def testCancerRisk():
    """Test the cancer risk model with sample patient data."""
    
    print("\n" + "="*70)
    print("CANCER RISK PREDICTION MODEL TEST - ENHANCED VERSION")
    print("Based on ACS Cancer Facts & Figures 2026")
    print("="*70)
    
    # Sample patient with multiple risk factors
    print("\nStep 1: Define patient data for prediction")
    patient = {
        'age': 60,
        'sex': 'male',
        'race': 'white',
        'smoking_status': 'former',
        'pack_years': 20,
        'bmi_category': 'overweight',
        'alcohol_consumption': 'moderate',
        'physical_activity': 'moderate',
        'diet_quality': 'average',
        'family_history': True,
        'diabetes': False,
        'hepatitis': False,
        'hpv': False,
        'h_pylori': True,
        'ibd': False,
        'radiation_history': False,
        'immunosuppression': False,
        'precancerous_lesions': True,
        'occupational_exposure': False,
        'uv_exposure': True
    }
    
    for key, value in patient.items():
        print(f"  {key}: {value}")
    
    # Get model instance
    print("\nStep 2: Load trained cancer risk model")
    model = CancerRiskModel.get_instance()
    print("  Model loaded with logistic regression, random forest, and decision tree")
    
    # Predict risk
    print("\nStep 3: Predict cancer risk")
    prediction = model.predict(patient)
    print(f"  Low Risk Probability: {prediction['low_risk_probability']:.2%}")
    print(f"  High Risk Probability: {prediction['high_risk_probability']:.2%}")
    print(f"  Risk Category: {prediction['risk_category'].upper()}")
    print(f"  Model Confidence: {prediction['model_confidence']:.2%}")
    
    # Get risk factors
    print("\nStep 4: Analyze risk factors")
    risk_factors = model.get_risk_factors(patient)
    for factor in risk_factors:
        print(f"  [{factor['impact'].upper()}] {factor['factor']}")
        print(f"      {factor['detail']}")
    
    # Feature importances
    print("\nStep 5: Feature importance analysis (top 10)")
    importances = model.feature_importances()
    for feature, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {feature:30s} {importance:.2%}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    testCancerRisk()