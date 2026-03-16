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
    """ML model for predicting cancer risk based on patient demographics and lifestyle factors."""

    _instance = None

    # Cancer type metadata: label + sex restriction (None = applies to all)
    CANCER_TYPES_META = {
        'lung':       {'label': 'Lung & Bronchus',      'sex': None},
        'colorectal': {'label': 'Colorectal',            'sex': None},
        'breast':     {'label': 'Breast',                'sex': 'female'},
        'prostate':   {'label': 'Prostate',              'sex': 'male'},
        'melanoma':   {'label': 'Skin Melanoma',         'sex': None},
        'liver':      {'label': 'Liver',                 'sex': None},
        'cervical':   {'label': 'Cervical',              'sex': 'female'},
        'stomach':    {'label': 'Stomach / Gastric',     'sex': None},
        'bladder':    {'label': 'Bladder',               'sex': None},
        'lymphoma':   {'label': 'Non-Hodgkin Lymphoma',  'sex': None},
        'leukemia':   {'label': 'Leukemia',              'sex': None},
        'pancreatic': {'label': 'Pancreatic',            'sex': None},
    }

    def __init__(self):
        self.logreg_model = None
        self.rf_model = None
        self.dt_model = None
        self.scaler = StandardScaler()

        self.features = [
            'age', 'sex', 'race', 'smoking_status', 'pack_years',
            'bmi_category', 'alcohol_consumption', 'physical_activity',
            'diet_quality', 'family_history', 'diabetes', 'hepatitis',
            'hpv', 'h_pylori', 'ibd', 'radiation_history',
            'immunosuppression', 'precancerous_lesions',
            'occupational_exposure', 'uv_exposure'
        ]

        self.training_data = self._generate_training_data()

    def _generate_training_data(self, n_samples=10000):
        np.random.seed(42)
        ages = np.random.choice(
            [25, 35, 45, 55, 65, 75, 85], n_samples,
            p=[0.10, 0.15, 0.20, 0.20, 0.20, 0.10, 0.05]
        )
        sex = np.random.binomial(1, 0.5, n_samples)
        race = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.60, 0.13, 0.18, 0.02, 0.07])
        smoking_status = np.random.choice([0, 1, 2], n_samples, p=[0.60, 0.30, 0.10])
        pack_years = np.where(smoking_status > 0, np.random.gamma(2, 10, n_samples), 0)
        bmi_category = np.random.choice([0, 1, 2, 3], n_samples, p=[0.28, 0.32, 0.30, 0.10])
        alcohol = np.random.choice([0, 1, 2, 3], n_samples, p=[0.31, 0.40, 0.23, 0.06])
        activity = np.random.choice([0, 1, 2], n_samples, p=[0.26, 0.26, 0.48])
        diet = np.random.choice([0, 1, 2], n_samples, p=[0.20, 0.60, 0.20])
        family_history = np.random.binomial(1, 0.13, n_samples)
        diabetes = np.random.binomial(1, 0.11, n_samples)
        hepatitis = np.random.binomial(1, 0.02, n_samples)
        hpv = np.random.binomial(1, 0.12, n_samples)
        h_pylori = np.random.binomial(1, 0.15, n_samples)
        ibd = np.random.binomial(1, 0.013, n_samples)
        radiation_history = np.random.binomial(1, 0.05, n_samples)
        immunosuppression = np.random.binomial(1, 0.03, n_samples)
        precancerous_lesions = np.random.binomial(1, 0.08, n_samples)
        occupational_exposure = np.random.binomial(1, 0.10, n_samples)
        uv_exposure = np.random.binomial(1, 0.20, n_samples)

        risk_score = self._calculate_synthetic_risk(
            ages, sex, race, smoking_status, pack_years, bmi_category,
            alcohol, activity, diet, family_history, diabetes, hepatitis,
            hpv, h_pylori, ibd, radiation_history, immunosuppression,
            precancerous_lesions, occupational_exposure, uv_exposure
        )

        high_risk_threshold = np.percentile(risk_score, 75)
        high_risk = (risk_score >= high_risk_threshold).astype(int)

        return pd.DataFrame({
            'age': ages, 'sex': sex, 'race': race,
            'smoking_status': smoking_status, 'pack_years': pack_years,
            'bmi_category': bmi_category, 'alcohol_consumption': alcohol,
            'physical_activity': activity, 'diet_quality': diet,
            'family_history': family_history, 'diabetes': diabetes,
            'hepatitis': hepatitis, 'hpv': hpv, 'h_pylori': h_pylori,
            'ibd': ibd, 'radiation_history': radiation_history,
            'immunosuppression': immunosuppression,
            'precancerous_lesions': precancerous_lesions,
            'occupational_exposure': occupational_exposure,
            'uv_exposure': uv_exposure,
            'risk_score': risk_score, 'high_risk': high_risk
        })

    def _calculate_synthetic_risk(self, age, sex, race, smoking, pack_years, bmi,
                                   alcohol, activity, diet, family_hist, diabetes, hepatitis,
                                   hpv, h_pylori, ibd, radiation_hist, immunosupp,
                                   precancer, occup_exp, uv_exp):
        base_risk = np.exp((age - 50) / 20)
        smoke_mult = np.where(smoking == 2, 5.0, np.where(smoking == 1, 2.0, 1.0))
        smoke_mult = smoke_mult * (1 + pack_years / 100)
        bmi_mult = np.where(bmi == 3, 2.5, np.where(bmi == 2, 1.8, np.where(bmi == 1, 1.3, 1.0)))
        alc_mult = np.where(alcohol == 3, 1.8, np.where(alcohol == 2, 1.3, np.where(alcohol == 1, 1.1, 1.0)))
        activity_mult = np.where(activity == 2, 0.85, np.where(activity == 1, 0.95, 1.15))
        diet_mult = np.where(diet == 2, 0.9, np.where(diet == 1, 1.0, 1.2))
        family_mult = np.where(family_hist == 1, 1.7, 1.0)
        diabetes_mult = np.where(diabetes == 1, 1.3, 1.0)
        hepatitis_mult = np.where(hepatitis == 1, 3.0, 1.0)
        hpv_mult = np.where(hpv == 1, 2.0, 1.0)
        h_pylori_mult = np.where(h_pylori == 1, 2.5, 1.0)
        ibd_mult = np.where(ibd == 1, 2.5, 1.0)
        radiation_mult = np.where(radiation_hist == 1, 2.0, 1.0)
        immunosupp_mult = np.where(immunosupp == 1, 2.2, 1.0)
        precancer_mult = np.where(precancer == 1, 3.5, 1.0)
        occup_mult = np.where(occup_exp == 1, 1.5, 1.0)
        uv_mult = np.where(uv_exp == 1, 1.8, 1.0)
        sex_mult = np.where(sex == 1, 1.15, 1.0)
        race_mult = np.where(race == 1, 1.14, np.where(race == 3, 1.20,
                    np.where(race == 4, 0.90, np.where(race == 2, 0.95, 1.0))))

        risk_score = (base_risk * smoke_mult * bmi_mult * alc_mult *
                      activity_mult * diet_mult * family_mult *
                      diabetes_mult * hepatitis_mult * hpv_mult *
                      h_pylori_mult * ibd_mult * radiation_mult *
                      immunosupp_mult * precancer_mult * occup_mult *
                      uv_mult * sex_mult * race_mult)
        return risk_score * np.random.normal(1.0, 0.1, len(age))

    def _train(self):
        X = self.training_data[self.features]
        y = self.training_data['high_risk']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        self.logreg_model = LogisticRegression(max_iter=1000, random_state=42)
        self.logreg_model.fit(X_train_scaled, y_train)

        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train, y_train)

        self.dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
        self.dt_model.fit(X_train, y_train)

        logreg_acc = self.logreg_model.score(X_test_scaled, y_test)
        rf_acc     = self.rf_model.score(X_test, y_test)
        dt_acc     = self.dt_model.score(X_test, y_test)
        print(f"Model Training Complete:")
        print(f"  Logistic Regression Accuracy: {logreg_acc:.2%}")
        print(f"  Random Forest Accuracy: {rf_acc:.2%}")
        print(f"  Decision Tree Accuracy: {dt_acc:.2%}")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._train()
        return cls._instance

    def predict(self, patient_data):
        sex_map      = {'female': 0, 'male': 1}
        race_map     = {'white': 0, 'black': 1, 'hispanic': 2, 'aian': 3, 'aapi': 4}
        smoking_map  = {'never': 0, 'former': 1, 'current': 2}
        bmi_map      = {'normal': 0, 'overweight': 1, 'obese': 2, 'severely-obese': 3}
        alcohol_map  = {'none': 0, 'light': 1, 'moderate': 2, 'heavy': 3}
        activity_map = {'sedentary': 0, 'moderate': 1, 'active': 2}
        diet_map     = {'poor': 0, 'average': 1, 'healthy': 2}

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

        features_scaled = self.scaler.transform(features)
        logreg_proba = self.logreg_model.predict_proba(features_scaled)[0]
        rf_proba     = self.rf_model.predict_proba(features)[0]

        low_risk_prob  = (logreg_proba[0] + rf_proba[0]) / 2
        high_risk_prob = (logreg_proba[1] + rf_proba[1]) / 2

        return {
            'low_risk_probability':  float(low_risk_prob),
            'high_risk_probability': float(high_risk_prob),
            'risk_category':         'high' if high_risk_prob > 0.5 else 'low',
            'model_confidence':      float(max(low_risk_prob, high_risk_prob))
        }

    def feature_importances(self):
        importances = self.dt_model.feature_importances_
        return {f: float(i) for f, i in zip(self.features, importances)}

    # ── PER-CANCER-TYPE RISK ──────────────────────────────────────────────

    def predict_cancer_types(self, patient_data, selected_types=None):
        """Return per-cancer relative risk scores for the patient.

        Args:
            patient_data: dict of patient fields
            selected_types: list of cancer type keys to compute (None = all)

        Returns:
            dict keyed by cancer type with relative_risk, risk_level, key_factors, note
        """
        types_to_compute = (
            [t for t in selected_types if t in self.CANCER_TYPES_META]
            if selected_types
            else list(self.CANCER_TYPES_META.keys())
        )

        results = {}
        for ct in types_to_compute:
            meta = self.CANCER_TYPES_META[ct]

            # Sex-specific cancers
            if meta['sex'] and patient_data.get('sex') != meta['sex']:
                results[ct] = {
                    'label': meta['label'],
                    'applicable': False,
                    'relative_risk': None,
                    'risk_level': 'n/a',
                    'key_factors': [],
                    'note': f'Not applicable based on biological sex.'
                }
                continue

            rr, factors, note = self._compute_type_risk(ct, patient_data)

            if rr < 1.5:
                level = 'low'
            elif rr < 3.0:
                level = 'moderate'
            else:
                level = 'high'

            results[ct] = {
                'label': meta['label'],
                'applicable': True,
                'relative_risk': round(rr, 2),
                'risk_level': level,
                'key_factors': factors,
                'note': note
            }

        return results

    def _compute_type_risk(self, cancer_type, d):
        """Compute relative risk ratio (vs. population average = 1.0) for one cancer type."""
        age = d.get('age', 50)
        # Gentle age scaling — steeper after 50
        age_factor = 1.0 + max(0, (age - 50)) / 35

        rr = 1.0
        factors = []

        # ── Lung ──────────────────────────────────────────────────────────
        if cancer_type == 'lung':
            if d.get('smoking_status') == 'current':
                mult = 10.0 + d.get('pack_years', 0) * 0.08
                rr *= mult
                factors.append('Current smoking')
            elif d.get('smoking_status') == 'former':
                rr *= 3.5
                factors.append('Former smoking')
            if d.get('occupational_exposure'):
                rr *= 1.5
                factors.append('Occupational exposure')
            note = 'Smoking accounts for ~85% of lung cancer cases (ACS 2026).'

        # ── Colorectal ────────────────────────────────────────────────────
        elif cancer_type == 'colorectal':
            if d.get('bmi_category') in ['obese', 'severely-obese']:
                rr *= 1.5; factors.append('Obesity')
            if d.get('physical_activity') == 'sedentary':
                rr *= 1.3; factors.append('Physical inactivity')
            if d.get('alcohol_consumption') in ['moderate', 'heavy']:
                rr *= 1.4; factors.append('Alcohol consumption')
            if d.get('diet_quality') == 'poor':
                rr *= 1.3; factors.append('Poor diet')
            if d.get('ibd'):
                rr *= 2.5; factors.append('Inflammatory bowel disease')
            if d.get('family_history'):
                rr *= 1.7; factors.append('Family history')
            if d.get('precancerous_lesions'):
                rr *= 3.0; factors.append('Precancerous lesions (polyps)')
            if d.get('diabetes'):
                rr *= 1.3; factors.append('Type 2 diabetes')
            note = 'Lifestyle factors strongly drive colorectal cancer risk; screening from age 45 is recommended.'

        # ── Breast ────────────────────────────────────────────────────────
        elif cancer_type == 'breast':
            if d.get('bmi_category') in ['obese', 'severely-obese']:
                rr *= 1.4; factors.append('Obesity')
            if d.get('alcohol_consumption') in ['moderate', 'heavy']:
                rr *= 1.4; factors.append('Alcohol consumption')
            if d.get('family_history'):
                rr *= 2.0; factors.append('Family history')
            if d.get('physical_activity') == 'sedentary':
                rr *= 1.2; factors.append('Physical inactivity')
            if d.get('smoking_status') == 'current':
                rr *= 1.15; factors.append('Smoking')
            note = 'Annual mammography screening is recommended starting at age 40 (ACS 2026).'

        # ── Prostate ──────────────────────────────────────────────────────
        elif cancer_type == 'prostate':
            if d.get('family_history'):
                rr *= 2.5; factors.append('Family history')
            if d.get('bmi_category') in ['obese', 'severely-obese']:
                rr *= 1.3; factors.append('Obesity')
            if d.get('diet_quality') == 'poor':
                rr *= 1.2; factors.append('Poor diet')
            if d.get('race') == 'black':
                rr *= 1.7; factors.append('Higher incidence in Black men (ACS 2026)')
            note = 'Discuss PSA screening timing with your healthcare provider.'

        # ── Melanoma ──────────────────────────────────────────────────────
        elif cancer_type == 'melanoma':
            if d.get('uv_exposure'):
                rr *= 2.5; factors.append('UV / sun exposure history')
            if d.get('family_history'):
                rr *= 2.0; factors.append('Family history')
            if d.get('immunosuppression'):
                rr *= 1.8; factors.append('Immunosuppression')
            note = 'Daily SPF use and annual dermatology checks are the strongest preventive measures.'

        # ── Liver ─────────────────────────────────────────────────────────
        elif cancer_type == 'liver':
            if d.get('hepatitis'):
                rr *= 8.0; factors.append('Hepatitis B / C infection')
            if d.get('alcohol_consumption') == 'heavy':
                rr *= 3.0; factors.append('Heavy alcohol use')
            elif d.get('alcohol_consumption') == 'moderate':
                rr *= 1.5; factors.append('Moderate alcohol use')
            if d.get('diabetes'):
                rr *= 1.8; factors.append('Type 2 diabetes')
            if d.get('bmi_category') in ['obese', 'severely-obese']:
                rr *= 1.5; factors.append('Obesity / NAFLD')
            note = '~75% of liver cancers are linked to HBV/HCV infection (ACS 2026); antiviral treatment is available.'

        # ── Cervical ──────────────────────────────────────────────────────
        elif cancer_type == 'cervical':
            if d.get('hpv'):
                rr *= 5.0; factors.append('HPV infection')
            if d.get('smoking_status') == 'current':
                rr *= 1.6; factors.append('Smoking')
            if d.get('immunosuppression'):
                rr *= 1.8; factors.append('Immunosuppression')
            note = 'HPV vaccination and regular Pap/HPV co-testing are highly effective at prevention.'

        # ── Stomach ───────────────────────────────────────────────────────
        elif cancer_type == 'stomach':
            if d.get('h_pylori'):
                rr *= 4.0; factors.append('H. pylori infection')
            if d.get('diet_quality') == 'poor':
                rr *= 1.5; factors.append('Poor diet (high salt / processed meats)')
            if d.get('smoking_status') in ['current', 'former']:
                rr *= 1.5; factors.append('Smoking history')
            if d.get('family_history'):
                rr *= 2.0; factors.append('Family history')
            note = 'H. pylori is the leading modifiable risk factor; antibiotic treatment is effective.'

        # ── Bladder ───────────────────────────────────────────────────────
        elif cancer_type == 'bladder':
            if d.get('smoking_status') == 'current':
                rr *= 4.0; factors.append('Current smoking')
            elif d.get('smoking_status') == 'former':
                rr *= 2.0; factors.append('Former smoking')
            if d.get('occupational_exposure'):
                rr *= 2.0; factors.append('Occupational chemical exposure')
            note = 'Smoking is the single biggest modifiable risk factor for bladder cancer.'

        # ── Lymphoma ──────────────────────────────────────────────────────
        elif cancer_type == 'lymphoma':
            if d.get('immunosuppression'):
                rr *= 3.5; factors.append('Immunosuppression')
            if d.get('hpv') or d.get('hepatitis'):
                rr *= 1.5; factors.append('Viral infection (HPV / Hepatitis)')
            if d.get('family_history'):
                rr *= 1.5; factors.append('Family history')
            note = 'Immune system health plays a central role in lymphoma risk.'

        # ── Leukemia ──────────────────────────────────────────────────────
        elif cancer_type == 'leukemia':
            if d.get('radiation_history'):
                rr *= 3.0; factors.append('Prior radiation therapy')
            if d.get('immunosuppression'):
                rr *= 2.0; factors.append('Immunosuppression')
            if d.get('occupational_exposure'):
                rr *= 1.8; factors.append('Occupational exposure (benzene)')
            if d.get('smoking_status') == 'current':
                rr *= 1.4; factors.append('Smoking')
            note = 'Prior radiation exposure and certain chemical exposures are significant leukemia risk factors.'

        # ── Pancreatic ────────────────────────────────────────────────────
        elif cancer_type == 'pancreatic':
            if d.get('smoking_status') == 'current':
                rr *= 2.5; factors.append('Current smoking')
            elif d.get('smoking_status') == 'former':
                rr *= 1.5; factors.append('Former smoking')
            if d.get('diabetes'):
                rr *= 1.8; factors.append('Type 2 diabetes')
            if d.get('bmi_category') in ['obese', 'severely-obese']:
                rr *= 1.5; factors.append('Obesity')
            if d.get('family_history'):
                rr *= 2.0; factors.append('Family history')
            if d.get('alcohol_consumption') == 'heavy':
                rr *= 1.5; factors.append('Heavy alcohol use')
            note = 'Pancreatic cancer is often diagnosed late; managing modifiable risks is critical.'

        else:
            note = 'Unknown cancer type.'

        rr *= age_factor
        rr = max(0.1, rr)

        if not factors:
            note = 'No major identified risk factors for this cancer type in your profile.'

        return rr, factors, note

    # ── RISK FACTORS (overall) ────────────────────────────────────────────

    def get_risk_factors(self, patient_data):
        risk_factors = []

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
            risk_factors.append({'factor': 'Physical inactivity', 'impact': 'moderate',
                'detail': 'Regular physical activity reduces risk for multiple cancer types.'})
        if patient_data.get('diet_quality') == 'poor':
            risk_factors.append({'factor': 'Poor diet quality', 'impact': 'moderate',
                'detail': 'High red/processed meat intake and low fruit/vegetable consumption increase risk.'})
        if patient_data.get('family_history'):
            risk_factors.append({'factor': 'Family history of cancer', 'impact': 'high',
                'detail': 'First-degree relatives with cancer significantly increase risk (1.5–2×).'})
        if patient_data.get('diabetes'):
            risk_factors.append({'factor': 'Type 2 diabetes', 'impact': 'moderate',
                'detail': 'Diabetes increases risk for pancreatic, liver, colorectal, and uterine cancers.'})
        if patient_data.get('hepatitis'):
            risk_factors.append({'factor': 'Hepatitis B or C infection', 'impact': 'high',
                'detail': '~75% of liver cancers are attributable to HBV/HCV infection (ACS 2026).'})
        if patient_data.get('hpv'):
            risk_factors.append({'factor': 'HPV infection', 'impact': 'high',
                'detail': 'HPV causes cervical, oropharyngeal, and anal cancers. Vaccination recommended.'})
        if patient_data.get('h_pylori'):
            risk_factors.append({'factor': 'H. pylori infection', 'impact': 'high',
                'detail': 'H. pylori increases gastric cancer risk 2–6×. Treatment available.'})
        if patient_data.get('ibd'):
            risk_factors.append({'factor': 'Inflammatory bowel disease', 'impact': 'high',
                'detail': 'IBD increases colorectal cancer risk 2–3×. Regular screening essential.'})
        if patient_data.get('radiation_history'):
            risk_factors.append({'factor': 'Prior radiation therapy', 'impact': 'high',
                'detail': 'Radiation therapy increases risk for leukemia, thyroid, and breast cancer.'})
        if patient_data.get('immunosuppression'):
            risk_factors.append({'factor': 'Immunosuppression', 'impact': 'high',
                'detail': 'Immunosuppressive medications increase lymphoma and skin cancer risk.'})
        if patient_data.get('precancerous_lesions'):
            risk_factors.append({'factor': 'History of precancerous lesions', 'impact': 'high',
                'detail': 'Precancerous lesions (polyps, dysplasia) indicate 5–20× increased risk.'})
        if patient_data.get('occupational_exposure'):
            risk_factors.append({'factor': 'Occupational chemical exposure', 'impact': 'moderate',
                'detail': 'Exposure to asbestos, benzene, or other carcinogens increases risk.'})
        if patient_data.get('uv_exposure'):
            risk_factors.append({'factor': 'High UV exposure history', 'impact': 'moderate',
                'detail': 'Chronic sun exposure increases melanoma and non-melanoma skin cancer risk.'})

        impact_order = {'high': 0, 'moderate': 1, 'low': 2}
        risk_factors.sort(key=lambda x: impact_order[x['impact']])
        return risk_factors


def initCancerRisk():
    CancerRiskModel.get_instance()


def testCancerRisk():
    print("\n" + "="*70)
    print("CANCER RISK PREDICTION MODEL TEST - ENHANCED VERSION")
    print("Based on ACS Cancer Facts & Figures 2026")
    print("="*70)

    patient = {
        'age': 60, 'sex': 'male', 'race': 'white',
        'smoking_status': 'former', 'pack_years': 20,
        'bmi_category': 'overweight', 'alcohol_consumption': 'moderate',
        'physical_activity': 'moderate', 'diet_quality': 'average',
        'family_history': True, 'diabetes': False, 'hepatitis': False,
        'hpv': False, 'h_pylori': True, 'ibd': False,
        'radiation_history': False, 'immunosuppression': False,
        'precancerous_lesions': True, 'occupational_exposure': False,
        'uv_exposure': True
    }

    model = CancerRiskModel.get_instance()
    prediction = model.predict(patient)
    print(f"\nOverall Risk: {prediction['risk_category'].upper()} "
          f"({prediction['high_risk_probability']:.2%} high-risk probability)")

    print("\nCancer-Type Breakdown (selected):")
    ct_results = model.predict_cancer_types(patient,
        selected_types=['lung', 'colorectal', 'stomach', 'bladder', 'melanoma'])
    for ct, res in ct_results.items():
        if res['applicable']:
            print(f"  {res['label']:30s} RR={res['relative_risk']:.2f}  [{res['risk_level'].upper()}]")
            if res['key_factors']:
                print(f"    Factors: {', '.join(res['key_factors'])}")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    testCancerRisk()