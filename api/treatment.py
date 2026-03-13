"""
Treatment / Medication Tracker — Flask REST API
NO Flask-RESTful. Plain @blueprint.route only.
Flask-RESTful's Api/Resource hijacks OPTIONS before CORS can touch it.
Plain routes let Flask handle OPTIONS natively and Flask-CORS attach headers.
"""

import requests as http
from datetime import date, datetime

from flask import Blueprint, current_app, g, jsonify, request
from sqlalchemy.exc import IntegrityError

from __init__ import db
from model.treatment import Treatment, TreatmentLog

treatment_api = Blueprint('treatment_api', __name__, url_prefix='/api')


# ─── Auth (called manually so OPTIONS never hits it) ──────────────────────────

def _require_user():
    """Return (user, None) or (None, error_response_tuple)."""
    import jwt as pyjwt
    from model.user import User

    token = request.cookies.get(current_app.config.get('JWT_TOKEN_NAME', 'jwt_python_flask'))
    if not token:
        return None, (jsonify({'message': 'Token missing'}), 401)
    try:
        data = pyjwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
        user = User.query.filter_by(_uid=data['_uid']).first()
        if not user:
            return None, (jsonify({'message': 'User not found'}), 401)
        g.current_user = user
        return user, None
    except pyjwt.ExpiredSignatureError:
        return None, (jsonify({'message': 'Token expired'}), 401)
    except Exception:
        return None, (jsonify({'message': 'Token invalid'}), 401)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parse_date(s):
    try:
        return date.fromisoformat(s) if s else None
    except (ValueError, TypeError):
        return None


def _gemini_describe(medication_name):
    api_key = current_app.config.get('GEMINI_API_KEY')
    url = current_app.config.get('GEMINI_SERVER',
        'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent')
    if not api_key:
        return 'Add GEMINI_API_KEY to your .env to enable AI descriptions.'
    prompt = (
        f'Describe what the medication "{medication_name}" is used for in 2-3 sentences. '
        f'Include the drug class and primary indication. Do not give dosing advice.'
    )
    try:
        r = http.post(url, params={'key': api_key},
                      json={'contents': [{'parts': [{'text': prompt}]}],
                            'generationConfig': {
                                'maxOutputTokens': 1024,  # was 600 — bumped to avoid cutoff
                                'temperature': 0.2
                            }},
                      timeout=12)
        r.raise_for_status()
        return (r.json().get('candidates', [{}])[0]
                        .get('content', {})
                        .get('parts', [{}])[0]
                        .get('text', 'No description available.')).strip()
    except Exception as e:
        print(f'[Gemini] {e}')
        return 'Could not retrieve medication information.'


def _gemini_cancer_analysis(profile: dict) -> str:
    api_key = current_app.config.get('GEMINI_API_KEY')
    url = current_app.config.get('GEMINI_SERVER',
        'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent')
    if not api_key:
        return 'Add GEMINI_API_KEY to your .env to enable AI analysis.'

    prompt = f"""You are an oncology risk communication specialist. A patient has used an ACS Cancer Facts & Figures 2026-based calculator. Write a warm, clear, 3-paragraph personalized risk narrative. Do NOT use markdown or bullet points — write flowing prose only.

Patient profile:
- Age: {profile['age']}, Sex: {profile['sex']}, Race/Ethnicity: {profile['race']}
- Smoking: {profile['smoking']}{(', pack-years: ' + str(profile['packYears'])) if profile.get('packYears') else ''}
- Alcohol: {profile['alcohol']}, BMI: {profile['bmi']}, Activity: {profile['activity']}, Diet: {profile['diet']}
- Occupational exposures: {', '.join(profile['exposures']) if profile['exposures'] else 'none'}
- Family history of cancer: {profile['familyHistory']}, Genetic mutation (BRCA/Lynch): {profile['geneticMutation']}
- Personal cancer history: {profile['personalCancer']}, Hepatitis B/C: {profile['hepatitis']}, Type 2 Diabetes: {profile['diabetes']}, IBD: {profile['ibd']}, Radon exposure: {profile['radon']}, High UV exposure: {profile['uv']}

Cancer assessed: {profile['cancerLabel']}
US population baseline lifetime risk: {profile['baseRisk']}%
Estimated lifetime risk: {profile['finalRisk']}%
Overall risk multiplier: {profile['multiplier']}×
Key risk factors: {profile['factorSummary']}

Instructions:
Paragraph 1: Explain what the {profile['finalRisk']}% estimate means in plain language, how it compares to the US average, and acknowledge any race/ethnicity-specific context from ACS 2026 data.
Paragraph 2: Address the 2-3 highest-impact risk factors specifically and explain the underlying biology or mechanism in simple terms.
Paragraph 3: Give 3 specific, actionable prevention or early detection steps this person should discuss with their doctor, grounded in ACS 2026 screening guidelines. End with an encouraging but realistic tone."""

    try:
        r = http.post(url, params={'key': api_key},
                      json={'contents': [{'parts': [{'text': prompt}]}],
                            'generationConfig': {
                                'maxOutputTokens': 2048,  # was 900 — this was causing the cutoff
                                'temperature': 0.4
                            }},
                      timeout=20)
        r.raise_for_status()
        return (r.json().get('candidates', [{}])[0]
                        .get('content', {})
                        .get('parts', [{}])[0]
                        .get('text', 'No analysis available.')).strip()
    except Exception as e:
        print(f'[Gemini cancer] {e}')
        return 'Could not generate cancer risk analysis.'


# ─── /api/treatments ──────────────────────────────────────────────────────────

@treatment_api.route('/treatments', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def treatments():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    user, err = _require_user()
    if err:
        return err

    if request.method == 'GET':
        rows = Treatment.query.filter_by(user_id=user.id, active=True).order_by(Treatment.created_at).all()
        return jsonify([r.read() for r in rows])

    if request.method == 'POST':
        body = request.get_json() or {}
        name = (body.get('medication_name') or '').strip()
        if not name:
            return jsonify({'message': 'medication_name is required'}), 400
        t = Treatment(
            user_id=user.id, medication_name=name,
            dosage=(body.get('dosage') or '').strip(),
            frequency=body.get('frequency', 'daily'),
            times=body.get('times') or [],
            color=body.get('color', '#e07a6a'),
            notes=(body.get('notes') or '').strip(),
            ai_description=(body.get('ai_description') or '').strip(),
            start_date=_parse_date(body.get('start_date')) or date.today(),
            end_date=_parse_date(body.get('end_date')),
        )
        db.session.add(t)
        db.session.commit()
        return jsonify(t.read()), 201

    if request.method == 'PUT':
        body = request.get_json() or {}
        t = Treatment.query.filter_by(id=body.get('id'), user_id=user.id).first()
        if not t:
            return jsonify({'message': 'Not found'}), 404
        for f in ('medication_name', 'dosage', 'frequency', 'times', 'color', 'notes', 'ai_description'):
            if f in body:
                setattr(t, f, body[f])
        if 'start_date' in body:
            t.start_date = _parse_date(body['start_date']) or t.start_date
        if 'end_date' in body:
            t.end_date = _parse_date(body['end_date'])
        db.session.commit()
        return jsonify(t.read())

    if request.method == 'DELETE':
        t = Treatment.query.filter_by(id=(request.get_json() or {}).get('id'), user_id=user.id).first()
        if not t:
            return jsonify({'message': 'Not found'}), 404
        t.active = False
        db.session.commit()
        return jsonify({'message': 'Removed'})


# ─── /api/treatment/log ───────────────────────────────────────────────────────

@treatment_api.route('/treatment/log', methods=['GET', 'POST', 'OPTIONS'])
def treatment_log():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    user, err = _require_user()
    if err:
        return err

    if request.method == 'GET':
        d = _parse_date(request.args.get('date')) or date.today()
        logs = TreatmentLog.query.filter_by(user_id=user.id, log_date=d).all()
        return jsonify([l.read() for l in logs])

    body = request.get_json() or {}
    tid = body.get('treatment_id')
    if not tid:
        return jsonify({'message': 'treatment_id required'}), 400
    t = Treatment.query.filter_by(id=tid, user_id=user.id).first()
    if not t:
        return jsonify({'message': 'Treatment not found'}), 404

    time_slot = body.get('time_slot', 'anytime')
    taken = bool(body.get('taken', True))
    d = _parse_date(body.get('date')) or date.today()

    log = TreatmentLog.query.filter_by(treatment_id=tid, user_id=user.id, log_date=d, time_slot=time_slot).first()
    if log:
        log.taken = taken
        log.taken_at = datetime.utcnow() if taken else None
    else:
        log = TreatmentLog(treatment_id=tid, user_id=user.id, log_date=d,
                           time_slot=time_slot, taken=taken,
                           taken_at=datetime.utcnow() if taken else None)
        db.session.add(log)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        log = TreatmentLog.query.filter_by(treatment_id=tid, user_id=user.id, log_date=d, time_slot=time_slot).first()
    return jsonify(log.read())


# ─── /api/medication/info ─────────────────────────────────────────────────────

@treatment_api.route('/medication/info', methods=['GET', 'OPTIONS'])
def medication_info():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    user, err = _require_user()
    if err:
        return err

    name = (request.args.get('name') or '').strip()
    if not name:
        return jsonify({'message': 'name required'}), 400
    return jsonify({'description': _gemini_describe(name)})


@treatment_api.route('/cancer/risk-analysis', methods=['POST', 'OPTIONS'])
def cancer_risk_analysis():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    user, err = _require_user()
    if err:
        return err

    profile = request.get_json() or {}
    required = ['age', 'sex', 'race', 'smoking', 'alcohol', 'bmi',
                'activity', 'diet', 'cancer', 'cancerLabel',
                'baseRisk', 'finalRisk', 'multiplier', 'factorSummary']
    missing = [k for k in required if k not in profile]
    if missing:
        return jsonify({'message': f'Missing fields: {missing}'}), 400

    profile.setdefault('exposures', [])
    profile.setdefault('packYears', 0)
    for flag in ('familyHistory','geneticMutation','personalCancer','hepatitis','diabetes','ibd','radon','uv'):
        profile.setdefault(flag, False)

    return jsonify({'analysis': _gemini_cancer_analysis(profile)})