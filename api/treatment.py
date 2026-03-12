"""
Treatment / Medication Tracker — Flask REST API
================================================
Endpoints:
  GET    /api/treatments          — list current user's active treatments
  POST   /api/treatments          — create a treatment
  PUT    /api/treatments          — update a treatment (send id in body)
  DELETE /api/treatments          — soft-delete a treatment (send id in body)

  GET    /api/treatment/log?date=YYYY-MM-DD  — fetch check-offs for a day
  POST   /api/treatment/log                  — upsert a check-off

  GET    /api/medication/info?name=X         — Gemini-powered description

GEMINI_API_KEY and GEMINI_SERVER are read from app.config (set in __init__.py).
"""

import requests as http
from datetime import date, datetime

from flask import Blueprint, current_app, g, jsonify, request
from flask_restful import Api, Resource
from sqlalchemy.exc import IntegrityError
from __init__ import cors

from __init__ import db
from api.authorize import token_required
from model.treatment import Treatment, TreatmentLog

# ── Blueprint & API ────────────────────────────────────────────────────────────
treatment_api = Blueprint('treatment_api', __name__, url_prefix='/api')
api = Api(treatment_api)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _parse_date(s):
    """Parse ISO date string or return None."""
    try:
        return date.fromisoformat(s) if s else None
    except (ValueError, TypeError):
        return None


def _gemini_describe(medication_name: str) -> str:
    """
    Ask Gemini for a short description of a medication.
    Pulls GEMINI_API_KEY and GEMINI_SERVER from app.config so they stay
    in sync with the values loaded in __init__.py.
    """
    api_key = current_app.config.get('GEMINI_API_KEY')
    # __init__.py sets GEMINI_SERVER to gemini-2.5-flash by default
    gemini_url = current_app.config.get(
        'GEMINI_SERVER',
        'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent'
    )

    if not api_key:
        return 'AI descriptions are not configured. Add GEMINI_API_KEY to your .env.'

    prompt = (
        f'You are a helpful medical information assistant. '
        f'Describe what the medication "{medication_name}" is commonly used for '
        f'in exactly 2-3 clear sentences. Include the drug class and primary '
        f'indication. If the name does not match any known medication, say so '
        f'briefly. Do not provide dosing advice or suggest the user take any medication.'
    )

    payload = {
        'contents': [{'parts': [{'text': prompt}]}],
        'generationConfig': {'maxOutputTokens': 250, 'temperature': 0.2},
    }

    try:
        resp = http.post(
            gemini_url,
            params={'key': api_key},
            json=payload,
            timeout=12,
        )
        resp.raise_for_status()
        data = resp.json()

        text = (
            data
            .get('candidates', [{}])[0]
            .get('content', {})
            .get('parts', [{}])[0]
            .get('text', 'No description available.')
        )
        return text.strip()

    except http.exceptions.Timeout:
        return 'Medication lookup timed out. Please try again.'
    except http.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else '?'
        print(f'[MedicationInfo] Gemini HTTP {status}: {e}')
        if status == 400:
            return 'Invalid request — check that your GEMINI_API_KEY is correct.'
        if status == 429:
            return 'Gemini rate limit reached. Please wait a moment and try again.'
        return 'Could not retrieve medication information at this time.'
    except Exception as e:
        print(f'[MedicationInfo] Unexpected error: {e}')
        return 'Could not retrieve medication information at this time.'


# ── Resources ─────────────────────────────────────────────────────────────────

class TreatmentCRUD(Resource):
    """GET / POST / PUT / DELETE /api/treatments"""

    # Allow CORS preflight through without auth
    def options(self):
        return {}, 200

    @token_required()
    def get(self):
        user = g.current_user
        rows = (Treatment.query
                .filter_by(user_id=user.id, active=True)
                .order_by(Treatment.created_at)
                .all())
        return jsonify([r.read() for r in rows])

    @token_required()
    def post(self):
        user = g.current_user
        body = request.get_json() or {}

        name = (body.get('medication_name') or '').strip()
        if not name:
            return {'message': 'medication_name is required'}, 400

        t = Treatment(
            user_id         = user.id,
            medication_name = name,
            dosage          = (body.get('dosage') or '').strip(),
            frequency       = body.get('frequency', 'daily'),
            times           = body.get('times') or [],
            color           = body.get('color', '#e07a6a'),
            notes           = (body.get('notes') or '').strip(),
            ai_description  = (body.get('ai_description') or '').strip(),
            start_date      = _parse_date(body.get('start_date')) or date.today(),
            end_date        = _parse_date(body.get('end_date')),
        )
        db.session.add(t)
        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            return {'message': 'Failed to save treatment'}, 500

        return jsonify(t.read())

    @token_required()
    def put(self):
        user = g.current_user
        body = request.get_json() or {}

        tid = body.get('id')
        if not tid:
            return {'message': 'id is required'}, 400

        t = Treatment.query.filter_by(id=tid, user_id=user.id).first()
        if not t:
            return {'message': 'Treatment not found'}, 404

        for field in ('medication_name', 'dosage', 'frequency',
                      'times', 'color', 'notes', 'ai_description'):
            if field in body:
                setattr(t, field, body[field])

        if 'start_date' in body:
            t.start_date = _parse_date(body['start_date']) or t.start_date
        if 'end_date' in body:
            t.end_date = _parse_date(body['end_date'])

        db.session.commit()
        return jsonify(t.read())

    @token_required()
    def delete(self):
        user = g.current_user
        body = request.get_json() or {}

        tid = body.get('id')
        if not tid:
            return {'message': 'id is required'}, 400

        t = Treatment.query.filter_by(id=tid, user_id=user.id).first()
        if not t:
            return {'message': 'Treatment not found'}, 404

        t.active = False
        db.session.commit()
        return {'message': 'Treatment removed'}, 200


class TreatmentLogAPI(Resource):
    """GET  /api/treatment/log?date=YYYY-MM-DD
       POST /api/treatment/log
    """

    def options(self):
        return {}, 200

    @token_required()
    def get(self):
        user = g.current_user
        date_str = request.args.get('date', date.today().isoformat())
        d = _parse_date(date_str) or date.today()

        logs = TreatmentLog.query.filter_by(user_id=user.id, log_date=d).all()
        return jsonify([l.read() for l in logs])

    @token_required()
    def post(self):
        user = g.current_user
        body = request.get_json() or {}

        tid       = body.get('treatment_id')
        time_slot = body.get('time_slot', 'anytime')
        taken     = bool(body.get('taken', True))
        d         = _parse_date(body.get('date')) or date.today()

        if not tid:
            return {'message': 'treatment_id is required'}, 400

        t = Treatment.query.filter_by(id=tid, user_id=user.id).first()
        if not t:
            return {'message': 'Treatment not found'}, 404

        # Upsert
        log = TreatmentLog.query.filter_by(
            treatment_id=tid,
            user_id=user.id,
            log_date=d,
            time_slot=time_slot
        ).first()

        if log:
            log.taken    = taken
            log.taken_at = datetime.utcnow() if taken else None
        else:
            log = TreatmentLog(
                treatment_id = tid,
                user_id      = user.id,
                log_date     = d,
                time_slot    = time_slot,
                taken        = taken,
                taken_at     = datetime.utcnow() if taken else None,
            )
            db.session.add(log)

        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            log = TreatmentLog.query.filter_by(
                treatment_id=tid, user_id=user.id,
                log_date=d, time_slot=time_slot
            ).first()

        return jsonify(log.read())


class MedicationInfo(Resource):
    """GET /api/medication/info?name=Metformin"""

    # ↓ This is the critical fix — OPTIONS must return 200 without hitting @token_required
    def options(self):
        return {}, 200

    @token_required()
    def get(self):
        name = (request.args.get('name') or '').strip()
        if not name:
            return {'message': 'name is required'}, 400

        description = _gemini_describe(name)
        return jsonify({'description': description})


# ── Register routes ────────────────────────────────────────────────────────────
api.add_resource(TreatmentCRUD,   '/treatments')
api.add_resource(TreatmentLogAPI, '/treatment/log')
api.add_resource(MedicationInfo,  '/medication/info')