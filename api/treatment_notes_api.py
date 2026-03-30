"""
Treatment Notes API
GET    /api/treatment/notes            — fetch notes (filter by date and/or treatment_id)
POST   /api/treatment/notes            — create a note
DELETE /api/treatment/notes/<note_id>  — delete a note
"""

from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from __init__ import db
from model.treatment_notes import TreatmentNote

# Re-use the same auth helper your existing treatment API uses
from model.user import User  # adjust import if your project differs
import jwt, os
from functools import wraps
from flask import current_app

treatment_notes_api = Blueprint('treatment_notes_api', __name__, url_prefix='/api')
api = Api(treatment_notes_api)


# ── Auth helper (mirrors pattern used elsewhere in the project) ─────────────
def get_current_user():
    """Return the User object from the JWT cookie, or None."""
    # CORRECT - use the configured token name
    token = request.cookies.get(current_app.config.get('JWT_TOKEN_NAME', 'jwt_python_flask')) or request.headers.get('Authorization', '').replace('Bearer ', '')
    if not token:
        return None
    try:
        secret  = current_app.config.get('SECRET_KEY', os.environ.get('SECRET_KEY', 'secret'))
        payload = jwt.decode(token, secret, algorithms=['HS256'])
        uid     = payload.get('id') or payload.get('user_id') or payload.get('_uid')
        return User.query.get(uid) if uid else None
    except Exception:
        return None


# ── Resources ────────────────────────────────────────────────────────────────
class TreatmentNotesAPI(Resource):

    def get(self):
        """
        Query params (all optional):
          date         — YYYY-MM-DD   filter to a specific day
          treatment_id — integer       filter to a specific medication
        Returns all matching notes for the authenticated user, newest first.
        """
        user = get_current_user()
        if not user:
            return {'error': 'Unauthorized'}, 401

        q = TreatmentNote.query.filter_by(user_id=user.id)

        date = request.args.get('date')
        if date:
            q = q.filter_by(date=date)

        tid = request.args.get('treatment_id')
        if tid:
            q = q.filter_by(treatment_id=int(tid))

        notes = q.order_by(TreatmentNote.created_at.desc()).all()
        return jsonify([n.to_dict() for n in notes])

    def post(self):
        """
        Body JSON:
          content      (required)
          treatment_id (optional int)
          date         (optional YYYY-MM-DD)
          category     (optional string)
        """
        user = get_current_user()
        if not user:
            return {'error': 'Unauthorized'}, 401

        data = request.get_json(silent=True) or {}
        content = (data.get('content') or '').strip()
        if not content:
            return {'error': 'content is required'}, 400

        note = TreatmentNote(
            user_id      = user.id,
            treatment_id = data.get('treatment_id'),
            date         = data.get('date'),
            category     = data.get('category', 'general'),
            content      = content,
        )
        db.session.add(note)
        db.session.commit()
        return jsonify(note.to_dict())


class TreatmentNoteDeleteAPI(Resource):

    def delete(self, note_id):
        user = get_current_user()
        if not user:
            return {'error': 'Unauthorized'}, 401

        note = TreatmentNote.query.filter_by(id=note_id, user_id=user.id).first()
        if not note:
            return {'error': 'Not found'}, 404

        db.session.delete(note)
        db.session.commit()
        return {'deleted': note_id}, 200


api.add_resource(TreatmentNotesAPI,      '/treatment/notes')
api.add_resource(TreatmentNoteDeleteAPI, '/treatment/notes/<int:note_id>')