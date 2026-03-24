"""
Treatment Notes Model
Stores per-user notes tied to a medication and/or a calendar date.
"""

from datetime import datetime
from __init__ import db


class TreatmentNote(db.Model):
    __tablename__ = 'treatment_notes'

    id           = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id      = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    # Optional — null means the note is a general day note, not tied to one med
    treatment_id = db.Column(db.Integer, db.ForeignKey('treatments.id'), nullable=True)
    # YYYY-MM-DD string; null means note applies to the medication in general
    date         = db.Column(db.String(10), nullable=True)
    # 'side_effect' | 'reminder' | 'general' | 'dosage' | 'symptom'
    category     = db.Column(db.String(50), nullable=True, default='general')
    content      = db.Column(db.Text, nullable=False)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id':           self.id,
            'treatment_id': self.treatment_id,
            'date':         self.date,
            'category':     self.category,
            'content':      self.content,
            'created_at':   self.created_at.isoformat(),
        }