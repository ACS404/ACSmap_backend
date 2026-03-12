"""
Treatment / Medication Tracker Model
=====================================
Adds two tables:
  - treatments   : one row per medication a user is taking
  - treatment_logs: one row per (treatment, date, time_slot) check-off

To activate, add to your __init__.py / main Flask setup:
    from model.treatment import Treatment, TreatmentLog
    db.create_all()   # or let initUsers / your migration do it

And register the blueprint in __init__.py:
    from api.treatment import treatment_api
    app.register_blueprint(treatment_api)
"""

from __init__ import app, db
from datetime import date, datetime


class Treatment(db.Model):
    """One row = one medication for one user."""

    __tablename__ = 'treatments'

    id               = db.Column(db.Integer, primary_key=True)
    user_id          = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)

    # Medication info
    medication_name  = db.Column(db.String(255), nullable=False)
    dosage           = db.Column(db.String(100), nullable=True)   # "500 mg", "1 tablet"
    frequency        = db.Column(db.String(50),  nullable=False, default='daily')
    # times: JSON list of "HH:MM" strings, e.g. ["08:00", "20:00"]
    times            = db.Column(db.JSON, nullable=True)
    color            = db.Column(db.String(7),   nullable=False, default='#e07a6a')
    notes            = db.Column(db.Text,        nullable=True)
    ai_description   = db.Column(db.Text,        nullable=True)   # cached from Claude

    # Schedule window
    start_date       = db.Column(db.Date, nullable=False, default=date.today)
    end_date         = db.Column(db.Date, nullable=True)

    active           = db.Column(db.Boolean, default=True, nullable=False)
    created_at       = db.Column(db.DateTime, default=datetime.utcnow)

    # Cascade delete logs when treatment is hard-deleted
    logs = db.relationship(
        'TreatmentLog',
        backref='treatment',
        cascade='all, delete-orphan',
        lazy='dynamic'
    )

    def read(self):
        return {
            'id':               self.id,
            'medication_name':  self.medication_name,
            'dosage':           self.dosage or '',
            'frequency':        self.frequency,
            'times':            self.times or [],
            'color':            self.color,
            'notes':            self.notes or '',
            'ai_description':   self.ai_description or '',
            'start_date':       self.start_date.isoformat() if self.start_date else None,
            'end_date':         self.end_date.isoformat()   if self.end_date   else None,
            'active':           self.active,
        }

    def __repr__(self):
        return f'<Treatment id={self.id} user={self.user_id} med={self.medication_name}>'


class TreatmentLog(db.Model):
    """
    One row = one check-off attempt for a given (treatment, date, time_slot).
    Upserted via POST /api/treatment/log so there is at most one row per combo.
    """

    __tablename__ = 'treatment_logs'

    id           = db.Column(db.Integer, primary_key=True)
    treatment_id = db.Column(db.Integer, db.ForeignKey('treatments.id'), nullable=False, index=True)
    user_id      = db.Column(db.Integer, db.ForeignKey('users.id'),      nullable=False, index=True)
    log_date     = db.Column(db.Date,    nullable=False, index=True)
    time_slot    = db.Column(db.String(10), nullable=False)  # "08:00" or "anytime"
    taken        = db.Column(db.Boolean, default=False, nullable=False)
    taken_at     = db.Column(db.DateTime, nullable=True)

    __table_args__ = (
        db.UniqueConstraint('treatment_id', 'user_id', 'log_date', 'time_slot',
                            name='uq_treatment_log'),
    )

    def read(self):
        return {
            'id':           self.id,
            'treatment_id': self.treatment_id,
            'user_id':      self.user_id,
            'log_date':     self.log_date.isoformat(),
            'time_slot':    self.time_slot,
            'taken':        self.taken,
            'taken_at':     self.taken_at.isoformat() if self.taken_at else None,
        }

    def __repr__(self):
        return (f'<TreatmentLog id={self.id} treatment={self.treatment_id} '
                f'date={self.log_date} time={self.time_slot} taken={self.taken}>')