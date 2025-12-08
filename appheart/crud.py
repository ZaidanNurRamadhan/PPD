from sqlalchemy.orm import Session
from sqlalchemy import func, or_
from . import models, schemas
from datetime import datetime

# --- User ---
def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def create_user(db: Session, user: schemas.UserCreate):
    fake_hashed_password = user.password + "notreallyhashed"
    db_user = models.User(
        name=user.name, 
        email=user.email, 
        password_hash=fake_hashed_password, 
        role=user.role
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# --- Patient ---
def get_patient(db: Session, patient_id: int):
    return db.query(models.Patient).filter(models.Patient.id == patient_id).first()

def get_patients(db: Session, skip: int = 0, limit: int = 100, name: str = None):
    query = db.query(models.Patient)
    if name:
        query = query.filter(models.Patient.full_name.contains(name))
    return query.offset(skip).limit(limit).all()

def search_patients(db: Session, q: str):
    return db.query(models.Patient).filter(
        or_(
            models.Patient.full_name.contains(q),
            models.Patient.medical_record_number.contains(q)
        )
    ).limit(20).all()

def create_patient(db: Session, patient: schemas.PatientCreate):
    db_patient = models.Patient(**patient.dict())
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient

# --- Checkup ---
def create_checkup(db: Session, checkup: schemas.CheckupCreate, patient_id: int, probability: float, risk_label: int, risk_category: str, model_version: str, recommendations: str = None, shap_values: str = None):
    db_checkup = models.Checkup(
        **checkup.dict(),
        patient_id=patient_id,
        probability=probability,
        risk_label=risk_label,
        risk_category=risk_category,
        model_version=model_version,
        recommendations=recommendations,
        shap_values=shap_values
    )
    db.add(db_checkup)
    db.commit()
    db.refresh(db_checkup)
    return db_checkup

def get_checkups_by_patient(db: Session, patient_id: int, skip: int = 0, limit: int = 100):
    return db.query(models.Checkup).filter(models.Checkup.patient_id == patient_id).order_by(models.Checkup.created_at.desc()).offset(skip).limit(limit).all()

def get_all_checkups(db: Session, limit: int = 1000):
    return db.query(models.Checkup).order_by(models.Checkup.created_at.desc()).limit(limit).all()

# --- Analytics ---
def get_checkup_stats(db: Session):
    total_patients = db.query(models.Patient).count()
    total_checkups = db.query(models.Checkup).count()
    
    # Risk Distribution
    risk_dist = db.query(
        models.Checkup.risk_category, func.count(models.Checkup.id)
    ).group_by(models.Checkup.risk_category).all()
    
    # Average Stats
    avg_stats = db.query(
        func.avg(models.Checkup.bmi).label('avg_bmi'),
        func.avg(models.Checkup.map).label('avg_map'),
        func.avg(models.Checkup.probability).label('avg_risk')
    ).first()
    
    # Risk Factors Counts
    smokers = db.query(models.Checkup).filter(models.Checkup.smoke == 1).count()
    high_chol = db.query(models.Checkup).filter(models.Checkup.cholesterol >= 2).count()
    diabetes = db.query(models.Checkup).filter(models.Checkup.gluc >= 2).count()
    hypertension = db.query(models.Checkup).filter(models.Checkup.map > 105).count()
    
    return {
        "total_patients": total_patients,
        "total_checkups": total_checkups,
        "risk_distribution": {k: v for k, v in risk_dist},
        "averages": {
            "bmi": avg_stats.avg_bmi or 0,
            "map": avg_stats.avg_map or 0,
            "risk": avg_stats.avg_risk or 0
        },
        "risk_factors": {
            "Merokok": smokers,
            "Kolesterol Tinggi": high_chol,
            "Diabetes": diabetes,
            "Hipertensi": hypertension
        }
    }
