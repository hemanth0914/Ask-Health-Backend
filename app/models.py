from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Child(Base):
    __tablename__ = "Children"
    
    child_id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String(50))
    last_name = Column(String(50))
    date_of_birth = Column(Date)
    gender = Column(String(10))
    blood_type = Column(String(5))
    pincode = Column(String(10))
    email = Column(String(255))
    parent_id = Column(Integer, ForeignKey("Parents.parent_id"))
    
    appointments = relationship("Appointment", back_populates="child")
    vaccination_records = relationship("VaccinationTakenRecord", back_populates="child")

class Parent(Base):
    __tablename__ = "Parents"
    
    parent_id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255))
    
    children = relationship("Child", backref="parent")

class HealthcareProvider(Base):
    __tablename__ = "HealthcareProviders"
    
    provider_id = Column(Integer, primary_key=True, index=True)
    provider_name = Column(String(100))
    specialty = Column(String(50))
    address = Column(String(255))
    pincode = Column(String(10))
    phone = Column(String(20))
    email = Column(String(255))
    
    appointments = relationship("Appointment", back_populates="provider")

class VaccineSchedule(Base):
    __tablename__ = "VaccineSchedule"
    
    schedule_id = Column(Integer, primary_key=True, index=True)
    vaccine_name = Column(String(100))
    dose_number = Column(Integer)
    recommended_age_months = Column(Integer)
    mandatory = Column(Boolean, default=True)
    description = Column(String(500))
    benefits = Column(String(500))

class VaccinationTakenRecord(Base):
    __tablename__ = "VaccinationTakenRecords"
    
    record_id = Column(Integer, primary_key=True, index=True)
    child_id = Column(Integer, ForeignKey("Children.child_id"))
    vaccine_name = Column(String(100))
    dose_number = Column(Integer)
    date_administered = Column(Date)
    administered_by = Column(String(100))
    reminder_sent = Column(Boolean, default=False)
    notes = Column(String(500))
    
    child = relationship("Child", back_populates="vaccination_records")

class Appointment(Base):
    __tablename__ = "Appointments"
    
    appointment_id = Column(Integer, primary_key=True, index=True)
    child_id = Column(Integer, ForeignKey("Children.child_id"))
    provider_id = Column(Integer, ForeignKey("HealthcareProviders.provider_id"))
    schedule_id = Column(Integer, ForeignKey("VaccineSchedule.schedule_id"), nullable=True)
    appointment_date = Column(DateTime)
    is_general_appointment = Column(Boolean, default=False)
    notes = Column(String(500), nullable=True)
    
    child = relationship("Child", back_populates="appointments")
    provider = relationship("HealthcareProvider", back_populates="appointments")

class HealthAlert(Base):
    __tablename__ = "HealthAlerts"
    
    alert_id = Column(Integer, primary_key=True, index=True)
    child_id = Column(Integer, ForeignKey("Children.child_id"))
    disease_name = Column(String(100))
    confidence_score = Column(Integer)
    matching_symptoms = Column(String(1000))  # JSON string
    created_at = Column(DateTime)

class Disease(Base):
    __tablename__ = "Diseases"
    
    disease_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100))
    primary_symptoms = Column(String(500))  # Comma-separated
    secondary_symptoms = Column(String(500))  # Comma-separated

class ScreeningResults(Base):
    __tablename__ = "ScreeningResults"
    
    screening_id = Column(Integer, primary_key=True, index=True)
    child_id = Column(Integer, ForeignKey("Children.child_id"))
    screening_type = Column(String(50))  # e.g., 'measles'
    risk_level = Column(String(20))  # HIGH, MODERATE, LOW
    summary = Column(String(1000))
    created_at = Column(DateTime) 