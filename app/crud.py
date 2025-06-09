from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime, timedelta
from typing import List, Optional
from app.models import Child, Appointment

async def get_health_alerts(db: Session, user_id: int):
    """Get health alerts for a specific user"""
    # This is a placeholder implementation - you'll need to adapt it to your actual database schema
    alerts = []
    try:
        # Get the user's information
        user = db.query(Child).filter(Child.id == user_id).first()
        if not user:
            return []
            
        # Get any active appointments
        appointments = db.query(Appointment)\
            .filter(Appointment.patient_name == user.first_name)\
            .all()
            
        # Add appointment reminders
        for appt in appointments:
            alerts.append({
                "type": "appointment",
                "message": f"Upcoming appointment on {appt.slot.start_time}",
                "severity": "info",
                "created_at": appt.created_at
            })
            
        # Add any health-related alerts (you'll need to implement this based on your needs)
        # For example, checking vaccination schedules, symptom patterns, etc.
        
        return alerts
    except Exception as e:
        print(f"Error getting health alerts: {str(e)}")
        return []

def get_provider_appointments(db: Session, provider_id: int, start_date: datetime, end_date: datetime) -> List[dict]:
    """Get all appointments for a provider within a date range"""
    query = text("""
        SELECT 
            a.appointment_date,
            c.first_name || ' ' || c.last_name as patient_name,
            TIMESTAMPDIFF(YEAR, c.date_of_birth, CURDATE()) as patient_age,
            vs.vaccine_name,
            vs.dose_number
        FROM Appointments a
        JOIN Children c ON a.child_id = c.child_id
        JOIN VaccineSchedule vs ON a.schedule_id = vs.schedule_id
        WHERE a.provider_id = :provider_id
        AND a.appointment_date BETWEEN :start_date AND :end_date
        ORDER BY a.appointment_date ASC
    """)
    
    try:
        result = db.execute(query, {
            "provider_id": provider_id,
            "start_date": start_date,
            "end_date": end_date
        })
        
        # Convert each row to a dictionary with proper formatting
        appointments = []
        for row in result:
            appointments.append({
                "appointment_date": row.appointment_date.isoformat() if row.appointment_date else None,
                "patient_name": row.patient_name,
                "patient_age": row.patient_age,
                "vaccine_name": row.vaccine_name,
                "dose_number": row.dose_number
            })
        return appointments
        
    except Exception as e:
        print(f"Error fetching provider appointments: {str(e)}")
        return []

def get_provider_patients(db: Session, provider_id: int) -> List[dict]:
    """Get all patients who have had appointments with this provider"""
    query = text("""
        SELECT DISTINCT
            c.child_id,
            c.first_name,
            c.last_name,
            c.date_of_birth,
            MAX(a.appointment_date) as last_visit
        FROM Children c
        JOIN Appointments a ON c.child_id = a.child_id
        WHERE a.provider_id = :provider_id
        GROUP BY c.child_id, c.first_name, c.last_name, c.date_of_birth
        ORDER BY last_visit DESC
    """)
    
    result = db.execute(query, {"provider_id": provider_id})
    return [dict(row) for row in result]

def get_patient_vaccination_history(db: Session, child_id: int) -> List[dict]:
    """Get vaccination history for a specific patient"""
    query = text("""
        SELECT 
            vtr.vaccine_name,
            vtr.dose_number,
            vtr.date_administered,
            vtr.administered_by,
            vtr.notes,
            vs.recommended_age_months
        FROM VaccinationTakenRecords vtr
        JOIN VaccineSchedule vs ON 
            vtr.vaccine_name = vs.vaccine_name AND 
            vtr.dose_number = vs.dose_number
        WHERE vtr.child_id = :child_id
        ORDER BY vtr.date_administered DESC
    """)
    
    result = db.execute(query, {"child_id": child_id})
    return [dict(row) for row in result]

def get_patient_health_alerts(db: Session, child_id: int) -> List[dict]:
    """Get recent health alerts for a specific patient"""
    query = text("""
        SELECT 
            disease_name,
            confidence_score,
            matching_symptoms,
            created_at
        FROM HealthAlerts
        WHERE child_id = :child_id
        ORDER BY created_at DESC
        LIMIT 5
    """)
    
    result = db.execute(query, {"child_id": child_id})
    return [dict(row) for row in result]

def get_upcoming_vaccinations(db: Session, child_id: int) -> List[dict]:
    """Get upcoming vaccinations for a specific patient"""
    query = text("""
        SELECT 
            vs.vaccine_name,
            vs.dose_number,
            vs.recommended_age_months,
            c.date_of_birth
        FROM VaccineSchedule vs
        CROSS JOIN Children c
        LEFT JOIN VaccinationTakenRecords vtr ON 
            vtr.child_id = c.child_id AND
            vtr.vaccine_name = vs.vaccine_name AND
            vtr.dose_number = vs.dose_number
        WHERE c.child_id = :child_id
        AND vtr.record_id IS NULL
        ORDER BY vs.recommended_age_months
    """)
    
    result = db.execute(query, {"child_id": child_id})
    return [dict(row) for row in result] 