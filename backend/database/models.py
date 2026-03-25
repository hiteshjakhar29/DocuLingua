import uuid
import datetime
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Text, Boolean, Integer, Index
from sqlalchemy.orm import relationship
from backend.database.database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String)
    file_path = Column(String, nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    file_type = Column(String)
    file_size = Column(Integer)
    status = Column(String, default="uploaded")

    # Relationships
    extractions = relationship("Extraction", back_populates="document", cascade="all, delete-orphan")

    # Create specialized physical index for performance
    __table_args__ = (
        Index('idx_document_status', 'status'),
    )


class Extraction(Base):
    __tablename__ = "extractions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    document_class = Column(String)
    detected_language = Column(String)
    raw_text = Column(Text)
    processing_time_seconds = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="extractions")
    entities = relationship("Entity", back_populates="extraction", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_extraction_class', 'document_class'),
    )


class Entity(Base):
    __tablename__ = "entities"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    extraction_id = Column(String(36), ForeignKey("extractions.id", ondelete="CASCADE"), nullable=False)
    entity_type = Column(String)  # PERSON/INSTITUTION/DEGREE/DATE/CERT_NUMBER/GRADE
    entity_value = Column(String)
    confidence_score = Column(Float)
    manual_review_required = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    extraction = relationship("Extraction", back_populates="entities")

    __table_args__ = (
        Index('idx_entity_confidence', 'confidence_score'),
    )
