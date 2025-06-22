from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class FineTuneFiles(Base):
    __tablename__ = "fine_tune_candidates"

    id = Column(Integer, primary_key=True)
    file_blob_url = Column(String, nullable=False)
    model = Column(String, nullable=False)
    doc_type = Column(String, nullable=False)  # Invoice or Bank Statement
    expected_json = Column(Text, nullable=False)
    status = Column(String, default="needs_finetune")
    timestamp = Column(DateTime, default=datetime.utcnow)
