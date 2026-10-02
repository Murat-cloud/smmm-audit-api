from sqlalchemy import Column, Integer, String, Boolean, Numeric, TIMESTAMP, ForeignKey, JSON
from sqlalchemy.sql import func
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=False)
    firm_name = Column(String(150))
    role = Column(String(50), default="smmm")
    subscription_status = Column(String(50), default="trial")
    created_at = Column(TIMESTAMP, server_default=func.now())

class Client(Base):
    __tablename__ = "clients"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    company_name = Column(String(200), nullable=False)
    tax_number = Column(String(20), nullable=False)
    sector = Column(String(100))
    created_at = Column(TIMESTAMP, server_default=func.now())

class TrialBalance(Base):
    __tablename__ = "trial_balances"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id", ondelete="CASCADE"))
    period = Column(String(20), nullable=False)
    file_name = Column(String(255), nullable=False)
    raw_data = Column(JSON, nullable=False)
    is_balanced = Column(Boolean, default=False)
    balance_diff = Column(Numeric(15, 2), default=0.00)
    created_at = Column(TIMESTAMP, server_default=func.now())
