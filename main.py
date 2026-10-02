from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import engine, Base, get_db
import models, schemas, auth
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Her yerden gelen isteklere izin ver
    allow_credentials=True,
    allow_methods=["*"],  # Tüm HTTP metodlarına izin ver (GET, POST vb.)
    allow_headers=["*"],
)
# Tabloları oluştur
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="SMMM Mizan Denetim SaaS API",
    version="1.0.0"
)

# --- BASİT DENETİM MOTORU (Python Sürümü) ---
def run_python_audit(accounts):
    findings = []
    total_debit = 0
    total_credit = 0
    
    for row in accounts:
        code = str(row.get("code", "")).strip()
        name = str(row.get("name", "")).strip()
        debit = float(row.get("debit", 0))
        credit = float(row.get("credit", 0))
        debit_bal = float(row.get("debitBal", debit - credit if debit > credit else 0))
        credit_bal = float(row.get("creditBal", credit - debit if credit > debit else 0))
        
        total_debit += debit
        total_credit += credit
        
        # 100 Kasa Alacak Bakiyesi (Ters Bakiye) Kontrolü
        if code.startswith("100") and credit_bal > 0:
            findings.append({
                "code": code,
                "name": name,
                "level": "KRİTİK",
                "category": "Kasa Denetimi",
                "title": "100 Kasa Hesabı Alacak Bakiyesi Veremez",
                "amount": credit_bal,
                "law": "VUK Madde 134, 175"
            })
            
        # 131 Ortaklar Alacak (Adat Riski) Kontrolü
        elif code.startswith("131") and debit_bal > 0:
            findings.append({
                "code": code,
                "name": name,
                "level": "KRİTİK",
                "category": "Ortaklar Adat Riski",
                "title": "Ortaklar Cari Adat Faizi ve %20 KDV Faturası Kontrolü",
                "amount": debit_bal,
                "law": "KVK Madde 13"
            })

    balance_diff = abs(total_debit - total_credit)
    is_balanced = balance_diff < 0.05
    
    if not is_balanced:
        findings.insert(0, {
            "code": "GENEL",
            "name": "Mizan Denkliği",
            "level": "KRİTİK",
            "category": "Mizan Denkliği",
            "title": f"Mizan Borç ve Alacak Toplamı Eşit Değil! Fark: {balance_diff:,.2f} TL",
            "amount": balance_diff,
            "law": "VUK Madde 215"
        })

    return {
        "is_balanced": is_balanced,
        "balance_diff": balance_diff,
        "total_debit": total_debit,
        "total_credit": total_credit,
        "findings_count": len(findings),
        "findings": findings
    }

# --- API ENDPOINTLERİ ---

@app.post("/register", response_model=schemas.Token)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Bu e-posta adresi ile zaten kayıt olunmuş.")
    
    hashed_password = auth.get_password_hash(user.password)
    new_user = models.User(
        email=user.email,
        password_hash=hashed_password,
        full_name=user.full_name,
        firm_name=user.firm_name
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    access_token = auth.create_access_token(data={"sub": new_user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/login", response_model=schemas.Token)
def login(credentials: schemas.UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == credentials.email).first()
    if not db_user or not auth.verify_password(credentials.password, db_user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Geçersiz e-posta veya şifre.")
    
    access_token = auth.create_access_token(data={"sub": db_user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/audit/run")
def run_audit(payload: dict):
    """
    Gönderilen mizan hesap listesini alır, kural motorundan geçirerek risk raporunu döner.
    """
    accounts = payload.get("accounts", [])
    if not accounts:
        raise HTTPException(status_code=400, detail="Denetlenecek mizan hesapları bulunamadı.")
    
    result = run_python_audit(accounts)
    return result

@app.get("/")
def read_root():
    return {"message": "SMMM Mizan Denetim SaaS Motoru Aktiftir!", "status": "active"}
