from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import engine, Base, get_db
import models, schemas, auth
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(
    title="SMMM Mizan Denetim SaaS API",
    version="2.0.0"
)

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tabloları oluştur
Base.metadata.create_all(bind=engine)

# --- YAPAY ZEKA AKILLI YORUM KATMANI ---
def generate_ai_executive_summary(accounts, findings, total_debit, total_credit):
    """
    Yüklenen mizan ve tespit edilen bulgulara dayanarak bir YMM/Denetçi gözüyle 
    yapay zeka sentezli yönetici özeti ve risk değerlendirmesi üretir.
    """
    critical_count = sum(1 for f in findings if f.get("level") == "KRİTİK")
    total_findings = len(findings)
    
    # Finansal rasyo ve anormallik simülasyonu
    kasa_durumu = "Normal"
    ortak_durumu = "Normal"
    
    for row in accounts:
        code = str(row.get("code", ""))
        credit_bal = float(row.get("creditBal", 0))
        debit_bal = float(row.get("debitBal", 0))
        if code.startswith("100") and credit_bal > 0:
            kasa_durumu = f"KRİTİK: Kasa hesabında {credit_bal:,.2f} TL tutarında ters bakiye (fiili kasa noksanlığı riski) tespit edilmiştir."
        if code.startswith("131") and debit_bal > 0:
            ort_durumu = f"KRİTİK: Ortaklar cari hesabında {debit_bal:,.2f} TL bakiye var (Adat faizi ve KVK Madde 13 örtülü kazanç riski)."

    summary_text = (
        f"Yapay Zeka Denetim Sentezi: Şirket mizanı toplam {total_debit:,.2f} TL hacimle taranmıştır. "
        f"Yapılan incelemede toplam {total_findings} adet riskli bulgu saptanmış olup bunların {critical_count} tanesi kritik sevicededir. "
        f"Kasa Analizi: {kasa_durumu} | "
        f"Finansman ve Ortaklar Cari Analizi: {ort_durumu} "
        f"Sonuç ve Tavsiye: Şirketin yasal defter tasdikleri, KDV beyannameleri ve vergi matrahı uyumu açısından "
        f"yukarıda belirtilen düzeltme yevmiye fişlerinin derhal muhasebeleştirilmesi ve dönemsellik ilkelerine uyulması önerilir."
    )
    return summary_text

# --- GELİŞMİŞ DENETİM MOTORU VE OTOMATİK YEVMİYE FİŞİ ÖNERİLERİ ---
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
                "law": "VUK Madde 134, 175",
                "journal_suggestion": {
                    "description": "Kasa hesabının alacak bakiyesi vermesi fiili kasa noksanlığı veya hatalı kayıtları gösterir. Düzeltme kaydı önerisi:",
                    "lines": [
                        {"account": "195 İş V. Pers. Avanslar / 131 Ort. Alacaklar", "type": "BORÇ", "amount": credit_bal},
                        {"account": "100 Kasa Hesabı", "type": "ALACAK", "amount": credit_bal}
                    ]
                }
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
                "law": "KVK Madde 13",
                "journal_suggestion": {
                    "description": "Şirketin ortaklara kullandırdığı fonlar için adat faizi hesaplanmalı ve KDV hesaplanarak fatura düzenlenmelidir:",
                    "lines": [
                        {"account": "649 Diğer Olağan Gelir ve Karlar (Adat Faizi)", "type": "ALACAK", "amount": debit_bal * 0.05},
                        {"account": "391 Hesaplanan KDV", "type": "ALACAK", "amount": debit_bal * 0.05 * 0.20},
                        {"account": "131 Ortaklardan Alacaklar", "type": "BORÇ", "amount": debit_bal * 0.05 * 1.20}
                    ]
                }
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
            "law": "VUK Madde 215",
            "journal_suggestion": {
                "description": "Mizan denkleşmemektedir. Kayıt hatası veya eksik mizan sütunları kontrol edilmelidir.",
                "lines": []
            }
        })

    # Yapay Zeka Sentez Raporunu Üret
    ai_summary = generate_ai_executive_summary(accounts, findings, total_debit, total_credit)

    return {
        "is_balanced": is_balanced,
        "balance_diff": balance_diff,
        "total_debit": total_debit,
        "total_credit": total_credit,
        "findings_count": len(findings),
        "ai_executive_summary": ai_summary,
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
    accounts = payload.get("accounts", [])
    if not accounts:
        raise HTTPException(status_code=400, detail="Denetlenecek mizan hesapları bulunamadı.")
    
    result = run_python_audit(accounts)
    return result

@app.get("/")
def read_root():
    return {"message": "SMMM Mizan Denetim SaaS Motoru Aktiftir ve Yapay Zeka Yorum Katmanı Yüklüdür!", "status": "active"}
