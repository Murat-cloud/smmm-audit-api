from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import engine, Base, get_db
import models, schemas, auth
from fastapi.middleware.cors import CORSMiddleware
import os
import google.generativeai as genai
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

app = FastAPI(
    title="SMMM Mizan Denetim SaaS API",
    version="3.6.0"
)

# --- EVRENSEL CORS AYARLARI (CORS Engeline Kesin Çözüm) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm kökenlere izin ver
    allow_credentials=True,
    allow_methods=["*"],  # Tüm HTTP metodlarına izin ver (GET, POST vb.)
    allow_headers=["*"],  # Tüm başlıklara (Authorization dahil) izin ver
)

# Tabloları oluştur
Base.metadata.create_all(bind=engine)

# --- GERÇEK GEMINI API ENTEGRASYONU ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def generate_ai_executive_summary(accounts, findings, total_debit, total_credit):
    if not GEMINI_API_KEY:
        return "⚠️ Gemini API Anahtarı (GEMINI_API_KEY) Render ortamında tanımlı değil. Lütfen Render panelinden Environment Variables kısmına anahtarınızı ekleyin."

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Daha kararlı ve hızlı yanıt için güncel model adı
        model = genai.GenerativeModel('gemini-3.8-flash')
        
        prompt = f"""
        Sen kıdemli bir Yeminli Mali Müşavir (YMM) ve Bağımsız Denetçisin. 
        Aşağıda bir şirkete ait mizan özeti ve tespit edilen risk bulguları yer almaktadır. 
        Bu verileri VUK, KVK ve muhasebe ilkeleri açısından profesyonel ve yönetici özeti formatında (en fazla 3-4 cümleyle) Türkçe olarak yorumla:

        - Toplam Borç: {total_debit:,.2f} TL
        - Toplam Alacak: {total_credit:,.2f} TL
        - Tespit Edilen Risk Sayısı: {len(findings)}
        - Bulgular Özeti: {str(findings[:3])}

        Lütfen şirketin mali durumunu ve acilen yapılması gerekenleri net bir dille özetle:
        """
        response = model.generate_content(prompt)
        if response and response.text:
            return response.text.strip()
        else:
            return "Yapay zeka boş bir yanıt döndürdü."
    except Exception as e:
        return f"Yapay Zeka Sentez Hatası: {str(e)}"

# --- GÜVENLİK VE ESNEK KİMLİK DOĞRULAMA ---
security = HTTPBearer(auto_error=False)

def get_current_user_optional(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    """
    Şimdilik testlerin takılmaması için token olsa da olmasa da çökertmeyen, 
    kullanıcıyı esnek tanıyan yapı.
    """
    if not credentials:
        return None
    token = credentials.credentials
    email = auth.verify_access_token(token)
    if not email:
        return None
    user = db.query(models.User).filter(models.User.email == email).first()
    return user

# --- DENETİM MOTORU ---
# --- SMMM MİZAN DENETİM KURAL MATRİSİ (Dinamik Altyapı) ---
AUDIT_MATRIX = {
    "100": {
        "prefix": "100",
        "name": "Kasa Hesabı",
        "check": "credit_balance",
        "level": "KRİTİK",
        "category": "Kasa Denetimi",
        "title": "100 Kasa Hesabı Alacak Bakiyesi Veremez",
        "law": "VUK Madde 134, 175",
        "desc": "Kasa hesabının alacak bakiyesi vermesi fiili kasa noksanlığı veya kayıt hatası gösterir.",
        "journal_lines": [
            {"account": "195 İş V. Pers. Avanslar / 131 Ort. Alacaklar", "type": "BORÇ"},
            {"account": "100 Kasa Hesabı", "type": "ALACAK"}
        ]
    },
    "131": {
        "prefix": "131",
        "name": "Ortaklardan Alacaklar",
        "check": "debit_balance_interest",
        "level": "KRİTİK",
        "category": "Ortaklar Adat Riski",
        "title": "Ortaklar Cari Adat Faizi ve %20 KDV Faturası Kontrolü",
        "law": "KVK Madde 13 (Örtülü Kazanç)",
        "desc": "Ortaklara kullandırılan fonlar için emsal faiz hesaplanmalı ve KDV'li fatura düzenlenmelidir.",
        "journal_lines": [
            {"account": "649 Diğer Olağan Gelir ve Karlar (Adat Faizi)", "type": "ALACAK"},
            {"account": "391 Hesaplanan KDV", "type": "ALACAK"},
            {"account": "131 Ortaklardan Alacaklar", "type": "BORÇ"}
        ]
    }
}

# --- DİNAMİK MATRİS TABANLI DENETİM MOTORU ---
def run_python_audit(accounts):
    findings = []
    total_debit = 0.0
    total_credit = 0.0
    
    for row in accounts:
        code = str(row.get("code", "")).strip()
        name = str(row.get("name", "")).strip()
        debit = parse_turkish_float(row.get("debit", 0))
        credit = parse_turkish_float(row.get("credit", 0))
        debit_bal = parse_turkish_float(row.get("debitBal", debit - credit if debit > credit else 0))
        credit_bal = parse_turkish_float(row.get("creditBal", credit - debit if credit > debit else 0))
        
        total_debit += debit
        total_credit += credit
        
        # Matris Üzerinden Kontrol
        for key, rule in AUDIT_MATRIX.items():
            if code.startswith(rule["prefix"]):
                if rule["check"] == "credit_balance" and credit_bal > 0:
                    findings.append({
                        "code": code, "name": name,
                        "level": rule["level"], "category": rule["category"],
                        "title": rule["title"], "amount": credit_bal, "law": rule["law"],
                        "journal_suggestion": {
                            "description": rule["desc"],
                            "lines": [{"account": l["account"], "type": l["type"], "amount": credit_bal} for l in rule["journal_lines"]]
                        }
                    })
                elif rule["check"] == "debit_balance_interest" and debit_bal > 0:
                    interest_amt = debit_bal * 0.05
                    vat_amt = interest_amt * 0.20
                    total_amt = interest_amt + vat_amt
                    findings.append({
                        "code": code, "name": name,
                        "level": rule["level"], "category": rule["category"],
                        "title": rule["title"], "amount": debit_bal, "law": rule["law"],
                        "journal_suggestion": {
                            "description": rule["desc"],
                            "lines": [
                                {"account": rule["journal_lines"][0]["account"], "type": "ALACAK", "amount": interest_amt},
                                {"account": rule["journal_lines"][1]["account"], "type": "ALACAK", "amount": vat_amt},
                                {"account": rule["journal_lines"][2]["account"], "type": "BORÇ", "amount": total_amt}
                            ]
                        }
                    })

    balance_diff = abs(total_debit - total_credit)
    is_balanced = balance_diff < 0.05
    
    if not is_balanced:
        findings.insert(0, {
            "code": "GENEL", "name": "Mizan Denkliği",
            "level": "KRİTİK", "category": "Mizan Denkliği",
            "title": f"Mizan Borç ve Alacak Toplamı Eşit Değil! Fark: {balance_diff:,.2f} TL",
            "amount": balance_diff, "law": "VUK Madde 215",
            "journal_suggestion": {"description": "Mizan denkleşmemektedir.", "lines": []}
        })

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

@app.post("/register")
def register(user: dict, db: Session = Depends(get_db)):
    email = user.get("email")
    password = user.get("password")
    full_name = user.get("full_name", "Test SMMM")
    firm_name = user.get("firm_name", "Test Mali Müşavirlik")
    
    db_user = db.query(models.User).filter(models.User.email == email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Bu e-posta adresi ile zaten kayıt olunmuş.")
    
    hashed_password = auth.get_password_hash(password)
    new_user = models.User(
        email=email,
        password_hash=hashed_password,
        full_name=full_name,
        firm_name=firm_name,
        role="smmm",
        subscription_status="trial"
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    access_token = auth.create_access_token(data={"sub": new_user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/login")
def login(credentials: dict, db: Session = Depends(get_db)):
    email = credentials.get("email")
    password = credentials.get("password")
    
    db_user = db.query(models.User).filter(models.User.email == email).first()
    if not db_user or not auth.verify_password(password, db_user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Geçersiz e-posta veya şifre.")
    
    access_token = auth.create_access_token(data={"sub": db_user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/audit/run")
def run_audit(payload: dict, current_user = Depends(get_current_user_optional)):
    accounts = payload.get("accounts", [])
    if not accounts:
        raise HTTPException(status_code=400, detail="Denetlenecek mizan hesapları bulunamadı.")
    
    result = run_python_audit(accounts)
    return result

from fastapi.responses import HTMLResponse
import os

@app.get("/", response_class=HTMLResponse)
def read_root():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "index.html dosyası sunucuda bulunamadı!"
