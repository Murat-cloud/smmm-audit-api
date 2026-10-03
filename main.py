from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
import os
import google.generativeai as genai

# Proje içi modülleriniz
from database import engine, Base, get_db
import models, schemas, auth

app = FastAPI(
    title="SMMM Mizan Denetim SaaS API",
    version="4.1.0"
)

# --- EVRENSEL CORS AYARLARI ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Veritabanı tablolarını oluştur
Base.metadata.create_all(bind=engine)

# --- GÜVENLİ TÜRK FORMATI SAYI ÇEVİRİCİ ---
def parse_turkish_float(val) -> float:
    """
    Türk muhasebe programlarından (Zirve, Luca, Logo) gelen
    1.250.500,50 veya 1250500.50 gibi sayı formatlarını güvenle float yapar.
    """
    if val is None or val == "" or val == "-" or val == "None":
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    
    val_str = str(val).strip()
    if "," in val_str and "." in val_str:
        val_str = val_str.replace(".", "").replace(",", ".")
    elif "," in val_str:
        val_str = val_str.replace(",", ".")
        
    try:
        return float(val_str)
    except ValueError:
        return 0.0

# --- GEMINI YAPAY ZEKA MOTORU (Dinamik Model Keşif Mimarisi) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

CACHED_AVAILABLE_MODELS = None

def get_active_gemini_models():
    """
    Google'a doğrudan 'list_models' çağrısı yaparak bu API anahtarına
    açık olan ve generateContent destekleyen güncel modelleri bulur.
    """
    global CACHED_AVAILABLE_MODELS
    if CACHED_AVAILABLE_MODELS:
        return CACHED_AVAILABLE_MODELS

    discovered = []
    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            for m in genai.list_models():
                methods = getattr(m, 'supported_generation_methods', [])
                if 'generateContent' in methods:
                    name = m.name.replace("models/", "")
                    discovered.append(name)
            
            # Flash modellerini önceliklendir (hızlı ve hafif)
            flashes = [m for m in discovered if "flash" in m.lower()]
            others = [m for m in discovered if "flash" not in m.lower()]
            discovered = flashes + others
            print(f"[Gemini Başarılı] Aktif bulunan modeller: {discovered}")
        except Exception as e:
            print(f"[Gemini ListModels Hatası]: {e}")

    if discovered:
        CACHED_AVAILABLE_MODELS = discovered
        return CACHED_AVAILABLE_MODELS

    # API erişiminde liste gelmezse en güncel standart modeller
    return ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"]

def generate_ai_executive_summary(accounts, findings, total_debit, total_credit):
    if not GEMINI_API_KEY:
        return "⚠️ Gemini API Anahtarı (GEMINI_API_KEY) ortam değişkenlerinde tanımlı değil. Lütfen Render panelinden Environment Variables alanına anahtarınızı ekleyin."

    prompt = f"""
Sen kıdemli bir Yeminli Mali Müşavir (YMM) ve Bağımsız Denetçisin. 
Aşağıda bir şirkete ait mizan özeti ve kural motoru tarafından tespit edilen vergi risk bulguları yer almaktadır:

- Toplam Borç Tutarı: {total_debit:,.2f} TL
- Toplam Alacak Tutarı: {total_credit:,.2f} TL
- Tespit Edilen Risk Sayısı: {len(findings)}
- Öne Çıkan Bulgular: {str(findings[:4])}

Lütfen bu verileri VUK, KVK ve Tekdüzen Hesap Planı ilkeleri açısından değerlendir.
Şirket yönetimi ve mali müşavir için 3-4 cümlelik, net, profesyonel bir 'Yönetici Denetim Özeti' yaz.
Varsa acilen atılması gereken düzeltme adımlarını vurgula.
"""

    models_to_try = get_active_gemini_models()
    last_error = ""

    for model_name in models_to_try:
        clean_name = model_name.replace("models/", "")
        try:
            model = genai.GenerativeModel(clean_name)
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            last_error = str(e)
            print(f"[Gemini Log] {clean_name} modeli denenirken hata alındı: {last_error}")
            continue

    if "429" in last_error or "quota" in last_error.lower():
        return "⚠️ Google Gemini API ücretsiz kota sınırına ulaşıldı. Kural analizleri tamamlandı; yapay zekâ metin özeti için 30 saniye sonra tekrar deneyebilirsiniz."
    
    return f"Yapay zekâ yönetici özeti oluşturulamadı (Hata: {last_error})"

# --- GÜVENLİK VE ESNEK KİMLİK DOĞRULAMA ---
security = HTTPBearer(auto_error=False)

def get_current_user_optional(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    if not credentials:
        return None
    token = credentials.credentials
    email = auth.verify_access_token(token)
    if not email:
        return None
    user = db.query(models.User).filter(models.User.email == email).first()
    return user

# --- SMMM MİZAN DENETİM KURAL MATRİSİ ---
import json

# --- SMMM MİZAN DENETİM KURAL MATRİSİ (JSON'DAN OTOMATİK YÜKLENİR) ---
def load_audit_rules():
    """
    rules.json dosyasını okur. Dosya bulunamazsa veya bozuksa
    sistemin çökmemesi için temel kurallarla başlar.
    """
    if os.path.exists("rules.json"):
        try:
            with open("rules.json", "r", encoding="utf-8") as f:
                rules = json.load(f)
                print(f"[Bilgi] rules.json başarıyla yüklendi. Toplam kural: {len(rules)}")
                return rules
        except Exception as e:
            print(f"[Hata] rules.json okunurken hata oluştu: {e}")
    
    # Dosya yoksa veya hata verirse yedek olarak çalışacak temel kurallar
    return {
        "100": {
            "prefix": "100",
            "name": "Kasa Hesabı",
            "check": "credit_balance",
            "level": "KRİTİK",
            "category": "Kasa Denetimi",
            "title": "100 Kasa Hesabı Alacak Bakiyesi Veremez",
            "law": "VUK Madde 134, 175",
            "desc": "Kasa hesabı alacak bakiyesi veremez. Fiili noksanlık veya kayıt hatası işaretidir.",
            "journal_lines": [
                {"account": "131 Ort. Alacaklar", "type": "BORÇ"},
                {"account": "100 Kasa Hesabı", "type": "ALACAK"}
            ]
        }
    }

# Kuralları uygulamaya yükle
AUDIT_MATRIX = load_audit_rules()

# --- DİNAMİK MİZAN DENETİM ÇEKİRDEĞİ ---
def run_python_audit(accounts):
    findings = []
    total_debit = 0.0
    total_credit = 0.0

    # Çift saymayı önleme kontrolü (Mizanda 3 haneli ana hesaplar var mı?)
    has_three_digit_codes = any(len(str(r.get("code", "")).strip().split(".")[0]) == 3 for r in accounts)

    for row in accounts:
        raw_code = str(row.get("code", "")).strip()
        name = str(row.get("name", "")).strip()
        debit = parse_turkish_float(row.get("debit", 0))
        credit = parse_turkish_float(row.get("credit", 0))

        # Ana hesap / alt hesap ayrımıyla denkliği doğru hesaplama
        if has_three_digit_codes:
            if len(raw_code) == 3 or (not "." in raw_code and len(raw_code) == 3):
                total_debit += debit
                total_credit += credit
        else:
            total_debit += debit
            total_credit += credit

        # Bakiye hesaplama
        debit_bal = parse_turkish_float(row.get("debitBal", debit - credit if debit > credit else 0))
        credit_bal = parse_turkish_float(row.get("creditBal", credit - debit if credit > debit else 0))

        # Kural Matrisi Eşleştirmesi
        for key, rule in AUDIT_MATRIX.items():
            if raw_code == rule["prefix"] or raw_code.startswith(rule["prefix"] + ".") or raw_code.startswith(rule["prefix"]):
                
                # 1. Kural: Alacak Bakiyesi Veremez
                if rule["check"] == "credit_balance" and credit_bal > 0.01:
                    findings.append({
                        "code": raw_code, "name": name,
                        "level": rule["level"], "category": rule["category"],
                        "title": rule["title"], "amount": credit_bal, "law": rule["law"],
                        "journal_suggestion": {
                            "description": rule["desc"],
                            "lines": [{"account": l["account"], "type": l["type"], "amount": credit_bal} for l in rule["journal_lines"]]
                        }
                    })

                # 2. Kural: Borç Bakiyesi Veremez
                elif rule["check"] == "debit_balance" and debit_bal > 0.01:
                    findings.append({
                        "code": raw_code, "name": name,
                        "level": rule["level"], "category": rule["category"],
                        "title": rule["title"], "amount": debit_bal, "law": rule["law"],
                        "journal_suggestion": {
                            "description": rule["desc"],
                            "lines": [{"account": l["account"], "type": l["type"], "amount": debit_bal} for l in rule["journal_lines"]]
                        }
                    })

                # 3. Kural: 131 Adatlandırma Hesabı
                elif rule["check"] == "debit_balance_interest" and debit_bal > 0.01:
                    interest_amt = debit_bal * 0.05
                    vat_amt = interest_amt * 0.20
                    total_amt = interest_amt + vat_amt

                    j_lines = rule.get("journal_lines", [])
                    findings.append({
                        "code": raw_code, "name": name,
                        "level": rule["level"], "category": rule["category"],
                        "title": rule["title"], "amount": debit_bal, "law": rule["law"],
                        "journal_suggestion": {
                            "description": rule["desc"],
                            "lines": [
                                {"account": j_lines[0]["account"], "type": "BORÇ", "amount": total_amt},
                                {"account": j_lines[1]["account"], "type": "ALACAK", "amount": interest_amt},
                                {"account": j_lines[2]["account"], "type": "ALACAK", "amount": vat_amt}
                            ]
                        }
                    })

                # 4. Kural: 331 Örtülü Sermaye Kontrolü
                elif rule["check"] == "credit_balance_equity_risk" and credit_bal > 0.01:
                    findings.append({
                        "code": raw_code, "name": name,
                        "level": rule["level"], "category": rule["category"],
                        "title": rule["title"], "amount": credit_bal, "law": rule["law"],
                        "journal_suggestion": {
                            "description": rule["desc"],
                            "lines": [{"account": l["account"], "type": l["type"], "amount": credit_bal} for l in rule["journal_lines"]]
                        }
                    })

    # Mizan Denkliği
    balance_diff = abs(total_debit - total_credit)
    is_balanced = balance_diff < 1.0  # 1 TL altı kuruş farklarını tolere et

    if not is_balanced and (total_debit > 0 or total_credit > 0):
        findings.insert(0, {
            "code": "DENK", "name": "Mizan Denkliği",
            "level": "KRİTİK", "category": "Mizan Denkliği",
            "title": f"Mizan Borç ve Alacak Toplamı Eşit Değil! Fark: {balance_diff:,.2f} TL",
            "amount": balance_diff, "law": "VUK Madde 215",
            "journal_suggestion": {"description": "Mizan borç ve alacak toplamları birbirine eşit olmalıdır.", "lines": []}
        })

    # Gemini Yönetici Özeti
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

@app.get("/", response_class=HTMLResponse)
def read_root():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>SMMM Mizan Denetim API Çalışıyor</h1><p>index.html dosyası bulunamadı.</p>"
