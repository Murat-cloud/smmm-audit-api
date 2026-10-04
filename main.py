"""
SMMM Mizan Denetim & Vergi Mevzuatı V10 RAG SaaS API (Production Hardened)
- Global CORS Exception Handler (Failed to Fetch Korumalı)
- Esnek rules.json Okuyucu (List & Dict Uyumlu)
- 512 MB Render RAM Uyumlu
- Supabase V10 Hukuk Motoru
"""

import os
import re
import math
import json
import traceback
from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
import requests

# Supabase Client
from supabase import create_client, Client

# Hafif BM25 Motoru
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

# Veritabanı ve Güvenlik Modülleri
from database import engine, Base, get_db, SessionLocal
import models, schemas, auth


# ============================================================
# FASTAPI UYGULAMA VE CORS AYARLARI
# ============================================================

app = FastAPI(
    title="SMMM Mizan Denetim & V10 RAG SaaS API",
    version="10.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# 500 Hatası Oluştuğunda Tarayıcının "Failed to fetch" Demesini Önleyen Global CORS Kalkanı
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    err_tb = traceback.format_exc()
    print(f"[Kritik Sunucu Hatası]:\n{err_tb}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Sunucu Hatası: {str(exc)}", "type": exc.__class__.__name__},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

# Preflight Garanti Yakalayıcı
@app.options("/{full_path:path}")
async def preflight_handler(full_path: str):
    return JSONResponse(
        content={"status": "ok"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

# Tabloları oluştur
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"[Veritabanı Tablo Oluşturma Uyarısı]: {e}")


# ============================================================
# ORTAM DEĞİŞKENLERİ VE SUPABASE BAĞLANTISI
# ============================================================

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "384"))

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[Supabase] Bağlantı başarılı.")
    except Exception as e:
        print(f"[Supabase Bağlantı Hatası]: {e}")


# ============================================================
# 512 MB BELLEK DOSTU 384 BOYUTLU EMBEDDING MOTORU
# ============================================================

def get_text_embedding(text: str, target_dim: int = VECTOR_DIM) -> List[float]:
    clean_text = text.strip().replace("\n", " ")
    if not clean_text:
        return [0.0] * target_dim

    # 1. HuggingFace Serverless Inference (multilingual-e5-small)
    if HF_API_KEY:
        try:
            hf_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/intfloat/multilingual-e5-small"
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            resp = requests.post(hf_url, headers=headers, json={"inputs": f"query: {clean_text}"}, timeout=8)
            if resp.status_code == 200:
                res_json = resp.json()
                if isinstance(res_json, list) and len(res_json) > 0:
                    if isinstance(res_json[0], (int, float)):
                        return res_json[:target_dim]
                    elif isinstance(res_json[0], list):
                        return res_json[0][:target_dim]
        except Exception as e:
            print(f"[HF Embedding Hatası]: {e}")

    # 2. Google Gemini text-embedding-004 REST API
    if GEMINI_API_KEY:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={GEMINI_API_KEY}"
            payload = {
                "model": "models/text-embedding-004",
                "content": {"parts": [{"text": clean_text}]},
                "outputDimensionality": target_dim
            }
            resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
            if resp.status_code == 200:
                values = resp.json().get("embedding", {}).get("values", [])
                if values:
                    return values
        except Exception as e:
            print(f"[Gemini Embedding Hatası]: {e}")

    return []


# ============================================================
# V10 RAG ARAMA VE MEVZUAT RETRIEVAL
# ============================================================

def retrieve_v10_mevzuat(query: str, top_k: int = 4) -> Tuple[str, List[Dict[str, Any]]]:
    if not supabase:
        return "", []

    query_embedding = get_text_embedding(query, target_dim=VECTOR_DIM)
    raw_candidates: List[Dict[str, Any]] = []

    # 1. match_tax_documents RPC
    if query_embedding:
        try:
            rpc_res = supabase.rpc("match_tax_documents", {
                "query_embedding": query_embedding,
                "match_threshold": 0.30,
                "match_count": 12
            }).execute()
            if rpc_res.data:
                raw_candidates.extend(rpc_res.data)
        except Exception as e:
            print(f"[Supabase RPC Uyarısı]: {e}")

    # 2. Tablo Doğrudan Arama Fallback
    if not raw_candidates:
        try:
            q_words = [w for w in re.findall(r"\w+", query) if len(w) > 3]
            search_word = q_words[0] if q_words else "vergi"
            res = supabase.table("tax_documents").select("*").ilike("content", f"%{search_word}%").limit(10).execute()
            if res.data:
                for row in res.data:
                    row["similarity"] = 0.5
                    raw_candidates.append(row)
        except Exception as e:
            print(f"[Supabase Fallback Hatası]: {e}")

    if not raw_candidates:
        return "", []

    # Sonuçları formatla
    formatted_context_list = []
    selected_docs = raw_candidates[:top_k]
    for d in selected_docs:
        law_name = d.get("law_name") or d.get("kanun") or "Vergi Mevzuatı"
        article_no = d.get("article_no") or d.get("madde") or "İlgili Madde"
        title = d.get("title") or d.get("baslik") or ""
        content = d.get("content") or d.get("icerik") or ""
        formatted_context_list.append(f"KANUN: {law_name}\nMADDE: {article_no} - {title}\nİÇERİK:\n{content}")

    return "\n\n---\n\n".join(formatted_context_list), selected_docs


# ============================================================
# GEMINI REST API ÇAĞRICISI
# ============================================================

def call_gemini_api(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "Gemini API Anahtarı (GEMINI_API_KEY) tanımlı değil."

    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash"
    ]

    last_error = ""
    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 2048
            }
        }
        try:
            resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"].strip()
            elif resp.status_code in [429, 503]:
                last_error = f"HTTP {resp.status_code}: Model meşgul"
                continue
            else:
                last_error = f"HTTP {resp.status_code}: {resp.text[:100]}"
        except Exception as e:
            last_error = str(e)
            continue

    return f"Yapay zekâ yanıtı oluşturulamadı ({last_error})"


def generate_rag_answer(question: str, context: str) -> str:
    prompt = f"""
Sen Türkiye vergi mevzuatı konusunda uzman kıdemli bir Yeminli Mali Müşavirsin.
Kullanıcının sorusunu YALNIZCA aşağıda verilen mevzuat kaynaklarına dayanarak net bir dille yanıtla.

KURALLAR:
1. Kaynaklarda bulunmayan bilgiyi uydurma.
2. Her tespitin yanında ilgili Kanun ve Madde numarasını parantez içinde belirt (Örn: VUK Madde 374).
3. Cevabın sonunda mutlaka '### Dayanak' başlığı altında kanun maddelerini listele.

KULLANICI SORUSU:
{question}

MEVZUAT KAYNAKLARI:
{context}

CEVAP:
"""
    return call_gemini_api(prompt)


def generate_ai_executive_summary(accounts, findings, total_debit, total_credit):
    prompt = f"""
Sen kıdemli bir Bağımsız Denetçi ve Yeminli Mali Müşavirsin.
Aşağıdaki mizan denetim sonuçlarını VUK, KVK ve Tekdüzen Hesap Planı ilkelerine göre değerlendir:

- Toplam Borç: {total_debit:,.2f} TL
- Toplam Alacak: {total_credit:,.2f} TL
- Risk Sayısı: {len(findings)}
- Öne Çıkan Bulgular: {str(findings[:4])}

Şirket yönetimi ve mali müşavir için 3-4 cümlelik, net bir Yönetici Denetim Özeti yaz.
Özellikle Adat faizi, kasa fazlası ve 331 örtülü sermaye konularında atılması gereken adımları belirt.
"""
    return call_gemini_api(prompt)


# ============================================================
# SMMM MİZAN KURALLARI (HEM LİSTE HEM DICT DESTEKLİ)
# ============================================================

def load_audit_rules() -> List[Dict[str, Any]]:
    default_rules = [
        {
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
        },
        {
            "prefix": "331",
            "name": "Ortaklara Borçlar",
            "check": "credit_balance_equity_risk",
            "level": "KRİTİK",
            "category": "Örtülü Sermaye ve Finansman Gider Kısıtlaması",
            "title": "331 Ortaklara Borçlar: Örtülü Sermaye Riski",
            "law": "KVK Madde 12, KVK Madde 11/1-(i)",
            "desc": "Ortaklardan alınan borçlar özkaynakların 3 katını aşarsa örtülü sermaye sayılır.",
            "journal_lines": [
                {"account": "331 Ortaklara Borçlar", "type": "BORÇ"},
                {"account": "102 Bankalar", "type": "ALACAK"}
            ]
        }
    ]

    if os.path.exists("rules.json"):
        try:
            with open("rules.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return list(data.values())
        except Exception as e:
            print(f"[rules.json Okuma Uyarısı]: {e}")

    return default_rules

AUDIT_RULES = load_audit_rules()


def parse_turkish_float(val) -> float:
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


def run_python_audit(accounts):
    findings = []
    total_debit = 0.0
    total_credit = 0.0

    has_three_digit = any(
        len(str(r.get("code", "")).strip().split(".")[0]) == 3
        for r in accounts
    )

    for row in accounts:
        raw_code = str(row.get("code", "")).strip()
        name = str(row.get("name", "")).strip()
        debit = parse_turkish_float(row.get("debit", 0))
        credit = parse_turkish_float(row.get("credit", 0))

        if has_three_digit:
            if len(raw_code) == 3 or ("." not in raw_code and len(raw_code) == 3):
                total_debit += debit
                total_credit += credit
        else:
            total_debit += debit
            total_credit += credit

        debit_bal = parse_turkish_float(row.get("debitBal", debit - credit if debit > credit else 0))
        credit_bal = parse_turkish_float(row.get("creditBal", credit - debit if credit > debit else 0))

        for rule in AUDIT_RULES:
            prefix = str(rule.get("prefix", ""))
            if not prefix:
                continue

            if raw_code == prefix or raw_code.startswith(prefix + ".") or raw_code.startswith(prefix):
                check_type = rule.get("check")

                if check_type == "credit_balance" and credit_bal > 0.01:
                    findings.append({
                        "code": raw_code,
                        "name": name,
                        "level": rule.get("level", "ORTA"),
                        "category": rule.get("category", "Genel Denetim"),
                        "title": rule.get("title", "Alacak Bakiyesi Riski"),
                        "amount": credit_bal,
                        "law": rule.get("law", ""),
                        "journal_suggestion": {
                            "description": rule.get("desc", ""),
                            "lines": [
                                {
                                    "account": l.get("account", "Hesap"),
                                    "type": l.get("type", "BORÇ"),
                                    "amount": credit_bal
                                }
                                for l in rule.get("journal_lines", []) if isinstance(l, dict)
                            ]
                        }
                    })

                elif check_type == "debit_balance" and debit_bal > 0.01:
                    findings.append({
                        "code": raw_code,
                        "name": name,
                        "level": rule.get("level", "ORTA"),
                        "category": rule.get("category", "Genel Denetim"),
                        "title": rule.get("title", "Borç Bakiyesi Riski"),
                        "amount": debit_bal,
                        "law": rule.get("law", ""),
                        "journal_suggestion": {
                            "description": rule.get("desc", ""),
                            "lines": [
                                {
                                    "account": l.get("account", "Hesap"),
                                    "type": l.get("type", "ALACAK"),
                                    "amount": debit_bal
                                }
                                for l in rule.get("journal_lines", []) if isinstance(l, dict)
                            ]
                        }
                    })

                elif check_type == "debit_balance_interest" and debit_bal > 0.01:
                    interest_amt = debit_bal * 0.05
                    vat_amt = interest_amt * 0.20
                    total_amt = interest_amt + vat_amt

                    findings.append({
                        "code": raw_code,
                        "name": name,
                        "level": rule.get("level", "KRİTİK"),
                        "category": rule.get("category", "Adat"),
                        "title": rule.get("title", "Adat Faiz Hesabı Gerekli"),
                        "amount": debit_bal,
                        "law": rule.get("law", "KVK 13"),
                        "journal_suggestion": {
                            "description": rule.get("desc", "Adat faizi yürütülmelidir."),
                            "lines": [
                                {"account": "131 Ort. Alacaklar", "type": "BORÇ", "amount": total_amt},
                                {"account": "642 Faiz Gelirleri", "type": "ALACAK", "amount": interest_amt},
                                {"account": "391 Hesaplanan KDV", "type": "ALACAK", "amount": vat_amt}
                            ]
                        }
                    })

                elif check_type == "credit_balance_equity_risk" and credit_bal > 0.01:
                    findings.append({
                        "code": raw_code,
                        "name": name,
                        "level": rule.get("level", "KRİTİK"),
                        "category": rule.get("category", "Örtülü Sermaye"),
                        "title": rule.get("title", "Örtülü Sermaye Riski"),
                        "amount": credit_bal,
                        "law": rule.get("law", "KVK 12"),
                        "journal_suggestion": {
                            "description": rule.get("desc", "Örtülü sermaye sınırını aşan borçlanmalar."),
                            "lines": [
                                {"account": "331 Ort. Borçlar", "type": "BORÇ", "amount": credit_bal},
                                {"account": "102 Bankalar", "type": "ALACAK", "amount": credit_bal}
                            ]
                        }
                    })

    balance_diff = abs(total_debit - total_credit)
    is_balanced = balance_diff < 1.0

    if not is_balanced and (total_debit > 0 or total_credit > 0):
        findings.insert(0, {
            "code": "DENK",
            "name": "Mizan Denkliği",
            "level": "KRİTİK",
            "category": "Mizan Denkliği",
            "title": f"Mizan Borç ve Alacak Eşit Değil! Fark: {balance_diff:,.2f} TL",
            "amount": balance_diff,
            "law": "VUK Madde 215",
            "journal_suggestion": {"description": "Mizan denk olmalıdır.", "lines": []}
        })

    # AI Özetini Güvenli Al (Hata alsa bile mizan tablosu bozulmaz)
    try:
        ai_summary = generate_ai_executive_summary(accounts, findings, total_debit, total_credit)
    except Exception as e:
        ai_summary = f"Mizan analiz edildi ancak yapay zekâ özeti hazırlanamadı ({str(e)})."

    return {
        "is_balanced": is_balanced,
        "balance_diff": balance_diff,
        "total_debit": total_debit,
        "total_credit": total_credit,
        "findings_count": len(findings),
        "ai_executive_summary": ai_summary,
        "findings": findings
    }


# ============================================================
# API ENDPOINTLERİ
# ============================================================

security = HTTPBearer(auto_error=False)

def get_current_user_safe(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Veritabanı çökse bile endpoint'in çalışmasını sağlayan korumalı auth."""
    if not credentials:
        return None
    try:
        db = SessionLocal()
        try:
            token = credentials.credentials
            email = auth.verify_access_token(token)
            if not email:
                return None
            return db.query(models.User).filter(models.User.email == email).first()
        finally:
            db.close()
    except Exception:
        return None


@app.post("/register")
def register(user: dict):
    email = user.get("email")
    password = user.get("password")
    full_name = user.get("full_name", "Test SMMM")
    firm_name = user.get("firm_name", "Test Mali Müşavirlik")

    db = SessionLocal()
    try:
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
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Kayıt işlemi veritabanı hatası: {str(e)}")
    finally:
        db.close()


@app.post("/login")
def login(credentials: dict):
    email = credentials.get("email")
    password = credentials.get("password")

    db = SessionLocal()
    try:
        db_user = db.query(models.User).filter(models.User.email == email).first()
        if not db_user or not auth.verify_password(password, db_user.password_hash):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Geçersiz e-posta veya şifre.")

        access_token = auth.create_access_token(data={"sub": db_user.email})
        return {"access_token": access_token, "token_type": "bearer"}
    finally:
        db.close()


@app.post("/audit/run")
def run_audit(payload: dict, current_user=Depends(get_current_user_safe)):
    accounts = payload.get("accounts", [])
    if not accounts:
        raise HTTPException(status_code=400, detail="Denetlenecek mizan hesapları bulunamadı.")
    return run_python_audit(accounts)


@app.post("/rag/ask")
def rag_ask(payload: dict, current_user=Depends(get_current_user_safe)):
    question = payload.get("question", "").strip()
    context = payload.get("context", "").strip()

    if not question:
        raise HTTPException(status_code=400, detail="Soru boş bırakılamaz.")

    matched_sources = []
    if not context:
        context, matched_sources = retrieve_v10_mevzuat(question, top_k=4)

    if not context:
        context = "Kullanıcıya özel mevzuat maddesi eşleşmedi. Genel VUK, KVK ve KDVK prensipleri çerçevesinde yanıtlayınız."

    answer = generate_rag_answer(question, context)

    return {
        "question": question,
        "answer": answer,
        "sources_count": len(matched_sources),
        "sources": [
            {
                "law": s.get("law_name") or s.get("kanun"),
                "article": s.get("article_no") or s.get("madde"),
                "title": s.get("title") or s.get("baslik")
            }
            for s in matched_sources
        ]
    }


@app.get("/", response_class=HTMLResponse)
def read_root():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return (
        "<h1>SMMM Mizan Denetim & V10 RAG API Çalışıyor</h1>"
        "<p>API aktif. Dokümantasyon için <a href='/docs'>/docs</a> sayfasını ziyaret edin.</p>"
    )
