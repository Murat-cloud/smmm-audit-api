import os
import re
import math
import json
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
import requests

# Supabase Client
from supabase import create_client, Client

# Hafif BM25 (PyTorch gerektirmez, bellek dostudur)
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

# Veritabanı ve Auth modülleri
from database import engine, Base, get_db
import models, schemas, auth

# ============================================================
# FASTAPI UYGULAMA VE CORS AYARLARI
# ============================================================

app = FastAPI(
    title="SMMM Mizan Denetim SaaS API",
    version="4.2.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tabloları oluştur
Base.metadata.create_all(bind=engine)

# ============================================================
# ORTAM DEĞİŞKENLERİ VE KONFİGÜRASYON
# ============================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")  # Hugging Face Inference API Token (Opsiyonel)

# Supabase vector sütun boyutunuz (Varsayılan 384 - e5-small ile tam uyumlu)
# Eğer Supabase'de vector(768) kullandıysanız Render ortam değişkenine VECTOR_DIM=768 yazabilirsiniz.
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "384"))

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[Supabase] Bağlantı başarılı.")
    except Exception as e:
        print(f"[Supabase Bağlantı Hatası]: {e}")
else:
    print("[Uyarı] SUPABASE_URL veya SUPABASE_KEY tanımlı değil.")


# ============================================================
# 512 MB BELLEK DOSTU EMBEDDING & HİBRİT RERANK MOTORU
# ============================================================

def get_text_embedding(text: str, target_dim: int = VECTOR_DIM) -> List[float]:
    """
    Metnin embedding vektörünü alır.
    Supabase'deki vector(384) veya vector(768) boyutuna tam uyması için
    target_dim parametresini dinamik yönetir.
    RAM Tüketimi: 0 MB (Dış REST API üzerinden).
    """
    clean_text = text.strip().replace("\n", " ")
    if not clean_text:
        return [0.0] * target_dim

    # 1. Seçenek: HuggingFace Serverless API (intfloat/multilingual-e5-small -> 384 Boyut)
    if HF_API_KEY:
        try:
            hf_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/intfloat/multilingual-e5-small"
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            resp = requests.post(
                hf_url,
                headers=headers,
                json={"inputs": f"query: {clean_text}"},
                timeout=8
            )
            if resp.status_code == 200:
                res_json = resp.json()
                if isinstance(res_json, list) and len(res_json) > 0:
                    if isinstance(res_json[0], (int, float)):
                        return res_json[:target_dim]
                    elif isinstance(res_json[0], list):
                        return res_json[0][:target_dim]
        except Exception as e:
            print(f"[HF Inference API]: {e}")

    # 2. Seçenek: Google Gemini text-embedding-004 REST API
    # Gemini text-embedding-004 modeli 'outputDimensionality' desteği sayesinde
    # doğrudan 384 veya 768 boyutlu çıktı verebilir!
    if GEMINI_API_KEY:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={GEMINI_API_KEY}"
            payload = {
                "model": "models/text-embedding-004",
                "content": {"parts": [{"text": clean_text}]},
                "outputDimensionality": target_dim  # 384 vs 768 uyuşmazlığını çözen kilit parametre!
            }
            resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
            if resp.status_code == 200:
                values = resp.json().get("embedding", {}).get("values", [])
                if values:
                    return values
            else:
                # outputDimensionality parametresini desteklemeyen eski endpoint için fallback
                payload_fallback = {
                    "model": "models/text-embedding-004",
                    "content": {"parts": [{"text": clean_text}]}
                }
                resp_fb = requests.post(url, json=payload_fallback, headers={"Content-Type": "application/json"}, timeout=10)
                if resp_fb.status_code == 200:
                    values = resp_fb.json().get("embedding", {}).get("values", [])
                    return values[:target_dim] if values else []
        except Exception as e:
            print(f"[Gemini Embedding API]: {e}")

    return []


def lightweight_rerank(query: str, documents: List[str], top_k: int = 5) -> List[str]:
    """
    Ağır PyTorch CrossEncoder yerine 512 MB bellek dostu
    hibrit BM25 ve token örtüşme tabanlı reranker.
    """
    if not documents:
        return []
    if len(documents) <= top_k:
        return documents

    if BM25Okapi:
        try:
            tokenized_corpus = [re.findall(r"\w+", doc.lower()) for doc in documents]
            tokenized_query = re.findall(r"\w+", query.lower())
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(tokenized_query)
            scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, score in scored_docs[:top_k]]
        except Exception:
            pass

    # Token kesişimi fallback'i
    query_tokens = set(re.findall(r"\w+", query.lower()))
    def score_doc(doc: str) -> float:
        doc_tokens = set(re.findall(r"\w+", doc.lower()))
        return len(query_tokens.intersection(doc_tokens))

    scored = sorted(documents, key=score_doc, reverse=True)
    return scored[:top_k]


def search_supabase_knowledge_base(query: str, match_count: int = 6) -> str:
    """
    Supabase üzerinde kayıtlı mevzuat veya döküman veritabanında arama yapar.
    384/768 boyut hatasına karşı korumalıdır; hata olursa metin aramasına (ILIKE) düşer.
    """
    if not supabase:
        return ""

    query_embedding = get_text_embedding(query, target_dim=VECTOR_DIM)
    retrieved_texts: List[str] = []

    # 1. Supabase pgvector RPC Araması
    if query_embedding:
        try:
            rpc_res = supabase.rpc("match_documents", {
                "query_embedding": query_embedding,
                "match_threshold": 0.45,
                "match_count": match_count
            }).execute()

            if rpc_res.data:
                for row in rpc_res.data:
                    content = row.get("content") or row.get("text") or row.get("chunk") or ""
                    if content:
                        retrieved_texts.append(content)
        except Exception as e:
            # Boyut uyuşmazlığı ("different vector dimensions") veya RPC yoksa yakala
            print(f"[Supabase RPC Uyarısı - Metin aramasına geçiliyor]: {e}")

    # 2. Text / ILIKE Fallback Zinciri (Vektör aramasından sonuç dönmezse)
    if not retrieved_texts:
        try:
            query_words = [w for w in re.findall(r"\w+", query) if len(w) > 3]
            search_pattern = f"%{query_words[0]}%" if query_words else "%vergi%"

            for table_name in ["documents", "mevzuat", "knowledge_base", "law_articles"]:
                try:
                    res = supabase.table(table_name).select("*").ilike("content", search_pattern).limit(match_count).execute()
                    if res.data:
                        for row in res.data:
                            content = row.get("content") or row.get("text") or row.get("description") or ""
                            if content:
                                retrieved_texts.append(content)
                        if retrieved_texts:
                            break
                except Exception:
                    continue
        except Exception as e:
            print(f"[Supabase Metin Fallback]: {e}")

    # Reranking ile en alakalı parçaları seç
    top_docs = lightweight_rerank(query, retrieved_texts, top_k=4)
    return "\n\n---\n\n".join(top_docs)


# ============================================================
# GEMINI REST API ÇAĞRICISI
# ============================================================

def call_gemini_api(prompt: str) -> str:
    """
    Google GenAI SDK şişkinliği ve deprecation uyarıları olmaksızın
    doğrudan REST API ile yanıt üretir.
    """
    if not GEMINI_API_KEY:
        return "Gemini API Anahtarı (GEMINI_API_KEY) tanımlı değil."

    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-flash-latest"
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
            resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"].strip()
            elif resp.status_code == 429:
                last_error = "429 Kota Sınırı"
                continue
            else:
                last_error = f"HTTP {resp.status_code}: {resp.text[:120]}"
        except Exception as e:
            last_error = str(e)
            continue

    if "429" in last_error or "quota" in last_error.lower():
        return "Google Gemini API ücretsiz kota sınırına ulaşıldı. Lütfen kısa süre sonra tekrar deneyin."

    return f"Yapay zekâ yanıtı oluşturulamadı (Hata: {last_error})"


def generate_ai_executive_summary(accounts, findings, total_debit, total_credit):
    prompt = f"""
Sen kıdemli bir Yeminli Mali Müşavir (YMM) ve Bağımsız Denetçisin.

Aşağıda bir şirkete ait mizan özeti ve kural motoru tarafından tespit edilen vergi risk bulguları yer almaktadır:

- Toplam Borç Tutarı: {total_debit:,.2f} TL
- Toplam Alacak Tutarı: {total_credit:,.2f} TL
- Tespit Edilen Risk Sayısı: {len(findings)}
- Öne Çıkan Bulgular: {str(findings[:4])}

Lütfen bu verileri VUK, KVK ve Tekdüzen Hesap Planı ilkeleri açısından değerlendir.
Şirket yönetimi ve mali müşavir için 3-4 cümlelik, net, profesyonel bir Yönetici Denetim Özeti yaz.
Varsa acilen atılması gereken düzeltme adımlarını (özellikle Adat faizi, kasa fazlası ve örtülü sermaye konularında) vurgula.
"""
    return call_gemini_api(prompt)


def generate_rag_answer(question: str, context: str) -> str:
    prompt = f"""
Sen Türkiye vergi mevzuatı konusunda uzmanlaşmış bir yapay zekâ asistanısın.
Kullanıcının sorusunu yalnızca aşağıda verilen mevzuat kaynaklarına dayanarak cevapla.

KURALLAR:
1. Cevabı yalnızca verilen mevzuat metinlerine dayanarak oluştur.
2. Verilen kaynaklarda bulunmayan bir bilgiyi uydurma.
3. Her önemli hukuki açıklamanın yanında ilgili kanun ve madde numarasını belirt.
4. Birden fazla madde birlikte değerlendiriliyorsa bunu açıkça belirt.
5. Sorunun doğrudan dayanağı olan maddeyi öncelikle kullan.
6. Kaynaklardan kesin bir sonuç çıkarılamıyorsa bunu açıkça söyle.
7. Gereksiz uzun açıklamalar yapma; önce doğrudan cevabı ver, ardından gerekçeyi açıkla.
8. Kaynaklarda bulunmayan güncel oran veya cezaları tahmin etme.
9. Cevabın sonunda 'Dayanak' başlığı altında kullandığın kanun ve madde numaralarını belirt.

KULLANICI SORUSU:
{question}

MEVZUAT KAYNAKLARI:
{context}

CEVAP:
"""
    return call_gemini_api(prompt)


# ============================================================
# GÜVENLİK VE TÜRK FORMATI SAYI ÇEVİRİCİ
# ============================================================

security = HTTPBearer(auto_error=False)

def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    if not credentials:
        return None
    token = credentials.credentials
    email = auth.verify_access_token(token)
    if not email:
        return None
    user = db.query(models.User).filter(models.User.email == email).first()
    return user


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


# ============================================================
# SMMM MİZAN DENETİM KURAL MATRİSİ (YMM TERMİNOLOJİ GÜNCELLEMESİ)
# ============================================================

def load_audit_rules():
    if os.path.exists("rules.json"):
        try:
            with open("rules.json", "r", encoding="utf-8") as f:
                rules = json.load(f)
                print(f"[Bilgi] rules.json başarıyla yüklendi. Toplam kural: {len(rules)}")
                return rules
        except Exception as e:
            print(f"[Hata] rules.json okunurken hata oluştu: {e}")

    # Fallback kurallar (rules.json yoksa doğrudan devreye girer)
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
        },
        "331": {
            "prefix": "331",
            "name": "Ortaklara Borçlar",
            "check": "credit_balance_equity_risk",
            "level": "KRİTİK",
            "category": "Örtülü Sermaye ve Finansman Gider Kısıtlaması",
            "title": "331 Ortaklara Borçlar: Örtülü Sermaye ve Finansman Gider Kısıtlaması Riski",
            "law": "KVK Madde 12, KVK Madde 11/1-(i)",
            "desc": "Ortaklardan alınan borçlar özkaynakların 3 katını aştığında örtülü sermaye sayılır; faiz ve kur farkları KKEG yapılır. Ayrıca yabancı kaynaklar özkaynakları aşıyorsa finansman gider kısıtlaması doğar.",
            "journal_lines": [
                {"account": "331 Ortaklara Borçlar", "type": "BORÇ"},
                {"account": "102 Bankalar", "type": "ALACAK"}
            ]
        }
    }

AUDIT_MATRIX = load_audit_rules()


# ============================================================
# DİNAMİK MİZAN DENETİM ÇEKİRDEĞİ
# ============================================================

def run_python_audit(accounts):
    findings = []
    total_debit = 0.0
    total_credit = 0.0

    # Çift saymayı önleme kontrolü (3 haneli ana hesaplar varsa alt hesaplar toplamı şişirmesin)
    has_three_digit_codes = any(
        len(str(r.get("code", "")).strip().split(".")[0]) == 3
        for r in accounts
    )

    for row in accounts:
        raw_code = str(row.get("code", "")).strip()
        name = str(row.get("name", "")).strip()
        debit = parse_turkish_float(row.get("debit", 0))
        credit = parse_turkish_float(row.get("credit", 0))

        if has_three_digit_codes:
            if len(raw_code) == 3 or ("." not in raw_code and len(raw_code) == 3):
                total_debit += debit
                total_credit += credit
        else:
            total_debit += debit
            total_credit += credit

        debit_bal = parse_turkish_float(
            row.get("debitBal", debit - credit if debit > credit else 0)
        )
        credit_bal = parse_turkish_float(
            row.get("creditBal", credit - debit if credit > debit else 0)
        )

        for key, rule in AUDIT_MATRIX.items():
            prefix = rule.get("prefix", "")
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
                                    "account": l["account"],
                                    "type": l["type"],
                                    "amount": credit_bal
                                }
                                for l in rule.get("journal_lines", [])
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
                                    "account": l["account"],
                                    "type": l["type"],
                                    "amount": debit_bal
                                }
                                for l in rule.get("journal_lines", [])
                            ]
                        }
                    })

                elif check_type == "debit_balance_interest" and debit_bal > 0.01:
                    interest_amt = debit_bal * 0.05
                    vat_amt = interest_amt * 0.20
                    total_amt = interest_amt + vat_amt
                    j_lines = rule.get("journal_lines", [])

                    findings.append({
                        "code": raw_code,
                        "name": name,
                        "level": rule.get("level", "KRİTİK"),
                        "category": rule.get("category", "Transfer Fiyatlandırması ve Adat"),
                        "title": rule.get("title", "Adat Faiz Hesabı ve KDV Hesaplanması Gerekli"),
                        "amount": debit_bal,
                        "law": rule.get("law", "KVK Madde 13, KDVK Madde 24"),
                        "journal_suggestion": {
                            "description": rule.get("desc", "Ortaklara kullandırılan şirket paraları için adat faizi yürütülmelidir."),
                            "lines": [
                                {
                                    "account": j_lines[0]["account"] if len(j_lines) > 0 else "131 Ortaklardan Alacaklar",
                                    "type": "BORÇ",
                                    "amount": total_amt
                                },
                                {
                                    "account": j_lines[1]["account"] if len(j_lines) > 1 else "642 Faiz Gelirleri",
                                    "type": "ALACAK",
                                    "amount": interest_amt
                                },
                                {
                                    "account": j_lines[2]["account"] if len(j_lines) > 2 else "391 Hesaplanan KDV",
                                    "type": "ALACAK",
                                    "amount": vat_amt
                                }
                            ]
                        }
                    })

                # 331 Nolu Hesap İnce Ayarı: Örtülü Sermaye ve Finansman Gider Kısıtlaması
                elif check_type == "credit_balance_equity_risk" and credit_bal > 0.01:
                    findings.append({
                        "code": raw_code,
                        "name": name,
                        "level": rule.get("level", "KRİTİK"),
                        "category": rule.get("category", "Örtülü Sermaye ve Finansman Gider Kısıtlaması"),
                        "title": rule.get("title", "331 Ortaklara Borçlar: Örtülü Sermaye ve Finansman Gider Kısıtlaması Riski"),
                        "amount": credit_bal,
                        "law": rule.get("law", "KVK Madde 12, KVK Madde 11/1-(i)"),
                        "journal_suggestion": {
                            "description": rule.get("desc", "Ortaklardan alınan borçlar özkaynakların 3 katını aşarsa örtülü sermaye sayılır; aşan kısma ait faiz/kur farkı giderleri KKEG yapılır."),
                            "lines": [
                                {
                                    "account": l["account"],
                                    "type": l["type"],
                                    "amount": credit_bal
                                }
                                for l in rule.get("journal_lines", [])
                            ]
                        }
                    })

    # Mizan Denkliği
    balance_diff = abs(total_debit - total_credit)
    is_balanced = balance_diff < 1.0

    if not is_balanced and (total_debit > 0 or total_credit > 0):
        findings.insert(0, {
            "code": "DENK",
            "name": "Mizan Denkliği",
            "level": "KRİTİK",
            "category": "Mizan Denkliği",
            "title": f"Mizan Borç ve Alacak Toplamı Eşit Değil! Fark: {balance_diff:,.2f} TL",
            "amount": balance_diff,
            "law": "VUK Madde 215",
            "journal_suggestion": {
                "description": "Mizan borç ve alacak toplamları birbirine eşit olmalıdır.",
                "lines": []
            }
        })

    ai_summary = generate_ai_executive_summary(
        accounts,
        findings,
        total_debit,
        total_credit
    )

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

@app.post("/register")
def register(user: dict, db: Session = Depends(get_db)):
    email = user.get("email")
    password = user.get("password")
    full_name = user.get("full_name", "Test SMMM")
    firm_name = user.get("firm_name", "Test Mali Müşavirlik")

    db_user = db.query(models.User).filter(models.User.email == email).first()
    if db_user:
        raise HTTPException(
            status_code=400,
            detail="Bu e-posta adresi ile zaten kayıt olunmuş."
        )

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
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Geçersiz e-posta veya şifre."
        )

    access_token = auth.create_access_token(data={"sub": db_user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/audit/run")
def run_audit(payload: dict, current_user=Depends(get_current_user_optional)):
    accounts = payload.get("accounts", [])
    if not accounts:
        raise HTTPException(
            status_code=400,
            detail="Denetlenecek mizan hesapları bulunamadı."
        )

    result = run_python_audit(accounts)
    return result


@app.post("/rag/ask")
def rag_ask(payload: dict, current_user=Depends(get_current_user_optional)):
    """
    RAG Soru-Cevap Uç Noktası.
    Context payload içinde gelirse doğrudan kullanır;
    gelmezse Supabase bilgi bankasında hafif vektör/metin araması yaparak
    en alakalı mevzuatı otomatik çeker ve cevaplar.
    """
    question = payload.get("question", "").strip()
    context = payload.get("context", "").strip()

    if not question:
        raise HTTPException(status_code=400, detail="Soru gönderilmedi.")

    # Context boşsa Supabase bilgi tabanından otomatik ara
    if not context:
        context = search_supabase_knowledge_base(question)

    if not context:
        context = "Kullanıcıya özel mevzuat kaynağı bulunamadı. Genel VUK, KVK ve Türk Vergi Hukuku prensipleri çerçevesinde değerlendiriniz."

    answer = generate_rag_answer(question, context)
    return {
        "question": question,
        "context_used": bool(context),
        "answer": answer
    }


@app.get("/", response_class=HTMLResponse)
def read_root():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return (
        "<h1>SMMM Mizan Denetim API Çalışıyor</h1>"
        "<p>index.html dosyası bulunamadı.</p>"
    )
