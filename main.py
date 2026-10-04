"""
SMMM Mizan Denetim & Vergi Mevzuatı V10 RAG SaaS API
- V10 Hukuk Motoru: BM25 + Semantic Search + Exact Legal Phrase + Intent/Law Bonus
- Supabase 'tax_documents' (384-dim) & 'match_tax_documents' RPC Entegrasyonu
- 512 MB Render RAM Uyumlu (Sıfır PyTorch/Transformers şişkinliği)
- REST Tabanlı Gemini Yapay Zekâ Motoru
"""

import os
import re
import math
import json
from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
import requests

# Supabase Client
from supabase import create_client, Client

# Hafif BM25 Motoru (Bellek dostu saf Python kütüphanesi)
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

# Veritabanı ve Güvenlik Modülleri
from database import engine, Base, get_db
import models, schemas, auth


# ============================================================
# FASTAPI UYGULAMA VE EVRENSEL CORS AYARLARI
# ============================================================

app = FastAPI(
    title="SMMM Mizan Denetim & V10 RAG SaaS API",
    version="10.0.0",
    description="SMMM Mizan Denetim ve Türk Vergi Mevzuatı V10 RAG Arama Motoru"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SQLAlchemy Tablolarını Oluştur
Base.metadata.create_all(bind=engine)


# ============================================================
# ORTAM DEĞİŞKENLERİ VE SUPABASE BAĞLANTISI
# ============================================================

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")

# Supabase vector(384) uyumu
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "384"))

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[Supabase] Bağlantı başarıyla kuruldu.")
    except Exception as e:
        print(f"[Supabase Bağlantı Hatası]: {e}")
else:
    print("[Uyarı] SUPABASE_URL veya SUPABASE_KEY tanımlı değil!")


# ============================================================
# 512 MB BELLEK DOSTU 384 BOYUTLU EMBEDDING MOTORU
# ============================================================

def get_text_embedding(text: str, target_dim: int = VECTOR_DIM) -> List[float]:
    """
    Kullanıcı sorusunun embedding vektörünü üretir.
    Supabase'deki 'tax_documents' tablosunun vector(384) sütununa tam uyum sağlar.
    RAM Tüketimi: 0 MB (Dış API çağrısı, yerel model yüklemez).
    """
    clean_text = text.strip().replace("\n", " ")
    if not clean_text:
        return [0.0] * target_dim

    # 1. HuggingFace Serverless Inference (multilingual-e5-small)
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
            print(f"[HF Embedding]: {e}")

    # 2. Google Gemini text-embedding-004 REST API (MRL outputDimensionality: 384 desteği)
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
            else:
                # Standart çağrı fallback
                payload_fb = {
                    "model": "models/text-embedding-004",
                    "content": {"parts": [{"text": clean_text}]}
                }
                resp_fb = requests.post(url, json=payload_fb, headers={"Content-Type": "application/json"}, timeout=10)
                if resp_fb.status_code == 200:
                    values = resp_fb.json().get("embedding", {}).get("values", [])
                    return values[:target_dim] if values else []
        except Exception as e:
            print(f"[Gemini Embedding]: {e}")

    return []


# ============================================================
# COLAB V10 RAG ÇEKİRDEĞİ: HUKUKİ NİYET & SKORLAMA MOTORU
# ============================================================

LAW_KEYWORDS = {
    "VUK": ["vergi usul", "vuk", "fatura", "defter", "zamanaşımı", "ceza", "yoklama", "tebligat", "değerleme", "amortisman"],
    "KVK": ["kurumlar vergisi", "kvk", "örtülü sermaye", "transfer fiyatlandırması", "iştirak kazancı", "kanunen kabul edilmeyen"],
    "KDVK": ["katma değer", "kdv", "kdvk", "indirim", "istisna", "tevkifat", "ihraç kayıtlı", "teslim"],
    "GVK": ["gelir vergisi", "gvk", "ücret", "serbest meslek", "kira geliri", "ticari kazanç", "istisna", "beyanname"],
    "AATUHK": ["amme alacakları", "6183", "haciz", "gecikme zammı", "tecil", "ödeme emri", "ihtiyati tahakkuk"],
    "TTK": ["türk ticaret", "ttk", "genel kurul", "sermaye", "esas sözleşme", "denetim", "limited", "anonim"]
}


def detect_law_context(query: str) -> Dict[str, float]:
    """Soru içerisindeki kanun bağlamını tespit eder."""
    q_lower = query.lower()
    scores = {}
    for law, keywords in LAW_KEYWORDS.items():
        score = sum(1.5 if kw in q_lower else 0.0 for kw in keywords)
        if law.lower() in q_lower:
            score += 3.0
        scores[law] = score
    return scores


def extract_phrases(text: str) -> List[str]:
    """Hukuki anahtar terimleri ve cümle parçacıklarını çıkarır."""
    words = re.findall(r"\w+", text.lower())
    phrases = []
    for i in range(len(words) - 1):
        phrases.append(f"{words[i]} {words[i+1]}")
    if len(words) >= 3:
        for i in range(len(words) - 2):
            phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
    return phrases


def exact_phrase_score(query: str, doc_text: str) -> float:
    """Tam hukuki tabirlerin döküman içerisindeki geçiş sıklığını puanlar."""
    phrases = extract_phrases(query)
    if not phrases:
        return 0.0
    doc_lower = doc_text.lower()
    matches = sum(1.0 for p in phrases if p in doc_lower)
    return min(matches / max(len(phrases), 1), 1.0)


def legal_intent_bonus(query: str, doc_text: str) -> float:
    """Soru kökü (nedir, süresi, oranı, istisnası) ile döküman eşleşme bonusu."""
    q = query.lower()
    d = doc_text.lower()
    bonus = 0.0

    if "zamanaşımı" in q and ("zamanaşımı" in d or "tarh zamanaşımı" in d or "tahsil zamanaşımı" in d):
        bonus += 0.25
    if ("ceza" in q or "usulsüzlük" in q) and ("ceza" in d or "usulsüzlük" in d or "vergi ziyaı" in d):
        bonus += 0.20
    if ("oran" in q or "kaçtır" in q or "yüzde" in q) and ("oran" in d or "%" in d):
        bonus += 0.15
    if "istisna" in q and "istisna" in d:
        bonus += 0.20
    if ("örtülü sermaye" in q or "331" in q) and ("örtülü sermaye" in d or "özkaynak" in d):
        bonus += 0.30
    if ("adat" in q or "faiz" in q) and ("faiz" in d or "adat" in d or "transfer fiyatlandırması" in d):
        bonus += 0.25

    return bonus


def calculate_v10_score(
    semantic_sim: float,
    bm25_score: float,
    query: str,
    doc: Dict[str, Any]
) -> float:
    """
    Colab'da geliştirilen V10 Final Sıralama Formülü.
    Ağır CrossEncoder yerine anlık ve sıfır RAM ile en doğru hukuki maddeyi öne çıkarır.
    """
    doc_content = doc.get("content") or doc.get("icerik") or doc.get("text") or ""
    doc_law = (doc.get("law_name") or doc.get("kanun") or "").upper()
    doc_title = doc.get("title") or doc.get("baslik") or ""

    # 1. Tam İfade Skoru (Exact Legal Phrase)
    phrase_score = exact_phrase_score(query, doc_content)

    # 2. Hukuki Niyet Bonusu (Legal Intent)
    intent_bonus = legal_intent_bonus(query, doc_content + " " + doc_title)

    # 3. Kanun Bağlam Bonusu (Law Context Alignment)
    law_contexts = detect_law_context(query)
    law_bonus = 0.0
    for law, weight in law_contexts.items():
        if weight > 0 and (law in doc_law or law in doc_content[:150].upper()):
            law_bonus += min(weight * 0.10, 0.35)

    # 4. Madde Başlık Eşleşme Bonusu
    title_bonus = 0.0
    q_words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 3]
    for w in q_words:
        if w in doc_title.lower():
            title_bonus += 0.10

    # V10 Ağırlıklı Toplam Skor
    final_score = (
        (semantic_sim * 0.40) +
        (bm25_score * 0.25) +
        (phrase_score * 0.15) +
        intent_bonus +
        law_bonus +
        title_bonus
    )

    return round(final_score, 4)


# ============================================================
# SUPABASE V10 HİBRİT ARAMA (MATCH_TAX_DOCUMENTS + BM25)
# ============================================================

def retrieve_v10_mevzuat(query: str, top_k: int = 4) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Colab V10 Pipeline'ının canlıya alınmış halidir:
    1. 384-boyutlu embedding üretir.
    2. Supabase match_tax_documents RPC'sini çağırır.
    3. Sonuçları BM25 ve V10 skorlama fonksiyonuyla yeniden sıralar.
    4. Gemini'ye verilecek zengin mevzuat context'ini oluşturur.
    """
    if not supabase:
        return "", []

    query_embedding = get_text_embedding(query, target_dim=VECTOR_DIM)
    raw_candidates: List[Dict[str, Any]] = []

    # 1. Supabase 'match_tax_documents' RPC çağrısı
    if query_embedding:
        try:
            rpc_res = supabase.rpc("match_tax_documents", {
                "query_embedding": query_embedding,
                "match_threshold": 0.30,
                "match_count": 15
            }).execute()

            if rpc_res.data:
                for row in rpc_res.data:
                    raw_candidates.append(row)
        except Exception as e:
            print(f"[Supabase RPC Uyarısı - Standart tablo aramasına geçiliyor]: {e}")

    # 2. RPC yoksa veya boş döndüyse 'tax_documents' tablosundan direkt çek
    if not raw_candidates:
        try:
            q_words = [w for w in re.findall(r"\w+", query) if len(w) > 3]
            search_word = q_words[0] if q_words else "vergi"

            res = supabase.table("tax_documents").select("*").ilike("content", f"%{search_word}%").limit(15).execute()
            if res.data:
                for row in res.data:
                    row["similarity"] = 0.50
                    raw_candidates.append(row)
        except Exception as e:
            print(f"[Supabase Tablo Fallback]: {e}")

    if not raw_candidates:
        return "", []

    # 3. BM25 Skorlarını Hesapla
    corpus = [
        re.findall(r"\w+", (c.get("content") or c.get("icerik") or "").lower())
        for c in raw_candidates
    ]
    query_tokens = re.findall(r"\w+", query.lower())

    bm25_scores = [0.0] * len(raw_candidates)
    if BM25Okapi and corpus and any(len(doc) > 0 for doc in corpus):
        try:
            bm25 = BM25Okapi(corpus)
            raw_bm25 = bm25.get_scores(query_tokens)
            max_bm25 = max(raw_bm25) if len(raw_bm25) > 0 and max(raw_bm25) > 0 else 1.0
            bm25_scores = [score / max_bm25 for score in raw_bm25]
        except Exception:
            pass

    # 4. V10 Hibrit Skorlama ile Adayları Derecelendir
    scored_candidates = []
    for idx, cand in enumerate(raw_candidates):
        sim = float(cand.get("similarity", 0.5))
        bm25_val = bm25_scores[idx]
        v10_score = calculate_v10_score(sim, bm25_val, query, cand)

        cand["v10_score"] = v10_score
        scored_candidates.append(cand)

    scored_candidates.sort(key=lambda x: x["v10_score"], reverse=True)
    selected_docs = scored_candidates[:top_k]

    # 5. Gemini için Profesyonel Mevzuat Metni İnşa Et
    formatted_context_list = []
    for d in selected_docs:
        law_name = d.get("law_name") or d.get("kanun") or "İlgili Mevzuat"
        article_no = d.get("article_no") or d.get("madde") or "Belirtilmemiş"
        title = d.get("title") or d.get("baslik") or ""
        content = d.get("content") or d.get("icerik") or ""

        section = f"KANUN: {law_name}\nMADDE: {article_no} - {title}\nİÇERİK:\n{content}"
        formatted_context_list.append(section)

    full_context = "\n\n========================================\n\n".join(formatted_context_list)
    return full_context, selected_docs


# ============================================================
# GEMINI REST API (DEPRECATION VE ŞİŞKİNLİK KORUMALI)
# ============================================================

def call_gemini_api(prompt: str) -> str:
    """Google Gemini REST API ile doğrudan, hafif ve kararlı iletişim."""
    if not GEMINI_API_KEY:
        return "Gemini API Anahtarı (GEMINI_API_KEY) ortam değişkenlerinde tanımlı değil."

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
                "temperature": 0.15,
                "maxOutputTokens": 2048
            }
        }

        try:
            resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"].strip()
            elif resp.status_code == 429:
                last_error = "429 Ücretsiz Kota Sınırı"
                continue
            else:
                last_error = f"HTTP {resp.status_code}: {resp.text[:120]}"
        except Exception as e:
            last_error = str(e)
            continue

    if "429" in last_error or "quota" in last_error.lower():
        return "Google Gemini API ücretsiz kota sınırına ulaşıldı. Lütfen 1-2 dakika sonra tekrar deneyin."

    return f"Yapay zekâ yanıtı oluşturulamadı (Hata: {last_error})"


def generate_rag_answer(question: str, context: str) -> str:
    prompt = f"""
Sen Türkiye vergi mevzuatı, VUK, KVK, GVK, KDVK ve Tekdüzen Hesap Planı konularında uzmanlaşmış kıdemli bir Yeminli Mali Müşavir (YMM) ve Hukuk Danışmanısın.

Kullanıcının sorusunu YALNIZCA aşağıda verilen mevzuat kaynaklarına dayanarak cevapla.

KURALLAR:
1. Cevabı doğrudan verilen mevzuat maddelerine dayandır.
2. Kaynaklarda açıkça yer almayan bir bilgiyi kesinlikle uydurma.
3. Her önemli tespitin yanında ilgili Kanun ve Madde numarasını parantez içinde belirt (Örn: VUK Madde 374).
4. Birden fazla madde birlikte değerlendiriliyorsa bunu açıkça ifade et.
5. Gereksiz dolambaçlı cümleler kurma; önce doğrudan sonucu söyle, ardından hukuki gerekçesini açıkla.
6. Kaynaklardan kesin bir sonuç çıkmıyorsa bunu açıkça belirt.
7. Cevabın en sonunda mutlaka '### Dayanak' başlığı açarak kullandığın kanun ve madde numaralarını listele.

KULLANICI SORUSU:
{question}

MEVZUAT KAYNAKLARI (V10 MOTORU İLE GETİRİLDİ):
{context}

YMM DEĞERLENDİRMESİ VE CEVAP:
"""
    return call_gemini_api(prompt)


def generate_ai_executive_summary(accounts, findings, total_debit, total_credit):
    prompt = f"""
Sen kıdemli bir Bağımsız Denetçi ve Yeminli Mali Müşavirsin.
Aşağıdaki mizan denetim sonuçlarını VUK, KVK ve Tekdüzen Hesap Planı ilkelerine göre değerlendir:

- Toplam Borç Tutarı: {total_debit:,.2f} TL
- Toplam Alacak Tutarı: {total_credit:,.2f} TL
- Tespit Edilen Risk Sayısı: {len(findings)}
- Öne Çıkan Bulgular: {str(findings[:4])}

Şirket yönetimi ve mali müşavir için 3-4 cümlelik, net, profesyonel bir Yönetici Denetim Özeti yaz.
Özellikle Adat faizi, kasa fazlası ve 331 örtülü sermaye konularında acil atılması gereken düzeltme adımlarını belirt.
"""
    return call_gemini_api(prompt)


# ============================================================
# GÜVENLİK VE TÜRK SAYI FORMATI ÇEVİRİCİ
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
    return db.query(models.User).filter(models.User.email == email).first()


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
# SMMM MİZAN DENETİM KURAL MATRİSİ (YMM UYUMLU)
# ============================================================

def load_audit_rules():
    if os.path.exists("rules.json"):
        try:
            with open("rules.json", "r", encoding="utf-8") as f:
                rules = json.load(f)
                print(f"[Bilgi] rules.json başarıyla yüklendi. Kural sayısı: {len(rules)}")
                return rules
        except Exception as e:
            print(f"[Hata] rules.json okunamadı: {e}")

    # Fallback kurallar
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
            "desc": "Ortaklardan alınan borçlar dönem başı özkaynakların 3 katını aşarsa örtülü sermaye sayılır; aşan kısma ait faiz ve kur farkları KKEG yapılır.",
            "journal_lines": [
                {"account": "331 Ortaklara Borçlar", "type": "BORÇ"},
                {"account": "102 Bankalar", "type": "ALACAK"}
            ]
        }
    }

AUDIT_MATRIX = load_audit_rules()


# ============================================================
# MİZAN DENETİM ÇEKİRDEĞİ
# ============================================================

def run_python_audit(accounts):
    findings = []
    total_debit = 0.0
    total_credit = 0.0

    # Çift saymayı önleyen 3 haneli hesap kontrolü
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

        debit_bal = parse_turkish_float(row.get("debitBal", debit - credit if debit > credit else 0))
        credit_bal = parse_turkish_float(row.get("creditBal", credit - debit if credit > debit else 0))

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
                                {"account": l["account"], "type": l["type"], "amount": credit_bal}
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
                                {"account": l["account"], "type": l["type"], "amount": debit_bal}
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
                            "description": rule.get("desc", "Ortaklardan alınan borçlar özkaynakların 3 katını aşarsa örtülü sermaye sayılır; faiz/kur farkı giderleri KKEG yapılır."),
                            "lines": [
                                {"account": l["account"], "type": l["type"], "amount": credit_bal}
                                for l in rule.get("journal_lines", [])
                            ]
                        }
                    })

    # Mizan Denklik Hesabı
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
def run_audit(payload: dict, current_user=Depends(get_current_user_optional)):
    accounts = payload.get("accounts", [])
    if not accounts:
        raise HTTPException(status_code=400, detail="Denetlenecek mizan hesapları bulunamadı.")
    return run_python_audit(accounts)


@app.post("/rag/ask")
def rag_ask(payload: dict, current_user=Depends(get_current_user_optional)):
    """
    V10 RAG Uç Noktası:
    Kullanıcı sadece soruyu gönderdiğinde, Supabase 'tax_documents' tablosundan
    V10 Motoru (BM25 + Semantic + Exact Phrase + Intent) ile kanun maddelerini bulur,
    Gemini'ye ileterek mevzuata dayalı kesin cevap ve Dayanak listesi üretir.
    """
    question = payload.get("question", "").strip()
    context = payload.get("context", "").strip()

    if not question:
        raise HTTPException(status_code=400, detail="Soru boş bırakılamaz.")

    matched_sources = []
    # Eğer kullanıcı dışarıdan context göndermediyse V10 Motorunu çalıştır
    if not context:
        context, matched_sources = retrieve_v10_mevzuat(question, top_k=4)

    if not context:
        context = "Kullanıcıya özel mevzuat maddesi eşleşmedi. Genel VUK, KVK ve KDVK prensipleri çerçevesinde yanıtlayınız."

    # Gemini'ye ilet ve cevabı al
    answer = generate_rag_answer(question, context)

    return {
        "question": question,
        "answer": answer,
        "sources_count": len(matched_sources),
        "sources": [
            {
                "law": s.get("law_name") or s.get("kanun"),
                "article": s.get("article_no") or s.get("madde"),
                "title": s.get("title") or s.get("baslik"),
                "v10_score": s.get("v10_score")
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
