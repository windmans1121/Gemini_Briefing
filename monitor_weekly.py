import os, json, time
import requests
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google import genai

# -----------------------------
# ENV / CONFIG
# -----------------------------
SCOPUS_API_KEY = os.environ["SCOPUS_API_KEY"].strip()
EMAIL_FROM = os.environ["EMAIL_FROM"].strip()
EMAIL_TO = os.environ["EMAIL_TO"].strip()  # comma-separated
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"].strip()

# Query core (robust fallback)
QUERY_CORE = os.environ.get("SCOPUS_QUERY_CORE", "").strip()
if not QUERY_CORE:
    QUERY_CORE = 'TITLE-ABS-KEY("additive manufacturing") AND KEY("superalloys")'

# Lab context (IMPORTANT for tailored directions)
LAB_CONTEXT = os.environ.get("LAB_CONTEXT", "").strip()

# Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "").strip() or "gemini-2.0-flash"

# Scopus API
SCOPUS_URL = "https://api.elsevier.com/content/search/scopus"
SCOPUS_HEAD = {"X-ELS-APIKey": SCOPUS_API_KEY, "Accept": "application/json"}

# Your proven STANDARD field set (safe)
FIELD = (
    "eid,dc:title,dc:creator,prism:coverDate,prism:publicationName,prism:doi,"
    "prism:issn,prism:eIssn,prism:volume,prism:issueIdentifier,prism:pageRange,"
    "subtypeDescription,citedby-count,authkeywords,prism:aggregationType,prism:url,"
    "openaccess,openaccessFlag,afid,affiliation,prism:coverDisplayDate,prism:publicationDate"
)

STATE_FILE = "state.json"
SNAPSHOT_DIR = "snapshots"

KST = ZoneInfo("Asia/Seoul")


# -----------------------------
# State
# -----------------------------
def load_state():
    if not os.path.exists(STATE_FILE):
        return {"notified_eids": [], "last_report_kst": ""}
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# -----------------------------
# Scopus fetch
# -----------------------------
def scopus_search_all(query: str):
    start, count = 0, 25
    rows = []
    total = None

    while True:
        params = {
            "query": query,
            "count": count,
            "start": start,
            "view": "STANDARD",
            "field": FIELD
        }
        r = requests.get(SCOPUS_URL, headers=SCOPUS_HEAD, params=params, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Scopus API error {r.status_code}: {r.text[:900]}")

        data = r.json()
        sr = data.get("search-results", {})
        entries = sr.get("entry", [])

        if total is None:
            total = int(sr.get("opensearch:totalResults", 0))

        if not entries:
            break

        rows.extend(entries)
        start += count
        if len(rows) >= total:
            break

        time.sleep(0.2)

    # Dedup by EID
    seen = set()
    out = []
    for e in rows:
        eid = e.get("eid")
        if not eid or eid in seen:
            continue
        seen.add(eid)
        out.append(e)

    return out


# -----------------------------
# Link helpers (human-friendly)
# -----------------------------
def paper_links(e):
    eid = (e.get("eid", "") or "").strip()
    doi = (e.get("prism:doi", "") or "").strip()

    doi_link = f"https://doi.org/{doi}" if doi else ""
    scopus_web = f"https://www.scopus.com/record/display.uri?eid={eid}&origin=resultslist" if eid else ""
    return doi_link, scopus_web

def format_affiliation(aff):
    """
    Convert Scopus 'affiliation' field (string / dict / list[dict]) into clean text.
    Example:
      Guangxi University (Nanning, China); Ningbo Institute ... (Ningbo, China)
    """
    if not aff:
        return ""

    if isinstance(aff, str):
        return aff.strip()

    def one(a):
        if not isinstance(a, dict):
            return str(a)
        name = a.get("affilname", "") or a.get("affiliation-name", "") or ""
        city = a.get("affiliation-city", "") or ""
        country = a.get("affiliation-country", "") or ""
        loc = ", ".join([x for x in [city, country] if x])
        if name and loc:
            return f"{name} ({loc})"
        return name or loc or ""

    if isinstance(aff, dict):
        return one(aff)

    if isinstance(aff, list):
        parts = [one(a) for a in aff if a]
        # de-dup while preserving order
        seen = set()
        cleaned = []
        for p in parts:
            if p and p not in seen:
                seen.add(p)
                cleaned.append(p)
        return "; ".join(cleaned)

    return str(aff)

def format_affiliation_one(aff):
    """Return only ONE affiliation in a clean form."""
    if not aff:
        return ""
    if isinstance(aff, str):
        return aff.strip()

    def one(a):
        if not isinstance(a, dict):
            return str(a)
        name = a.get("affilname", "") or ""
        city = a.get("affiliation-city", "") or ""
        country = a.get("affiliation-country", "") or ""
        loc = ", ".join([x for x in [city, country] if x])
        return f"{name} ({loc})" if loc else name

    if isinstance(aff, dict):
        return one(aff)
    if isinstance(aff, list) and len(aff) > 0:
        return one(aff[0])
    return str(aff)

# -----------------------------
# Gemini: trend + directions
# -----------------------------
def gemini_client():
    if not GEMINI_API_KEY:
        return None
    return genai.Client(api_key=GEMINI_API_KEY)

def build_metadata_context(entries, cap=50):
    lines = []
    for e in entries[:cap]:
        title = e.get("dc:title", "")
        author = e.get("dc:creator", "")
        aff = format_affiliation_one(e.get("affiliation", ""))
        kw = e.get("authkeywords", "")
        journal = e.get("prism:publicationName", "")
        cover = e.get("prism:coverDate", "")
        lines.append(
            f"- Title: {title}\n"
            f"  Author: {author}\n"
            f"  Affiliation: {aff}\n"
            f"  Journal: {journal}\n"
            f"  CoverDate: {cover}\n"
            f"  Keywords: {kw}"
        )
    return "\n".join(lines)

def generate_trend_summary(entries_30d, language="ko"):
    client = gemini_client()
    if client is None or not entries_30d:
        return ""

    context = build_metadata_context(entries_30d, cap=60)

    if language.lower().startswith("ko"):
        prompt = f"""
너는 적층제조(AM) 및 초내열합금(superalloys) 분야의 연구 동향 분석가다.
아래 메타데이터(제목/키워드/저자/소속/저널)만 근거로, 지난 30일 신규 유입 논문에 대한 '한 문단(5~7문장)' 동향 요약을 작성해라.

필수 포함:
- 반복적으로 등장하는 핵심 주제/문제(예: cracking, microstructure, creep 등)
- 공정/소재 관점의 흐름(가능한 범위 내)
- 소속기관/국가의 특징이 보이면 언급(불확실하면 "근거 부족"이라고 표시)
- 과장 금지, 근거 없는 사실 생성 금지

메타데이터:
{context}
"""
    else:
        prompt = f"""
Write ONE paragraph (5–7 sentences) trend summary based ONLY on the metadata below.
Include recurring themes, materials/process trends, and notable institutions/countries if clearly supported.
Do not invent facts.

Metadata:
{context}
"""

    try:
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return (resp.text or "").strip()
    except Exception as ex:
        print("⚠️ Gemini trend summary failed:", str(ex)[:300])
        return ""

def generate_research_directions(entries_30d, lab_context, language="ko"):
    client = gemini_client()
    if client is None or not entries_30d:
        return ""

    context = build_metadata_context(entries_30d, cap=60)

    # If lab_context is missing, still produce general directions but label it
    lab_block = lab_context if lab_context else "(Lab context not provided. Provide feasible directions with common AM/superalloy lab capabilities.)"

    if language.lower().startswith("ko"):
        prompt = f"""
너는 소재/제조 분야(특히 AM 및 초내열합금) 시니어 연구기획자다.
아래 '지난 30일 신규 유입 논문 메타데이터'를 검토하고, '우리 연구실 환경'에 맞는 새로운 연구 방향 5개를 제시해라.

[우리 연구실 환경 / 제약조건]
{lab_block}

[출력 형식]
각 연구 방향마다 반드시 포함:
1) 연구 방향 제목(12단어 이내)
2) 근거(메타데이터에서 관찰되는 반복 키워드/주제/기관 분포 등)
3) 핵심 가설(1문장)
4) 최소 검증 계획(실험/분석/모델링 2~3개, 우리 연구실 환경에 맞게)
5) 기대효과(1~2문장)
6) 리스크 및 대응(불릿 1~2개)

[주의]
- 메타데이터로부터 확인 불가능한 사실은 만들지 말고 "근거 부족"이라고 명시해라.
- 논문을 하나씩 나열하지 말고 종합적으로 제안하라.

[논문 메타데이터]
{context}
"""
    else:
        prompt = f"""
Propose 5 research directions tailored to the lab context below, based ONLY on the last-30-days paper metadata.

Lab context:
{lab_block}

For each direction include:
- Title (<= 12 words)
- Rationale grounded in metadata
- Key hypothesis (1 sentence)
- Minimal validation plan (2–3 tasks feasible for the lab)
- Expected impact
- Risks + mitigation (1–2 bullets)

Do not invent facts not supported by metadata.

Metadata:
{context}
"""

    try:
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return (resp.text or "").strip()
    except Exception as ex:
        print("⚠️ Gemini research directions failed:", str(ex)[:300])
        return ""


# -----------------------------
# Email
# -----------------------------
def send_email(subject, html_body):
    to_list = [x.strip() for x in EMAIL_TO.split(",") if x.strip()]
    msg = MIMEMultipart("alternative")
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(to_list)
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_FROM, GMAIL_APP_PASSWORD)
        server.sendmail(EMAIL_FROM, to_list, msg.as_string())


def save_snapshot(entries, label):
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    pd.DataFrame(entries).to_csv(os.path.join(SNAPSHOT_DIR, f"{label}.csv"), index=False)


def build_email_html(entries_30d, cutoff_yyyymmdd, trend_summary, directions_text, new_since_last):
    # Sort newest coverDate first
    def parse_date(x):
        try:
            return pd.to_datetime(x.get("prism:coverDate", None))
        except Exception:
            return pd.NaT
    entries_30d = sorted(entries_30d, key=parse_date, reverse=True)

    trend_html = ""
    if trend_summary:
        trend_html = f"""
        <div style="padding:12px;border:1px solid #ddd;border-radius:8px;background:#fafafa;margin:12px 0;">
          <b>30-Day Trend Summary</b><br/>
          <span>{trend_summary}</span>
        </div>
        """

    directions_html = ""
    if directions_text:
        directions_html = f"""
        <div style="padding:12px;border:1px solid #ddd;border-radius:8px;background:#f6fbff;margin:12px 0;">
          <b>New Research Directions (Lab-Tailored)</b><br/>
          <span style="white-space:pre-wrap;">{directions_text}</span>
        </div>
        """

    new_block = ""
    if new_since_last:
        new_block = f"<p><b>New since last weekly report:</b> {len(new_since_last)} papers</p>"

    items = []
    for e in entries_30d[:40]:
        title = e.get("dc:title", "(no title)")
        journal = e.get("prism:publicationName", "")
        cover = e.get("prism:coverDate", "")
        doi = (e.get("prism:doi", "") or "").strip()
        author = e.get("dc:creator", "")
        aff = format_affiliation_one(e.get("affiliation", ""))
        cited = e.get("citedby-count", "")

        doi_link, scopus_web = paper_links(e)

        links = []
        if doi_link:
            links.append(f'<a href="{doi_link}">DOI</a>')
        if scopus_web:
            links.append(f'<a href="{scopus_web}">Scopus</a>')
        link_html = " | ".join(links) if links else ""

        items.append(f"""
        <li style="margin-bottom:14px;">
          <b>{title}</b><br/>
          <span>{journal} | {cover} | cited-by: {cited}</span><br/>
          <span>Author: {author}</span><br/>
          <span>Affiliation: {aff}</span><br/>
          <span>DOI: {doi}</span><br/>
          {link_html}
        </li>
        """)

    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif;">
      <h2>🧭 Weekly Scopus Monitor (Last 30 Days)</h2>
      <p>Filter: <b>({QUERY_CORE}) AND ORIG-LOAD-DATE AFT {cutoff_yyyymmdd}</b></p>
      {new_block}
      {trend_html}
      {directions_html}

      <hr/>
      <p><b>Papers (top 40 by latest coverDate)</b></p>
      <ol>
        {''.join(items)}
      </ol>

      <hr/>
      <p style="color:#777;font-size:12px;">Auto-generated weekly report.</p>
    </body>
    </html>
    """
    return html


# -----------------------------
# Main
# -----------------------------
def main():
    kst_now = datetime.now(KST)
    cutoff = (kst_now - timedelta(days=30)).strftime("%Y%m%d")

    query = f"({QUERY_CORE}) AND ORIG-LOAD-DATE AFT {cutoff}"
    print("KST now:", kst_now.isoformat())
    print("Query:", query)

    state = load_state()
    notified = set(state.get("notified_eids", []))

    entries_30d = scopus_search_all(query)
    print("Fetched entries (30d):", len(entries_30d))

    # For "new since last report" info
    new_since_last = [e for e in entries_30d if e.get("eid") and e["eid"] not in notified]

    # Save snapshots
    today_kst = kst_now.strftime("%Y%m%d")
    save_snapshot(entries_30d, f"weekly_scopus_all_30d_{today_kst}")
    save_snapshot(new_since_last, f"weekly_scopus_new_since_last_{today_kst}")

    # Gemini analysis
    print("GEMINI_API_KEY present:", bool(GEMINI_API_KEY))
    print("GEMINI_MODEL:", GEMINI_MODEL)

    trend_summary = generate_trend_summary(entries_30d, language="ko")
    directions_text = generate_research_directions(entries_30d, LAB_CONTEXT, language="ko")

    print("trend_summary length:", len(trend_summary))
    print("directions_text length:", len(directions_text))


    subject = f"[Weekly Scopus] 30-day trends + directions ({today_kst})"
    html = build_email_html(entries_30d, cutoff, trend_summary, directions_text, new_since_last)

    send_email(subject, html)
    print("✅ Weekly email sent.")

    # Update state (store all EIDs seen in this report)
    for e in entries_30d:
        eid = e.get("eid")
        if eid:
            notified.add(eid)

    state["notified_eids"] = sorted(list(notified))
    state["last_report_kst"] = kst_now.strftime("%Y-%m-%d %H:%M:%S %Z")
    save_state(state)
    print("✅ State updated.")


if __name__ == "__main__":
    main()
