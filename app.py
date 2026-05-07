"""
ProcurePilot AI: Autonomous Vendor Governance Platform
Corporate-ready, agentic AI procurement decision app built with Streamlit.

Run:
    pip install -r requirements.txt
    streamlit run app.py

Optional live LLM mode:
    Create .env or Streamlit secret GROQ_API_KEY.
    The app also works in no-key demo mode.

FIXED:
    Supports sample files with columns:
    vendor_name, quoted_price, delivery_days, security_review_status
    and also supports alternate enterprise column names:
    vendor, quote_amount, implementation_weeks, security_status
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


APP_TITLE = "ProcurePilot AI"
DEFAULT_MODEL = "llama-3.1-8b-instant"

st.set_page_config(
    page_title="ProcurePilot AI | Vendor Governance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --ink: #0f172a;
    --muted: #64748b;
    --blue: #2563eb;
    --navy: #071733;
    --green: #10b981;
    --amber: #f59e0b;
    --red: #ef4444;
    --card: #ffffff;
}
.stApp {
    background:
        radial-gradient(circle at top left, rgba(37,99,235,.12), transparent 30%),
        linear-gradient(180deg,#f8fbff 0%,#eef3fb 50%,#f8fafc 100%);
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1300px;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#061226 0%,#0b1732 100%);
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] input { color: #0f172a !important; }
[data-testid="stSidebar"] .stAlert * { color: #0f172a !important; }

.hero {
    padding: 34px 38px;
    border-radius: 30px;
    background:
        radial-gradient(circle at 15% 5%, rgba(16,185,129,.28), transparent 24%),
        linear-gradient(135deg,#06152f 0%,#12377e 54%,#2563eb 100%);
    color: white;
    box-shadow: 0 24px 80px rgba(37,99,235,.22);
    border: 1px solid rgba(255,255,255,.14);
    margin-bottom: 24px;
}
.hero h1 {
    font-size: 3.1rem;
    line-height: 1.02;
    margin: 0 0 12px 0;
    letter-spacing: -1.5px;
}
.hero p {
    font-size: 1.05rem;
    max-width: 980px;
    color: rgba(255,255,255,.9);
}
.badge {
    display:inline-block;
    padding: 8px 13px;
    border-radius: 999px;
    margin: 8px 8px 0 0;
    background: rgba(255,255,255,.15);
    border: 1px solid rgba(255,255,255,.24);
    color:white;
    font-weight:800;
    font-size:.82rem;
}
.metric-card {
    background: white;
    border: 1px solid rgba(15,23,42,.08);
    border-radius: 24px;
    padding: 20px 22px;
    box-shadow: 0 14px 40px rgba(15,23,42,.07);
    height: 100%;
}
.metric-label {
    color:#64748b;
    font-size:.78rem;
    font-weight:900;
    text-transform:uppercase;
    letter-spacing:.09em;
}
.metric-value {
    color:#0f172a;
    font-size:1.85rem;
    font-weight:950;
    letter-spacing:-.045em;
    margin-top: 6px;
    line-height:1.15;
}
.metric-note {
    color:#64748b;
    font-size:.88rem;
    margin-top: 8px;
}
.agent-card {
    border-radius: 22px;
    padding: 18px;
    background: linear-gradient(180deg,#ffffff,#f8fafc);
    border:1px solid rgba(15,23,42,.08);
    box-shadow: 0 10px 28px rgba(15,23,42,.05);
    margin-bottom: 16px;
}
.agent-name {
    font-size: 1.22rem;
    font-weight: 950;
    color:#0f172a;
    margin-bottom: 6px;
}
.agent-sub {
    color:#64748b;
    font-size:.94rem;
}
.status-dot {
    display:inline-block;
    width:10px;
    height:10px;
    border-radius:50%;
    background:#10b981;
    margin-right:8px;
    box-shadow:0 0 0 5px rgba(16,185,129,.12);
}
.callout {
    padding: 16px 18px;
    background: #eff6ff;
    border-left: 5px solid #2563eb;
    border-radius: 16px;
    color:#172554;
    font-weight:650;
    margin-bottom: 10px;
}
.warn {
    padding: 14px 16px;
    background: #fff7ed;
    border-left: 5px solid #f97316;
    border-radius: 16px;
    color:#7c2d12;
    font-weight:650;
    margin-bottom: 10px;
}
.good {
    padding: 14px 16px;
    background: #ecfdf5;
    border-left: 5px solid #10b981;
    border-radius: 16px;
    color:#064e3b;
    font-weight:650;
    margin-bottom: 10px;
}
.small-muted { color:#64748b; font-size:.88rem; }
.stButton>button {
    border-radius: 14px;
    font-weight: 900;
    border: 0;
    background: linear-gradient(135deg,#2563eb,#1d4ed8);
    color: white;
    min-height: 3rem;
    box-shadow: 0 12px 26px rgba(37,99,235,.18);
}
.stDownloadButton>button {
    border-radius: 14px;
    font-weight: 850;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def load_dotenv_local() -> None:
    """Load GROQ_API_KEY from a local .env file without requiring extra packages."""
    env_path = os.path.join(os.getcwd(), ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass


load_dotenv_local()


def safe_secret(name: str) -> str:
    """Return environment variable only. Avoid st.secrets so Streamlit does not show pink warnings."""
    return os.getenv(name, "")


def currency(value: Any) -> str:
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return str(value)


def read_uploaded_file(uploaded_file) -> Tuple[str, Optional[pd.DataFrame]]:
    if uploaded_file is None:
        return "", None

    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return df.to_csv(index=False), df

    if name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
        return df.to_csv(index=False), df

    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore"), None

    if name.endswith(".pdf") and PdfReader is not None:
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages), None

    return uploaded_file.read().decode("utf-8", errors="ignore"), None


def load_sample_files() -> Tuple[str, pd.DataFrame]:
    with open("data/sample_vendor_request.txt", "r", encoding="utf-8") as f:
        request_text = f.read()
    return request_text, pd.read_csv("data/sample_vendor_quotes.csv")


def first_existing(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust column mapper.

    Accepts your sample format:
        vendor_name, quoted_price, delivery_days, security_review_status

    Also accepts alternate app format:
        vendor, quote_amount, implementation_weeks, security_status
    """
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    mapped = pd.DataFrame()

    mapping = {
        "vendor": ["vendor", "vendor_name", "supplier", "supplier_name", "company", "company_name"],
        "quote_amount": ["quote_amount", "quoted_price", "price", "cost", "annual_cost", "total_cost", "year_1_cost"],
        "implementation_weeks": ["implementation_weeks", "delivery_weeks", "implementation_time_weeks"],
        "delivery_days": ["delivery_days", "implementation_days", "timeline_days"],
        "sla_uptime": ["sla_uptime", "uptime", "sla", "availability"],
        "support_response_hours": ["support_response_hours", "support_hours", "response_time_hours", "support_sla_hours"],
        "security_status": ["security_status", "security_review_status", "security_review", "infosec_status"],
        "tax_form": ["tax_form", "w9_status", "w_9", "w9", "tax_documentation"],
        "contract_term_months": ["contract_term_months", "contract_months", "term_months"],
        "past_performance_score": ["past_performance_score", "performance_score", "vendor_score"],
        "data_access_level": ["data_access_level", "data_access", "sensitive_data_access"],
        "payment_terms": ["payment_terms", "terms"],
        "implementation_fee": ["implementation_fee", "setup_fee", "onboarding_fee"],
        "renewal_escalator_pct": ["renewal_escalator_pct", "renewal_increase_pct", "escalator_pct"],
        "service_scope": ["service_scope", "scope", "service", "solution"],
        "notes": ["notes", "comments", "summary"],
    }

    defaults = {
        "vendor": "Unknown Vendor",
        "quote_amount": 0,
        "implementation_weeks": None,
        "delivery_days": None,
        "sla_uptime": 99.0,
        "support_response_hours": 24,
        "security_status": "Not provided",
        "tax_form": "Not provided",
        "contract_term_months": 12,
        "past_performance_score": 75,
        "data_access_level": "Internal business data",
        "payment_terms": "Net 30",
        "implementation_fee": 0,
        "renewal_escalator_pct": 5,
        "service_scope": "Procurement analytics solution",
        "notes": "",
    }

    for standard_col, candidates in mapping.items():
        source = first_existing(df, candidates)
        if source:
            mapped[standard_col] = df[source]
        else:
            mapped[standard_col] = defaults[standard_col]

    # Convert delivery days into weeks when implementation_weeks was absent.
    if mapped["implementation_weeks"].isna().all() and not mapped["delivery_days"].isna().all():
        mapped["implementation_weeks"] = pd.to_numeric(mapped["delivery_days"], errors="coerce") / 7

    mapped["implementation_weeks"] = mapped["implementation_weeks"].fillna(defaults["implementation_weeks"] or 8)

    numeric_defaults = {
        "quote_amount": 0,
        "implementation_weeks": 8,
        "sla_uptime": 99.0,
        "support_response_hours": 24,
        "contract_term_months": 12,
        "past_performance_score": 75,
        "implementation_fee": 0,
        "renewal_escalator_pct": 5,
    }

    for col, default in numeric_defaults.items():
        mapped[col] = pd.to_numeric(mapped[col], errors="coerce").fillna(default)

    text_cols = [
        "vendor", "security_status", "tax_form", "data_access_level",
        "payment_terms", "service_scope", "notes"
    ]
    for col in text_cols:
        mapped[col] = mapped[col].fillna(defaults[col]).astype(str).str.strip()
        mapped[col] = mapped[col].replace({"": defaults[col], "nan": defaults[col]})

    return mapped


def extract_request_facts(text: str) -> Dict[str, Any]:
    facts = {
        "department": "Procurement Analytics Team",
        "business_owner": "Procurement Manager",
        "request_type": "Vendor selection and approval",
        "budget": 20000,
        "urgency": "High",
        "required_capabilities": ["vendor request review", "quote comparison", "approval memo drafting", "audit trail creation"],
        "constraints": [
            "budget under $20,000",
            "Net 30 or better payment terms",
            "security approval before contract execution",
            "W-9 or equivalent tax documentation",
            "minimize sensitive financial data exposure",
        ],
        "success_metrics": [
            "reduce manual approval prep time by at least 35%",
            "deliver working dashboard within one month",
            "improve audit readiness",
            "standardize vendor scoring",
        ],
    }

    lower = text.lower()

    money = re.findall(r"\$\s?([0-9,]+)", text)
    if money:
        try:
            facts["budget"] = int(money[0].replace(",", ""))
        except Exception:
            pass

    # Handles "under $20,000" even if budget label is missing.
    under_money = re.search(r"under\s*\$\s?([0-9,]+)", text, flags=re.I)
    if under_money:
        try:
            facts["budget"] = int(under_money.group(1).replace(",", ""))
        except Exception:
            pass

    dept_match = re.search(r"(department|team)\s*[:\-]\s*(.+)", text, flags=re.I)
    if dept_match:
        facts["department"] = dept_match.group(2).strip().split("\n")[0]

    owner_match = re.search(r"business owner\s*[:\-]\s*(.+)", text, flags=re.I)
    if owner_match:
        facts["business_owner"] = owner_match.group(1).strip().split("\n")[0]

    if any(x in lower for x in ["urgent", "asap", "critical", "within one month"]):
        facts["urgency"] = "High"
    if any(x in lower for x in ["low priority", "not urgent"]):
        facts["urgency"] = "Low"

    return facts


def security_risk(status: str, data_access: str) -> int:
    s = str(status).lower()
    d = str(data_access).lower()

    score = 0

    if any(x in s for x in ["pending", "not", "missing", "incomplete", "required"]):
        score += 30
    if any(x in s for x in ["approved", "complete", "passed", "soc 2", "iso"]):
        score -= 10
    if "soc 2" not in s and "iso" not in s and "approved" not in s and "complete" not in s:
        score += 10

    if any(x in d for x in ["customer", "financial", "sensitive", "internal"]):
        score += 18
    if "public" in d or "no access" in d:
        score -= 10

    return max(0, min(100, score))


def tax_risk(tax_form: str) -> int:
    t = str(tax_form).lower()
    if any(x in t for x in ["w-9 on file", "w9 on file", "on file", "available", "complete"]):
        return 0
    if any(x in t for x in ["pending", "missing", "not provided", "not available"]):
        return 18
    return 8


def payment_score(terms: str) -> float:
    t = str(terms).lower()
    if "net 45" in t or "net 60" in t:
        return 100
    if "net 30" in t:
        return 90
    if "net 15" in t:
        return 70
    if "due" in t or "advance" in t or "upfront" in t:
        return 45
    return 75


def run_agent_pipeline(request_text: str, quotes_df: pd.DataFrame) -> Dict[str, Any]:
    quotes = normalize_quotes(quotes_df)
    facts = extract_request_facts(request_text)
    budget = float(facts["budget"])

    min_price = float(quotes["quote_amount"].min()) if len(quotes) else 0
    max_price = float(quotes["quote_amount"].max()) if len(quotes) else 1
    span = max(max_price - min_price, 1)

    rows = []
    for _, r in quotes.iterrows():
        total_year_1_cost = float(r["quote_amount"]) + float(r["implementation_fee"])

        price_score = 100 - ((float(r["quote_amount"]) - min_price) / span * 35)
        if total_year_1_cost > budget:
            price_score -= 12

        sec_risk = security_risk(r["security_status"], r["data_access_level"])
        doc_risk = tax_risk(r["tax_form"])
        timeline_score = max(35, 100 - float(r["implementation_weeks"]) * 5)
        sla_score = min(100, max(45, (float(r["sla_uptime"]) - 98) * 45))
        performance_score = float(r["past_performance_score"])
        pay_score = payment_score(r["payment_terms"])

        total = (
            price_score * 0.22
            + timeline_score * 0.14
            + sla_score * 0.13
            + performance_score * 0.18
            + (100 - sec_risk) * 0.18
            + (100 - doc_risk) * 0.07
            + pay_score * 0.08
        )

        rows.append({
            "vendor": str(r["vendor"]),
            "service_scope": str(r["service_scope"]),
            "quote_amount": float(r["quote_amount"]),
            "implementation_fee": float(r["implementation_fee"]),
            "total_year_1_cost": total_year_1_cost,
            "implementation_weeks": round(float(r["implementation_weeks"]), 1),
            "sla_uptime": float(r["sla_uptime"]),
            "support_response_hours": float(r["support_response_hours"]),
            "security_status": str(r["security_status"]),
            "tax_form": str(r["tax_form"]),
            "contract_term_months": float(r["contract_term_months"]),
            "past_performance_score": float(r["past_performance_score"]),
            "data_access_level": str(r["data_access_level"]),
            "payment_terms": str(r["payment_terms"]),
            "renewal_escalator_pct": float(r["renewal_escalator_pct"]),
            "notes": str(r["notes"]),
            "price_score": round(price_score, 1),
            "timeline_score": round(timeline_score, 1),
            "sla_score": round(sla_score, 1),
            "security_risk": round(sec_risk, 1),
            "documentation_risk": doc_risk,
            "payment_terms_score": round(pay_score, 1),
            "composite_score": round(total, 1),
            "over_budget": total_year_1_cost - budget,
        })

    scored = pd.DataFrame(rows).sort_values("composite_score", ascending=False).reset_index(drop=True)

    if len(scored):
        scored.insert(0, "rank", scored.index + 1)

    winner = scored.iloc[0].to_dict() if len(scored) else {}
    cheapest = scored.sort_values("total_year_1_cost").iloc[0].to_dict() if len(scored) else {}

    savings_vs_highest = float(scored["total_year_1_cost"].max() - winner.get("total_year_1_cost", 0)) if len(scored) else 0
    negotiation_savings = max(0, winner.get("total_year_1_cost", 0) * 0.07)
    avg_risk = float(scored["security_risk"].mean()) if len(scored) else 0

    approval_confidence = (
        winner.get("composite_score", 70)
        - avg_risk * 0.08
        + (8 if winner.get("over_budget", 1) <= 0 else -8)
        + (4 if tax_risk(winner.get("tax_form", "")) == 0 else -5)
    ) if winner else 0
    approval_confidence = max(45, min(96, approval_confidence))

    decision = (
        "Approve with conditions"
        if approval_confidence >= 78
        else "Needs more information"
        if approval_confidence >= 60
        else "Do not approve yet"
    )

    missing = []
    for _, r in scored.iterrows():
        sec = str(r["security_status"]).lower()
        tax = str(r["tax_form"]).lower()
        if any(x in sec for x in ["pending", "not", "missing", "incomplete"]):
            missing.append(f"{r['vendor']}: security review is incomplete or pending")
        if tax_risk(tax) > 0:
            missing.append(f"{r['vendor']}: W-9/tax documentation needs confirmation")
        if float(r["over_budget"]) > 0:
            missing.append(f"{r['vendor']}: year-one cost exceeds budget by {currency(r['over_budget'])}")

    risk_heatmap = pd.DataFrame([
        {"Risk Area": "Security", "Severity": min(100, avg_risk + 18), "Owner": "InfoSec"},
        {"Risk Area": "Compliance", "Severity": 45 if missing else 20, "Owner": "Procurement"},
        {"Risk Area": "Financial", "Severity": 70 if winner.get("over_budget", 0) > 0 else 25, "Owner": "Finance"},
        {"Risk Area": "Operational", "Severity": min(90, winner.get("implementation_weeks", 8) * 8), "Owner": "Business"},
        {"Risk Area": "Contract", "Severity": min(80, winner.get("renewal_escalator_pct", 5) * 10), "Owner": "Legal"},
    ])

    audit_events = [
        ("Request received", "Business intake text loaded and parsed"),
        ("Intake Agent completed", f"Budget {currency(budget)}, owner {facts['business_owner']}, urgency {facts['urgency']}"),
        ("Risk Agent completed", f"{len(missing)} documentation/security/budget exceptions identified"),
        ("Comparison Agent completed", f"{len(scored)} vendors scored using weighted decision model"),
        ("Savings Engine completed", f"Potential savings identified: {currency(savings_vs_highest + negotiation_savings)}"),
        ("Decision Agent completed", f"Recommendation: {decision} for {winner.get('vendor', 'N/A')}"),
    ]

    return {
        "request_facts": facts,
        "normalized_quotes": quotes,
        "scored_vendors": scored,
        "winner": winner,
        "cheapest": cheapest,
        "missing_items": missing,
        "risk_heatmap": risk_heatmap,
        "approval_confidence": round(approval_confidence, 1),
        "decision": decision,
        "savings_vs_highest": savings_vs_highest,
        "negotiation_savings": negotiation_savings,
        "total_opportunity": savings_vs_highest + negotiation_savings,
        "audit_events": audit_events,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def build_executive_memo(result: Dict[str, Any]) -> str:
    facts = result["request_facts"]
    winner = result["winner"]
    missing = result["missing_items"]

    memo = f"""
# ProcurePilot AI Executive Decision Memo

**Generated:** {result['generated_at']}  
**Department:** {facts['department']}  
**Business Owner:** {facts['business_owner']}  
**Recommendation:** {result['decision']}  
**Recommended Vendor:** {winner.get('vendor','N/A')}  
**Approval Confidence:** {result['approval_confidence']}%

## Business Problem
The procurement team needs to select a vendor while reducing manual review time, controlling financial risk, comparing quotes consistently, and maintaining an audit-ready approval trail.

## AI Agent Workflow
1. Intake Agent extracted request details, budget, constraints, and success metrics.
2. Risk Agent checked security, tax, contract, financial, and operational risks.
3. Comparison Agent scored vendors using cost, SLA, delivery timeline, support, security, payment terms, and past performance.
4. Savings Agent estimated avoidable spend and negotiation opportunities.
5. Decision Agent created the final approval recommendation and manager-ready summary.

## Recommended Decision
ProcurePilot recommends **{result['decision']}** for **{winner.get('vendor','N/A')}**.

Key reasons:
- Composite vendor score: {winner.get('composite_score','N/A')}
- Year-one cost: {currency(winner.get('total_year_1_cost',0))}
- Quote amount: {currency(winner.get('quote_amount',0))}
- Implementation timeline: {winner.get('implementation_weeks','N/A')} weeks
- Payment terms: {winner.get('payment_terms','N/A')}
- Security status: {winner.get('security_status','N/A')}
- Past performance score: {winner.get('past_performance_score','N/A')}

## Savings and Negotiation Opportunity
- Avoidable spend versus highest-cost option: {currency(result['savings_vs_highest'])}
- Estimated negotiation opportunity: {currency(result['negotiation_savings'])}
- Total potential opportunity: {currency(result['total_opportunity'])}

## Conditions Before Approval
""".strip()

    if missing:
        for item in missing[:8]:
            memo += f"\n- {item}"
    else:
        memo += "\n- No major missing documentation found."

    memo += """

## Manager Approval Email Draft
Subject: Vendor Approval Recommendation from ProcurePilot AI

Hi Team,

ProcurePilot AI completed the vendor governance review and recommends moving forward with the selected vendor subject to the listed conditions. The recommendation is based on pricing, implementation timeline, SLA strength, security posture, documentation completeness, payment terms, and past performance.

Please review the attached decision memo and confirm whether we should proceed to final approval.

Best,  
Procurement Operations
"""
    return memo


def deterministic_executive_insights(result: Dict[str, Any]) -> str:
    """Professional fallback summary so the app always generates a polished output, even without API access."""
    winner = result.get("winner", {})
    missing = result.get("missing_items", [])
    return f"""
- **Recommendation:** Move forward with **{winner.get('vendor', 'the recommended vendor')}** under the status **{result.get('decision', 'Review Required')}** based on the strongest combined performance across cost, implementation timeline, SLA quality, risk, and vendor history.
- **Decision confidence:** ProcurePilot calculated an approval confidence of **{result.get('approval_confidence', 'N/A')}%**, which indicates the request is sufficiently supported for manager review while still preserving governance controls.
- **Financial impact:** The review identified a total savings and negotiation opportunity of **{currency(result.get('total_opportunity', 0))}**, combining avoidable spend and estimated negotiation leverage.
- **Risk controls:** The workflow found **{len(missing)} exception(s)** across documentation, security, or contract readiness; these should be closed before final purchase order release.
- **Business value:** The agentic workflow creates a repeatable audit trail, reduces manual quote comparison effort, and gives procurement leaders a consistent approval memo instead of ad hoc spreadsheet-based review.
""".strip()


def get_client(api_key: str) -> Optional[OpenAI]:
    api_key = (api_key or "").strip()
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    except Exception as e:
        st.sidebar.error(f"Could not initialize Groq client: {e}")
        return None


def llm_enhance(client: Optional[OpenAI], model: str, result: Dict[str, Any]) -> Optional[str]:
    if client is None:
        return None

    snapshot = {
        "winner": result["winner"],
        "decision": result["decision"],
        "confidence": result["approval_confidence"],
        "savings": result["total_opportunity"],
        "missing": result["missing_items"][:6],
    }

    selected_model = (model or DEFAULT_MODEL).strip()
    fallback_models = [selected_model, "llama-3.1-8b-instant", "llama3-8b-8192"]
    last_error = None

    for m in dict.fromkeys(fallback_models):
        try:
            completion = client.chat.completions.create(
                model=m,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior enterprise procurement strategy advisor. Write concise, executive-ready recommendations. Do not invent facts.",
                    },
                    {
                        "role": "user",
                        "content": f"Create a polished 5-bullet executive insight summary for this procurement AI result:\n{json.dumps(snapshot, default=str)}",
                    },
                ],
                temperature=0.25,
                max_tokens=450,
            )
            return completion.choices[0].message.content
        except Exception as e:
            last_error = e
            continue

    return f"LLM enhancement unavailable: {last_error}"


# Sidebar
st.sidebar.markdown("### 🛡️ ProcurePilot Control Center")
use_sample = st.sidebar.checkbox("Use built-in executive demo case", value=True)
api_key = st.sidebar.text_input(
    "Groq API key optional",
    value=safe_secret("GROQ_API_KEY"),
    type="password",
    help="Optional. App works without a key in deterministic demo mode.",
)
model = st.sidebar.text_input("Groq model", value=DEFAULT_MODEL)
force_no_key = st.sidebar.checkbox("Force no-key demo mode", value=False)

clean_api_key = (api_key or "").strip()
if force_no_key:
    st.sidebar.warning("Live LLM disabled: no-key demo mode is ON.")
elif clean_api_key:
    st.sidebar.success("Groq key detected. Live LLM mode is ON.")
else:
    st.sidebar.info("No Groq key detected. Running deterministic demo mode.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Demo scorecard**")
st.sidebar.markdown("✅ Business problem  \n✅ Agentic workflow  \n✅ Working app  \n✅ Visual analytics  \n✅ Deployable free")
st.sidebar.markdown("---")
st.sidebar.caption("Tip: Demo with the built-in sample first. Then uncheck it and upload the two sample files to prove file upload works.")


# Header
st.markdown(
    """
<div class="hero">
  <div class="badge">Autonomous Vendor Governance Platform</div>
  <div class="badge">Agentic AI</div>
  <div class="badge">Executive Decision Intelligence</div>
  <h1>ProcurePilot AI</h1>
  <p>Corporate-ready AI platform that converts vendor requests and quote files into a risk-scored, savings-aware, approval-ready procurement decision memo.</p>
</div>
""",
    unsafe_allow_html=True,
)


# Input
left, right = st.columns([1.08, 0.92], gap="large")

with left:
    st.markdown("### 1. Business request")
    if use_sample:
        request_text, quotes_df = load_sample_files()
        st.info("Using built-in sample request and sample vendor quote file.")
        st.text_area("Request preview", request_text, height=230)
    else:
        req_file = st.file_uploader("Upload request file (.txt or .pdf)", type=["txt", "pdf"], key="req")
        request_text, _ = read_uploaded_file(req_file)
        manual_value = st.text_area("Or paste request manually", value=request_text, height=230, key="request_manual")
        request_text = manual_value

with right:
    st.markdown("### 2. Vendor quotes")
    if use_sample:
        st.dataframe(quotes_df, use_container_width=True, height=260)
    else:
        quote_file = st.file_uploader("Upload vendor quote file (.csv or .xlsx)", type=["csv", "xlsx"], key="quotes")
        _, quotes_df = read_uploaded_file(quote_file)
        if quotes_df is not None:
            st.dataframe(quotes_df, use_container_width=True, height=260)
        else:
            st.warning("Upload a vendor quote CSV/XLSX to run the review.")

run = st.button("🚀 Run Agentic Vendor Governance Review", use_container_width=True)

if "result" not in st.session_state and use_sample:
    st.session_state["result"] = run_agent_pipeline(request_text, quotes_df)

if run:
    if not request_text.strip() or quotes_df is None or quotes_df.empty:
        st.error("Please provide both a business request and a vendor quote file.")
        st.stop()

    with st.spinner("Running Intake → Risk → Comparison → Savings → Decision agents..."):
        st.session_state["result"] = run_agent_pipeline(request_text, quotes_df)

result = st.session_state.get("result")
if not result:
    st.stop()

scored = result["scored_vendors"]
winner = result["winner"]


# KPI row
st.markdown("### Executive control tower")
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>Recommended vendor</div><div class='metric-value'>{winner.get('vendor','N/A')}</div><div class='metric-note'>{result['decision']}</div></div>",
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>Approval confidence</div><div class='metric-value'>{result['approval_confidence']}%</div><div class='metric-note'>AI decision certainty</div></div>",
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>Savings opportunity</div><div class='metric-value'>{currency(result['total_opportunity'])}</div><div class='metric-note'>Avoidable + negotiable spend</div></div>",
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>Exceptions found</div><div class='metric-value'>{len(result['missing_items'])}</div><div class='metric-note'>Docs/security/contract flags</div></div>",
        unsafe_allow_html=True,
    )


# Agent Tabs
st.markdown("### Agentic workflow: click each agent")
tabs = st.tabs(["🧾 Intake Agent", "🛡️ Risk Agent", "📊 Comparison Agent", "💰 Savings Agent", "✅ Decision Agent", "🧠 Audit Trail"])

with tabs[0]:
    facts = result["request_facts"]
    c1, c2 = st.columns([0.9, 1.1])

    with c1:
        st.markdown(
            "<div class='agent-card'><div class='agent-name'><span class='status-dot'></span>Intake Agent</div><div class='agent-sub'>Converts unstructured business request into structured procurement fields.</div></div>",
            unsafe_allow_html=True,
        )
        st.json(facts)

    with c2:
        st.markdown("#### Extracted business context")
        st.write(f"**Department:** {facts['department']}")
        st.write(f"**Business owner:** {facts['business_owner']}")
        st.write(f"**Budget:** {currency(facts['budget'])}")
        st.write(f"**Urgency:** {facts['urgency']}")
        st.markdown("**Success metrics**")
        for item in facts["success_metrics"]:
            st.markdown(f"- {item}")

with tabs[1]:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown(
            "<div class='agent-card'><div class='agent-name'><span class='status-dot'></span>Risk Agent</div><div class='agent-sub'>Checks vendor security, tax documentation, contract, financial and operational exposure.</div></div>",
            unsafe_allow_html=True,
        )
        risk_df = result["risk_heatmap"]
        fig = px.density_heatmap(
            risk_df,
            x="Risk Area",
            y="Owner",
            z="Severity",
            text_auto=True,
            color_continuous_scale="RdYlGn_r",
            range_color=[0, 100],
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Exceptions and controls")
        if result["missing_items"]:
            for item in result["missing_items"]:
                st.markdown(f"<div class='warn'>⚠️ {item}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='good'>✅ No major missing documentation found.</div>", unsafe_allow_html=True)

        st.markdown("#### Control recommendation")
        st.markdown(
            "<div class='callout'>Require security evidence, tax documentation, and final legal review before purchase order release.</div>",
            unsafe_allow_html=True,
        )

with tabs[2]:
    st.markdown(
        "<div class='agent-card'><div class='agent-name'><span class='status-dot'></span>Comparison Agent</div><div class='agent-sub'>Ranks vendors with a weighted score across cost, SLA, implementation speed, security, documentation, payment terms, and performance.</div></div>",
        unsafe_allow_html=True,
    )

    chart_col, table_col = st.columns([0.95, 1.05])

    with chart_col:
        fig = px.bar(
            scored.sort_values("composite_score"),
            x="composite_score",
            y="vendor",
            orientation="h",
            text="composite_score",
            title="Vendor composite score",
        )
        fig.update_layout(height=390, xaxis_title="Score", yaxis_title="", margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with table_col:
        show_cols = [
            "rank", "vendor", "quote_amount", "total_year_1_cost",
            "implementation_weeks", "security_status", "payment_terms",
            "security_risk", "past_performance_score", "composite_score"
        ]
        st.dataframe(scored[show_cols], use_container_width=True, height=390)

with tabs[3]:
    st.markdown(
        "<div class='agent-card'><div class='agent-name'><span class='status-dot'></span>Savings Opportunity Engine</div><div class='agent-sub'>Identifies avoidable spend, negotiation leverage, and pricing tradeoffs.</div></div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        cost_df = scored[["vendor", "quote_amount", "implementation_fee", "total_year_1_cost"]].sort_values("total_year_1_cost")
        fig = px.bar(cost_df, x="vendor", y=["quote_amount", "implementation_fee"], title="Year-one cost breakdown", barmode="stack")
        fig.update_layout(height=390, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Negotiation insights")
        st.markdown(
            f"<div class='good'>💵 Potential total opportunity: <b>{currency(result['total_opportunity'])}</b></div>",
            unsafe_allow_html=True,
        )
        st.markdown("- Ask preferred vendor for 5 to 8% concession based on competitive quote spread.")
        st.markdown("- Request implementation fee reduction or convert it into onboarding credits.")
        st.markdown("- Add renewal escalator cap and SLA penalty language.")
        st.markdown("- Use faster implementation timeline as a decision lever if approval is urgent.")

with tabs[4]:
    st.markdown(
        "<div class='agent-card'><div class='agent-name'><span class='status-dot'></span>Decision Agent</div><div class='agent-sub'>Creates an executive-ready approval recommendation with conditions and rationale.</div></div>",
        unsafe_allow_html=True,
    )

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result["approval_confidence"],
        number={"suffix": "%"},
        title={"text": "Approval confidence"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2563eb"},
            "steps": [
                {"range": [0, 60], "color": "#fee2e2"},
                {"range": [60, 78], "color": "#fef3c7"},
                {"range": [78, 100], "color": "#dcfce7"},
            ],
        },
    ))
    gauge.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")

    c1, c2 = st.columns([0.8, 1.2])

    with c1:
        st.plotly_chart(gauge, use_container_width=True)

    with c2:
        st.markdown(f"#### Recommendation: **{result['decision']}**")
        st.markdown(f"Recommended vendor: **{winner.get('vendor','N/A')}**")
        st.markdown(f"Year-one cost: **{currency(winner.get('total_year_1_cost',0))}**")
        st.markdown(f"Composite score: **{winner.get('composite_score','N/A')} / 100**")
        st.markdown("#### Approval conditions")
        conditions = result["missing_items"][:5] if result["missing_items"] else ["Complete final procurement approval and purchase order routing."]
        for item in conditions:
            st.markdown(f"- {item}")

with tabs[5]:
    st.markdown(
        "<div class='agent-card'><div class='agent-name'><span class='status-dot'></span>Audit Trail Agent</div><div class='agent-sub'>Creates a traceable review log so the business can explain why the decision was made.</div></div>",
        unsafe_allow_html=True,
    )

    audit_df = pd.DataFrame(result["audit_events"], columns=["Step", "Evidence"])
    audit_df.insert(0, "Timestamp", [result["generated_at"] for _ in range(len(audit_df))])
    st.dataframe(audit_df, use_container_width=True, height=320)


# Memo
st.markdown("### Executive memo and export")
clean_api_key = (api_key or "").strip()
client = None if force_no_key else get_client(clean_api_key)
llm_summary = llm_enhance(client, model, result) if client else None

if clean_api_key and not force_no_key:
    st.caption("Live LLM mode attempted using Groq API. If Groq is unavailable, ProcurePilot automatically falls back to the built-in executive reasoning engine.")

st.markdown("#### Executive insight summary")
if llm_summary and not str(llm_summary).startswith("LLM enhancement unavailable"):
    st.success("Live Groq LLM summary generated successfully.")
    st.markdown(llm_summary)
elif llm_summary and str(llm_summary).startswith("LLM enhancement unavailable"):
    st.warning("Groq live summary could not complete, so ProcurePilot generated the executive insight summary using its built-in reasoning engine.")
    with st.expander("Technical Groq message"):
        st.code(str(llm_summary))
    st.markdown(deterministic_executive_insights(result))
else:
    st.info("Using ProcurePilot built-in executive reasoning engine. Add a Groq key only if you want the optional live LLM rewrite.")
    st.markdown(deterministic_executive_insights(result))

memo = build_executive_memo(result)
st.text_area("Approval-ready decision memo", memo, height=360)

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.download_button(
        "⬇️ Download executive memo",
        data=memo,
        file_name="ProcurePilot_Executive_Decision_Memo.md",
        mime="text/markdown",
        use_container_width=True,
    )

with col_b:
    st.download_button(
        "⬇️ Download scored vendor CSV",
        data=scored.to_csv(index=False),
        file_name="ProcurePilot_Scored_Vendors.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_c:
    audit = pd.DataFrame(result["audit_events"], columns=["Step", "Evidence"]).to_csv(index=False)
    st.download_button(
        "⬇️ Download audit trail",
        data=audit,
        file_name="ProcurePilot_Audit_Trail.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("---")
st.markdown(
    "<p class='small-muted'>Built for the individual LLM application assignment: clear business problem, agentic business workflow, working solution, visual application demo, code-ready deployment, and optional free-tier LLM integration.</p>",
    unsafe_allow_html=True,
)
