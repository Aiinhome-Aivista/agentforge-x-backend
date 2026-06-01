"""
app/services/source_detector_service.py
─────────────────────────────────────────────────────────────────────────────
Source / Target Detection Service

Implements the two detection flows requested in the spec (section 2):

  1. CSV Data-Source Identification
     ─ Given an uploaded CSV (raw bytes + filename + parsed header),
       infer the *probable source system* (SAP, Oracle, SQL Server,
       MySQL, Salesforce, Excel export, generic) using a hybrid of
       (a) deterministic schema/column heuristics and
       (b) an LLM-based metadata analysis pass.

  2. Document Upload Analysis Logic
     ─ Given an uploaded document's extracted text, the system attempts
       to identify Data Source and Data Target referenced inside the
       document.  If neither can be confidently identified, the
       deterministic fallback assumes ADF (Azure Data Factory) is
       the source system so the rest of the workflow automation remains
       uninterrupted.

This module is *standalone* — it has no hard dependencies on the rest of
the analysis pipeline so it can be called from the upload route AND from
the technical-design route to enrich the exported PDF/PPTX/DOCX payload.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.core.mistral_client import get_mistral_client

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CSV HEURISTICS  ── column signatures that strongly imply a source system
# ─────────────────────────────────────────────────────────────────────────────
# Each rule: (system_name, signature_columns, confidence_boost)
# A row is "probably this system" when ≥2 signature columns match (case-insens.)
_CSV_RULES: List[Tuple[str, List[str], int]] = [
    # ── SAP — table names like EKPO, LFA1, RBKP & 4-char uppercase tables
    ("SAP", [
        "ekko", "ekpo", "lfa1", "lfb1", "lfm1", "rbkp", "rseg",
        "mara", "mard", "bseg", "bkpf", "t001", "t024e", "tbslt",
        "matnr", "lifnr", "kunnr", "werks", "mandt", "ebeln", "ebelp",
        "waers", "kursf", "land1",
    ], 30),

    # ── Oracle / Oracle EBS / Fusion
    ("Oracle", [
        "po_header_id", "po_line_id", "ap_invoice_id", "vendor_id",
        "rcv_transaction", "po_distributions", "ap_payment",
        "oracle_object_id", "creation_date", "last_update_date",
        "created_by", "last_updated_by", "object_version_number",
    ], 25),

    # ── Microsoft SQL Server export tendencies
    ("SQL Server", [
        "rowversion", "uniqueidentifier", "dbo_", "@@identity",
        "sysobjects", "syscolumns", "nvarchar", "datetime2",
    ], 20),

    # ── MySQL export tendencies
    ("MySQL", [
        "auto_increment", "innodb", "myisam", "utf8mb4",
        "information_schema",
    ], 18),

    # ── Salesforce — sObject API names always end in __c and ID is 18/15 char
    ("Salesforce", [
        "accountid", "opportunityid", "contactid", "ownerid",
        "isdeleted", "createdbyid", "lastmodifiedbyid", "systemmodstamp",
        "external_id__c",
    ], 25),

    # ── Microsoft Excel native export — friendly column names with spaces
    ("Excel Export", [
        "sheet1", "sheet 1", "sheet_1", "unnamed:",
        "row labels", "column labels", "grand total",
    ], 15),

    # ── ServiceNow tendencies
    ("ServiceNow", [
        "sys_id", "sys_created_on", "sys_updated_on",
        "sys_created_by", "u_", "cmdb_ci", "incident_state",
    ], 20),

    # ── Workday tendencies
    ("Workday", [
        "worker_id", "wd_employee", "worker_reference", "position_id",
    ], 18),
]

# Mid-confidence cue: Salesforce custom field naming convention
_SF_CUSTOM_PATTERN = re.compile(r"__c$|__r\.", re.IGNORECASE)


def _read_csv_headers(file_bytes: bytes, sample_rows: int = 3) -> Tuple[List[str], List[List[str]]]:
    """Decode a small slice of the CSV — header + first few rows."""
    try:
        text = file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        text = file_bytes.decode("latin-1", errors="ignore")

    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except Exception:
        dialect = csv.excel  # safe default

    reader = csv.reader(io.StringIO(text), dialect)
    rows = []
    try:
        for i, row in enumerate(reader):
            rows.append(row)
            if i >= sample_rows:
                break
    except Exception as e:
        logger.warning(f"[source-detect] csv read failed: {e}")
        return [], []

    if not rows:
        return [], []
    return rows[0], rows[1:]


def _score_csv_heuristic(headers: List[str]) -> Dict[str, int]:
    """
    Score each candidate source system based on which signature columns appear.
    Returns {system_name: score}.
    """
    norm = [str(h or "").strip().lower() for h in headers]
    norm_set = set(norm)

    scores: Dict[str, int] = {}
    for system, signatures, boost in _CSV_RULES:
        hits = sum(1 for s in signatures if s.lower() in norm_set)
        # also partial-match — any header that CONTAINS the signature substring
        partial = sum(
            1 for sig in signatures
            for h in norm if sig.lower() in h and h not in norm_set
        )
        total_hits = hits * 2 + partial
        if total_hits >= 2:
            scores[system] = total_hits * boost

    # Salesforce extra cue: any custom-field __c column at all
    if any(_SF_CUSTOM_PATTERN.search(h) for h in headers):
        scores["Salesforce"] = scores.get("Salesforce", 0) + 30

    return scores


def _llm_csv_source(file_name: str, headers: List[str], sample_rows: List[List[str]]) -> Optional[Dict[str, Any]]:
    """
    Optional LLM pass — asks the model to inspect the schema and pick a source.
    Returns {source_system, confidence, reasoning} or None on failure.
    """
    if not headers:
        return None
    try:
        llm = get_mistral_client()
    except Exception as e:
        logger.warning(f"[source-detect] LLM unavailable: {e}")
        return None

    headers_text = " | ".join(str(h) for h in headers[:60])
    sample_text = "\n".join(
        " | ".join(str(c)[:40] for c in row[:60])
        for row in (sample_rows or [])[:3]
    )

    prompt = f"""You are a data-engineering classifier.  Given a CSV
schema and a few sample rows, identify the **most likely SOURCE SYSTEM**
that produced this CSV.

File name: {file_name}

Headers:
{headers_text}

Sample rows:
{sample_text or '(no sample rows)'}

Pick ONE source from this list and return ONLY a JSON object:
  - SAP
  - Oracle
  - SQL Server
  - MySQL
  - Salesforce
  - Excel Export
  - ServiceNow
  - Workday
  - Other Enterprise System

Return exactly this JSON shape — no fences, no commentary:
{{
  "source_system": "<one of the names above>",
  "confidence":    "<low|medium|high>",
  "reasoning":     "<one short sentence explaining the call>"
}}
""".strip()

    try:
        raw = llm._chat(
            "You are an enterprise data-source classifier. Return ONLY valid JSON.",
            prompt,
            temperature=0.1,
        )
        parsed = llm._parse_json(raw)
        if isinstance(parsed, dict) and parsed.get("source_system"):
            return parsed
    except Exception as e:
        logger.warning(f"[source-detect] LLM csv source call failed: {e}")
    return None


def detect_csv_source(file_bytes: bytes, file_name: str) -> Dict[str, Any]:
    """
    Public entry point — combines heuristic and LLM analysis.

    Returns:
        {
          "file_name":      str,
          "source_system":  str,            # winning system or "Unknown"
          "confidence":     "low|medium|high",
          "detection_method": "heuristic|llm|hybrid|fallback",
          "reasoning":      str,
          "schema_preview": [str, ...],     # first up to 20 headers
          "candidates":     [{system, score}, ...],
        }
    """
    headers, sample_rows = _read_csv_headers(file_bytes)
    schema_preview = [str(h or "").strip() for h in headers[:20]]

    # 1. Heuristic
    heuristic_scores = _score_csv_heuristic(headers)
    candidates = [
        {"system": s, "score": sc}
        for s, sc in sorted(heuristic_scores.items(), key=lambda kv: kv[1], reverse=True)
    ]
    top_h = candidates[0] if candidates else None

    # 2. LLM pass (only if no high-confidence heuristic hit)
    llm_result = None
    if not top_h or top_h["score"] < 40:
        llm_result = _llm_csv_source(file_name, headers, sample_rows)

    # 3. Combine
    if top_h and (not llm_result or top_h["score"] >= 60):
        return {
            "file_name":        file_name,
            "source_system":    top_h["system"],
            "confidence":       "high" if top_h["score"] >= 60 else "medium",
            "detection_method": "heuristic" if not llm_result else "hybrid",
            "reasoning": (
                f"Heuristic matched {top_h['score']} schema signals indicative of "
                f"{top_h['system']}."
            ),
            "schema_preview": schema_preview,
            "candidates":     candidates[:5],
        }

    if llm_result:
        return {
            "file_name":        file_name,
            "source_system":    llm_result.get("source_system", "Other Enterprise System"),
            "confidence":       llm_result.get("confidence", "medium"),
            "detection_method": "llm",
            "reasoning":        llm_result.get("reasoning", "LLM schema analysis."),
            "schema_preview":   schema_preview,
            "candidates":       candidates[:5],
        }

    # 4. Last-resort fallback — schema unreadable / no signals
    return {
        "file_name":        file_name,
        "source_system":    "Unknown",
        "confidence":       "low",
        "detection_method": "fallback",
        "reasoning":        "No schema signals or LLM pass succeeded.",
        "schema_preview":   schema_preview,
        "candidates":       candidates[:5],
    }


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT DATA-SOURCE / DATA-TARGET DETECTION
#   (with ADF fallback per the spec)
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_SOURCE = {
    "name": "ADF (Azure Data Factory)",
    "type": "Data Integration Service",
    "reasoning": (
        "No explicit data source could be identified from the uploaded document. "
        "Falling back to ADF (Azure Data Factory) per platform policy so that "
        "workflow automation can proceed uninterrupted."
    ),
}

_SOURCE_KEYWORDS = {
    "SAP":               ["sap ", "s/4hana", "ecc", "ariba", "successfactors"],
    "Oracle":            ["oracle ", "ebs", "fusion", "peoplesoft", "netsuite"],
    "Salesforce":        ["salesforce", "sfdc"],
    "SQL Server":        ["sql server", "mssql", "azure sql"],
    "MySQL":             ["mysql"],
    "PostgreSQL":        ["postgres", "postgresql"],
    "Snowflake":         ["snowflake"],
    "Databricks":        ["databricks"],
    "ADF":               ["azure data factory", "adf "],
    "ServiceNow":        ["servicenow"],
    "Workday":           ["workday"],
    "REST API":          ["rest api", "restful"],
    "SharePoint":        ["sharepoint"],
    "Excel / Flat File": ["excel", ".xlsx", "csv file", "flat file"],
}


def _scan_keyword_hits(text: str) -> Dict[str, int]:
    if not text:
        return {}
    low = text.lower()
    out: Dict[str, int] = {}
    for system, kws in _SOURCE_KEYWORDS.items():
        n = sum(low.count(kw) for kw in kws)
        if n:
            out[system] = n
    return out


def _llm_doc_source_target(text_excerpt: str) -> Optional[Dict[str, Any]]:
    """Ask the LLM to pull explicit source/target system names from the doc."""
    if not text_excerpt.strip():
        return None
    try:
        llm = get_mistral_client()
    except Exception as e:
        logger.warning(f"[source-detect] LLM unavailable for doc parse: {e}")
        return None

    prompt = f"""You are reading an enterprise process document.  Identify any
explicit references to data SOURCE systems (where data originates) and data
TARGET systems (where data lands).  If you cannot find an explicit reference,
return null for that side — do NOT guess.

Document excerpt (truncated):
\"\"\"
{text_excerpt[:6000]}
\"\"\"

Return EXACTLY this JSON shape — no fences, no commentary:
{{
  "data_source": {{"name": "<system or null>", "evidence": "<short quote or null>"}},
  "data_target": {{"name": "<system or null>", "evidence": "<short quote or null>"}}
}}
""".strip()

    try:
        raw = llm._chat(
            "You are a precise enterprise-data-flow analyst. Return ONLY valid JSON.",
            prompt,
            temperature=0.1,
        )
        parsed = llm._parse_json(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception as e:
        logger.warning(f"[source-detect] LLM doc source/target call failed: {e}")
    return None


def detect_document_data_lineage(doc_text: str) -> Dict[str, Any]:
    """
    Public entry point for document source/target detection.

    Returns:
      {
        "data_source":    {"name": str, "type": str|None, "evidence": str|None},
        "data_target":    {"name": str, "type": str|None, "evidence": str|None},
        "detection_method": "explicit|heuristic|fallback",
        "fallback_applied": bool,
        "fallback_reason":  str|None,
      }
    """
    text = doc_text or ""

    # 1. LLM extraction first — most accurate
    llm = _llm_doc_source_target(text)

    src_name = None
    src_ev   = None
    tgt_name = None
    tgt_ev   = None
    if llm:
        src = llm.get("data_source") or {}
        tgt = llm.get("data_target") or {}
        src_name = src.get("name") if isinstance(src, dict) else None
        src_ev   = src.get("evidence") if isinstance(src, dict) else None
        tgt_name = tgt.get("name") if isinstance(tgt, dict) else None
        tgt_ev   = tgt.get("evidence") if isinstance(tgt, dict) else None

    # 2. Heuristic backup — if LLM yielded nothing usable
    if not src_name or not tgt_name:
        hits = _scan_keyword_hits(text)
        if hits:
            ordered = sorted(hits.items(), key=lambda kv: kv[1], reverse=True)
            if not src_name and ordered:
                src_name = ordered[0][0]
                src_ev = f"Heuristic: {ordered[0][1]} keyword matches in document."
            if not tgt_name and len(ordered) > 1:
                tgt_name = ordered[1][0]
                tgt_ev = f"Heuristic: {ordered[1][1]} keyword matches in document."

    fallback_applied = False
    fallback_reason  = None

    if not src_name:
        src_name = _DEFAULT_SOURCE["name"]
        src_ev   = _DEFAULT_SOURCE["reasoning"]
        fallback_applied = True
        fallback_reason  = "No explicit data source identified in document."

    # Target falls back to "Process ERP" if still empty — keep workflow going
    if not tgt_name:
        tgt_name = "Target ERP / Downstream Process System"
        tgt_ev   = (
            "No explicit data target identified. Defaulting to downstream "
            "process system so automation routing can proceed."
        )

    detection_method = (
        "explicit" if llm and (llm.get("data_source") or {}).get("name") else
        ("fallback" if fallback_applied else "heuristic")
    )

    return {
        "data_source": {
            "name": src_name,
            "type": "Data Integration Service" if src_name.startswith("ADF") else "Source System",
            "evidence": src_ev,
        },
        "data_target": {
            "name": tgt_name,
            "type": "Downstream System",
            "evidence": tgt_ev,
        },
        "detection_method":  detection_method,
        "fallback_applied":  fallback_applied,
        "fallback_reason":   fallback_reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helper — runs both detections over a set of uploaded files.
#
# Returns the shape the technical-design route will inject into the export
# payload.  Safe to call repeatedly; CSV-only files contribute to
# csv_source_detection; doc-only files contribute to document_data_lineage.
# ─────────────────────────────────────────────────────────────────────────────
def build_source_target_report(
    files: List[Tuple[bytes, str]],
    combined_doc_text: str = "",
) -> Dict[str, Any]:
    csv_results: List[Dict[str, Any]] = []
    for file_bytes, file_name in (files or []):
        if not file_name:
            continue
        ext = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""
        if ext == "csv":
            try:
                csv_results.append(detect_csv_source(file_bytes, file_name))
            except Exception as e:
                logger.warning(f"[source-detect] csv detect failed for {file_name}: {e}")

    lineage = detect_document_data_lineage(combined_doc_text or "")

    return {
        "csv_source_detection":  csv_results,
        "document_data_lineage": lineage,
    }
