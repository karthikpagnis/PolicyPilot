"""Page classification logic for claim document routing."""

from utils.pdf_utils import pdf_pages_to_images
from utils.ocr_client import extract_text_from_base64_image
from utils.ollama_client import chat
from utils.model_output import parse_json_response

DOCUMENT_TYPES = [
    "claim_forms",
    "cheque_or_bank_details",
    "identity_document",
    "itemized_bill",
    "discharge_summary",
    "prescription",
    "investigation_report",
    "cash_receipt",
    "other"
]

LABEL_ALIASES = {
    "claim_forms": "claim_forms",
    "claim form": "claim_forms",
    "claim_forms_document": "claim_forms",
    "bank": "cheque_or_bank_details",
    "cheque_or_bank_details": "cheque_or_bank_details",
    "cheque": "cheque_or_bank_details",
    "cheque details": "cheque_or_bank_details",
    "cheque number": "cheque_or_bank_details",
    "date issued": "cheque_or_bank_details",
    "payee": "cheque_or_bank_details",
    "ifsc routing number": "cheque_or_bank_details",
    "routing number": "cheque_or_bank_details",
    "swift code": "cheque_or_bank_details",
    "bank_details": "cheque_or_bank_details",
    "bank account": "cheque_or_bank_details",
    "bank account details": "cheque_or_bank_details",
    "account holder name": "cheque_or_bank_details",
    "account type": "cheque_or_bank_details",
    "bank name": "cheque_or_bank_details",
    "account status": "cheque_or_bank_details",
    "identity_document": "identity_document",
    "id_document": "identity_document",
    "identity": "identity_document",
    "government id card": "identity_document",
    "government id": "identity_document",
    "government id document": "identity_document",
    "itemized_bill": "itemized_bill",
    "itemized bill": "itemized_bill",
    "itemized hospital bill": "itemized_bill",
    "bill": "itemized_bill",
    "hospital_bill": "itemized_bill",
    "hospital bill": "itemized_bill",
    "pharmacy bill": "itemized_bill",
    "pharmacy outpatient bill": "itemized_bill",
    "pharmacy and outpatient bill": "itemized_bill",
    "medication bill": "itemized_bill",
    "outpatient bill": "itemized_bill",
    "invoice": "itemized_bill",
    "charges": "itemized_bill",
    "discharge_summary": "discharge_summary",
    "discharge summary": "discharge_summary",
    "discharge note": "discharge_summary",
    "prescription": "prescription",
    "medication": "prescription",
    "investigation_report": "investigation_report",
    "investigation report": "investigation_report",
    "lab_report": "investigation_report",
    "lab report": "investigation_report",
    "lab result": "investigation_report",
    "test result": "investigation_report",
    "blood test": "investigation_report",
    "lab test": "investigation_report",
    "pathology laboratory": "investigation_report",
    "comprehensive metabolic panel": "investigation_report",
    "lipid panel": "investigation_report",
    "thyroid function": "investigation_report",
    "lipid panel thyroid function": "investigation_report",
    "diagnostic": "investigation_report",
    "pathology": "investigation_report",
    "radiology": "investigation_report",
    "cash_receipt": "cash_receipt",
    "cash receipt": "cash_receipt",
    "receipt": "cash_receipt",
    "other": "other",
}


def _classify_with_llm(raw_text: str) -> dict:
    """Use phi3 LLM to classify OCR text into one of 9 document types."""
    text_sample = raw_text[:1500]

    prompt = f"""Classify this medical document OCR text into ONE category.

Categories:
1. claim_forms - Insurance claim forms
2. cheque_or_bank_details - Bank/cheque details
3. identity_document - Government/patient ID cards
4. itemized_bill - Bills with line items and prices
5. discharge_summary - Hospital discharge medical summary
6. prescription - Medical prescriptions
7. investigation_report - Lab/test results
8. cash_receipt - Payment receipts
9. other - Anything else

Text:
{text_sample}

Rules (apply in order):
1. Has "discharge" AND "bill"? -> itemized_bill
2. Has "discharge" WITHOUT "bill"? -> discharge_summary
3. Has line items + prices? -> itemized_bill
4. Has claim/policy keywords? -> claim_forms
5. Has bank/account keywords? -> cheque_or_bank_details
6. Has ID card keywords? -> identity_document
7. Has lab/test results? -> investigation_report
8. Has Rx/medications? -> prescription

Respond JSON only: {{"document_type": "category", "confidence": "high"}}"""

    try:
        response = chat(prompt)
        parsed, error = parse_json_response(response)
        if error or not parsed:
            print(f"Classification parse error: {error}")
            return {"document_type": "other", "confidence": "low"}

        doc_type = parsed.get("document_type", "other")
        if doc_type not in DOCUMENT_TYPES:
            print(f"Invalid doc_type '{doc_type}', defaulting to other")
            return {"document_type": "other", "confidence": "low"}

        return {
            "document_type": doc_type,
            "confidence": parsed.get("confidence", "medium")
        }
    except Exception as e:
        print(f"Classification LLM error: {e}")
        return {"document_type": "other", "confidence": "low"}


def segregator_agent(state: dict) -> dict:
    """
    Classifies all PDF pages and builds routing map for extraction agents.

    Returns updated state with page_images, page_classifications, and routing.
    """
    print("Segregator: Converting PDF pages to images")
    page_images = pdf_pages_to_images(state["pdf_bytes"])
    print(f"Segregator: Found {len(page_images)} pages")

    classifications = {}

    for page_id, b64_image in page_images.items():
        print(f"Classifying {page_id}...", end=" ", flush=True)
        raw = extract_text_from_base64_image(b64_image)
        classification = _classify_with_llm(raw)
        doc_type = classification["document_type"]
        confidence = classification.get("confidence", "low")

        if doc_type not in DOCUMENT_TYPES:
            doc_type = "other"

        classifications[page_id] = doc_type
        print(f"-> {doc_type} ({confidence})")

    # Build routing map: doc_type -> [page_numbers]
    raw_routing = {}
    for page_id, doc_type in classifications.items():
        page_num = int(page_id.split("_")[1])
        raw_routing.setdefault(doc_type, []).append(page_num)

    for doc_type in raw_routing:
        raw_routing[doc_type] = sorted(raw_routing[doc_type])

    # Single-page enforcement for identity and bank documents
    single_page_types = ["cheque_or_bank_details", "identity_document"]
    for stype in single_page_types:
        if stype in raw_routing and len(raw_routing[stype]) > 1:
            kept = raw_routing[stype][0]
            extras = raw_routing[stype][1:]
            raw_routing[stype] = [kept]
            raw_routing.setdefault("other", []).extend(extras)
            for page_num in extras:
                classifications[f"page_{page_num}"] = "other"
            print(f"{stype}: kept page {kept}, reclassified {extras} to other")

    
    if "discharge_summary" in raw_routing and len(raw_routing["discharge_summary"]) > 1:
        kept = raw_routing["discharge_summary"][0]
        extras = raw_routing["discharge_summary"][1:]
        raw_routing["discharge_summary"] = [kept]
        raw_routing.setdefault("investigation_report", []).extend(extras)
        raw_routing["investigation_report"] = sorted(raw_routing["investigation_report"])
        for page_num in extras:
            classifications[f"page_{page_num}"] = "investigation_report"
        print(f"discharge_summary: kept page {kept}, reclassified {extras} to investigation_report")

    # Merge cash_receipt into itemized_bill for routing
    if "cash_receipt" in raw_routing:
        raw_routing.setdefault("itemized_bill", []).extend(raw_routing["cash_receipt"])
        raw_routing["itemized_bill"] = sorted(set(raw_routing["itemized_bill"]))
        print(f"Merged cash_receipt pages {raw_routing['cash_receipt']} into itemized_bill routing")

    # Order routing map with priority types first
    priority_order = [
        "claim_forms",
        "cheque_or_bank_details",
        "identity_document",
        "discharge_summary",
    ]
    routing = {}
    for doc_type in priority_order:
        if doc_type in raw_routing:
            routing[doc_type] = raw_routing[doc_type]
    for doc_type, pages in raw_routing.items():
        if doc_type not in routing:
            routing[doc_type] = pages

    print("Routing map:")
    for doc_type, pages in routing.items():
        print(f"  {doc_type}: pages {pages}")

    return {
        "page_images": page_images,
        "page_classifications": classifications,
        "routing": routing,
    }
