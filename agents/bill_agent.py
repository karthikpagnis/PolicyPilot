"""Itemized bill extraction agent."""

from utils.pdf_utils import get_specific_page_images
from utils.ocr_client import extract_text_from_base64_image
from utils.ollama_client import chat
from utils.model_output import parse_json_response


def _extract_bill_with_llm(text: str) -> dict:
    """Use phi3 to extract itemized bill fields from OCR text."""
    prompt = f"""You are a medical billing data extraction assistant. Extract itemized bill information from the OCR text below.

OCR Text:
{text[:4000]}

Extract ONLY these fields:
- bill_number: Bill or invoice number (or null)
- bill_date: Date in DD/MM/YYYY format (or null)
- patient_name: Patient name (or null)
- facility_name: Hospital/clinic/pharmacy name (or null)
- items: Array of line items with description, quantity, rate, amount (or empty array [])
- total_amount: Total as number without currency symbols (or null)
- payment_method: cash, cheque, card, or bank_transfer (or null)

CRITICAL: Respond with VALID JSON only. No markdown, no explanation, no code blocks.
Use null for missing fields. Use empty array [] for no items. Use double quotes.

Example format:
{{"bill_number": "INV123", "bill_date": "01/15/2025", "patient_name": "John Doe", "facility_name": "City Hospital", "items": [{{"description": "Consultation", "quantity": 1, "rate": 50.00, "amount": 50.00}}, {{"description": "Medicine", "quantity": 2, "rate": 25.00, "amount": 50.00}}], "total_amount": 100.00, "payment_method": "cash"}}

Now extract from the OCR text above:"""

    try:
        response = chat(prompt)
        parsed, error = parse_json_response(response)
        if error:
            print(f"Bill LLM parse error: {error}")
            return {"error": f"JSON parse failed: {error}"}

        result = {}
        for k, v in parsed.items():
            if v is None or v == "null":
                continue
            if isinstance(v, list):
                result[k] = v
            elif v:
                result[k] = v

        return result if result else {"error": "No bill fields extracted"}
    except Exception as e:
        print(f"Bill LLM exception: {e}")
        return {"error": str(e)}


def bill_agent(state: dict) -> dict:
    """
    Extracts itemized bill data from all itemized_bill pages individually.
    Each page is processed separately, then results are merged.
    """
    routing = state["routing"]
    pdf_bytes = state["pdf_bytes"]

    bill_pages = sorted(routing.get("itemized_bill", []))

    if not bill_pages:
        print("Bill Agent: No itemized bill pages found")
        return {"bill_data": {"error": "No itemized bill pages found"}}

    print(f"Bill Agent: Processing pages {bill_pages}")

    page_images = get_specific_page_images(pdf_bytes, bill_pages)

    per_page_results = []
    for page_id, b64_image in page_images.items():
        print(f"Bill Agent: Extracting from {page_id}")
        text = extract_text_from_base64_image(b64_image)
        page_data = _extract_bill_with_llm(text)
        page_data["source_page"] = page_id
        per_page_results.append(page_data)

    bill_data = _merge_bill_results(per_page_results)

    total_items = len(bill_data.get("items", []))
    print(f"Bill Agent: Done - {total_items} line items from {len(per_page_results)} pages")
    return {"bill_data": bill_data}


def _merge_bill_results(page_results: list) -> dict:
    """Merge per-page bill extractions into a single combined result."""
    merged = {
        "bills": [],
        "items": [],
        "total_amount": 0.0,
        "page_totals": [],
    }

    common_fields = ["patient_name", "facility_name", "payment_method"]
    for field in common_fields:
        for result in page_results:
            if result.get(field):
                merged[field] = result[field]
                break

    for result in page_results:
        if result.get("error"):
            continue

        page_bill = {}
        for key in ["bill_number", "bill_date", "patient_name", "facility_name",
                    "total_amount", "payment_method", "source_page"]:
            if result.get(key):
                page_bill[key] = result[key]

        page_items = result.get("items", [])
        if page_items:
            for item in page_items:
                item["source_page"] = result.get("source_page", "unknown")
            merged["items"].extend(page_items)
            page_bill["item_count"] = len(page_items)

        page_total = result.get("total_amount")
        if page_total is not None:
            try:
                amount = float(str(page_total).replace(",", ""))
                merged["page_totals"].append({
                    "page": result.get("source_page", "unknown"),
                    "amount": amount
                })
                merged["total_amount"] += amount
            except (ValueError, TypeError):
                pass

        if page_bill:
            merged["bills"].append(page_bill)

    merged["total_amount"] = round(merged["total_amount"], 2)
    if not merged["bills"]:
        del merged["bills"]
    if not merged["items"]:
        del merged["items"]
    if not merged["page_totals"]:
        del merged["page_totals"]
    if merged["total_amount"] == 0.0:
        del merged["total_amount"]

    return merged
