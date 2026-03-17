"""Bank/cheque details extraction agent."""

from utils.pdf_utils import get_specific_page_images
from utils.ocr_client import extract_text_from_base64_image
from utils.ollama_client import chat
from utils.model_output import parse_json_response


def _extract_bank_with_llm(text: str) -> dict:
    """Use phi3 to extract bank/cheque details from OCR text."""
    prompt = f"""You are a financial data extraction assistant. Extract bank or cheque details from the OCR text below.

OCR Text:
{text[:3000]}

Extract ONLY these fields:
- account_holder_name: Name on the account (or null)
- bank_name: Name of the bank (or null)
- account_number: Bank account number (or null)
- account_type: savings, current, etc. (or null)
- ifsc_code: IFSC or routing number (or null)
- swift_code: SWIFT/BIC code (or null)
- branch_name: Bank branch name (or null)
- cheque_number: Cheque number (or null)
- cheque_date: Date on cheque in DD/MM/YYYY format (or null)
- payee: Payee name on cheque (or null)
- cheque_amount: Amount on cheque as number (or null)

CRITICAL: Respond with VALID JSON only. No markdown, no explanation, no code blocks.
Use null for missing fields. Use double quotes for all strings.

Example format:
{{"account_holder_name": "John Doe", "bank_name": "State Bank", "account_number": "1234567890", "account_type": "savings", "ifsc_code": "SBIN0001234", "swift_code": null, "branch_name": "Main Branch", "cheque_number": null, "cheque_date": null, "payee": null, "cheque_amount": null}}

Now extract from the OCR text above:"""

    try:
        response = chat(prompt)
        parsed, error = parse_json_response(response)
        if error:
            print(f"Bank LLM parse error: {error}")
            return {"error": f"JSON parse failed: {error}"}

        result = {k: v for k, v in parsed.items() if v and v != "null"}
        return result if result else {"error": "No bank details found"}
    except Exception as e:
        print(f"Bank LLM exception: {e}")
        return {"error": str(e)}


def bank_agent(state: dict) -> dict:
    """
    Extracts bank/cheque details from cheque_or_bank_details pages.
    """
    routing = state["routing"]
    pdf_bytes = state["pdf_bytes"]

    bank_pages = routing.get("cheque_or_bank_details", [])

    if not bank_pages:
        print("Bank Agent: No bank/cheque pages found")
        return {"bank_data": {"error": "No bank/cheque pages found"}}

    print(f"Bank Agent: Processing pages {bank_pages}")

    page_images = get_specific_page_images(pdf_bytes, bank_pages)

    all_text = ""
    for page_id, b64_image in page_images.items():
        text = extract_text_from_base64_image(b64_image)
        all_text += "\n" + text

    bank_data = _extract_bank_with_llm(all_text)

    print("Bank Agent: Done")
    return {"bank_data": bank_data}
