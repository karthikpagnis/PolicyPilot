"""Identity document extraction agent."""

from utils.pdf_utils import get_specific_page_images
from utils.ocr_client import extract_text_from_base64_image
from utils.ollama_client import chat
from utils.model_output import parse_json_response


def _extract_id_with_llm(text: str) -> dict:
    """Use phi3 to extract identity fields from OCR text."""
    prompt = f"""Extract identity fields from OCR text. Respond JSON only.

Text:
{text[:2500]}

Fields (null if missing):
- patient_name, date_of_birth (DD/MM/YYYY), id_number, policy_number
- gender (Male/Female), blood_group (A+/B+/O+/AB+), address
- contact_number, email

JSON: {{"patient_name": "...", "date_of_birth": "...", "id_number": null, ...}}"""

    try:
        response = chat(prompt)
        parsed, error = parse_json_response(response)
        if error:
            print(f"ID LLM parse error: {error}")
            return {"error": f"JSON parse failed: {error}"}

        result = {k: v for k, v in parsed.items() if v and v != "null"}
        return result if result else {"error": "No identity fields found"}
    except Exception as e:
        print(f"ID LLM exception: {e}")
        return {"error": str(e)}


def id_agent(state: dict) -> dict:
    """
    Extracts identity information from pages classified as identity_document.
    """
    routing = state["routing"]
    pdf_bytes = state["pdf_bytes"]

    id_pages = routing.get("identity_document", [])

    if not id_pages:
        print("ID Agent: No identity document pages found")
        return {"id_data": {"error": "No identity document pages found"}}

    print(f"ID Agent: Processing pages {id_pages}")

    page_images = get_specific_page_images(pdf_bytes, id_pages)

    all_text = ""
    for page_id, b64_image in page_images.items():
        text = extract_text_from_base64_image(b64_image)
        all_text += "\n" + text

    id_data = _extract_id_with_llm(all_text)

    print(f"ID Agent: Extracted fields: {list(id_data.keys())}")
    return {"id_data": id_data}
