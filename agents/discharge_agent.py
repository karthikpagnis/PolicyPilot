"""Discharge summary extraction agent."""

from utils.pdf_utils import get_specific_page_images
from utils.ocr_client import extract_text_from_base64_image
from utils.ollama_client import chat
from utils.model_output import parse_json_response


def _extract_discharge_with_llm(text: str) -> dict:
    """Use phi3 to extract discharge summary fields from OCR text."""
    prompt = f"""You are a medical data extraction assistant. Extract discharge summary information from the OCR text below.

OCR Text:
{text[:3000]}

Extract ONLY these fields:
- patient_name: Full patient name (or null)
- mrn: Medical Record Number (or null)
- date_of_birth: Date in DD/MM/YYYY format (or null)
- admission_date: Admission date in DD/MM/YYYY format (or null)
- discharge_date: Discharge date in DD/MM/YYYY format (or null)
- attending_physician: Doctor name (or null)
- admission_diagnosis: Initial diagnosis (or null)
- discharge_diagnosis: Final diagnosis (or null)
- condition_at_discharge: Patient condition status (or null)
- procedures_performed: Procedures as string (or null)
- discharge_medications: Medications as string (or null)

CRITICAL: Respond with VALID JSON only. No markdown, no explanation, no code blocks.
Use null for missing fields. Use double quotes for all strings.

Example format:
{{"patient_name": "John Doe", "mrn": "12345", "date_of_birth": null, "admission_date": "01/15/2025", "discharge_date": "01/20/2025", "attending_physician": "Dr. Smith", "admission_diagnosis": "Chest pain", "discharge_diagnosis": "Acute coronary syndrome", "condition_at_discharge": "Stable", "procedures_performed": null, "discharge_medications": "Aspirin, Lisinopril"}}

Now extract from the OCR text above:"""

    try:
        response = chat(prompt)
        parsed, error = parse_json_response(response)
        if error:
            print(f"Discharge LLM parse error: {error}")
            return {"error": f"JSON parse failed: {error}"}

        result = {k: v for k, v in parsed.items() if v and v != "null"}
        return result if result else {"error": "No fields extracted"}
    except Exception as e:
        print(f"Discharge LLM exception: {e}")
        return {"error": str(e)}


def discharge_agent(state: dict) -> dict:
    """
    Extracts discharge summary info from discharge_summary pages.
    """
    routing = state["routing"]
    pdf_bytes = state["pdf_bytes"]

    discharge_pages = routing.get("discharge_summary", [])

    if not discharge_pages:
        print("Discharge Agent: No discharge summary pages found")
        return {"discharge_data": {"error": "No discharge summary pages found"}}

    print(f"Discharge Agent: Processing pages {discharge_pages}")

    page_images = get_specific_page_images(pdf_bytes, discharge_pages)

    all_text = ""
    for page_id, b64_image in page_images.items():
        text = extract_text_from_base64_image(b64_image)
        all_text += "\n" + text

    discharge_data = _extract_discharge_with_llm(all_text)

    print("Discharge Agent: Done")
    return {"discharge_data": discharge_data}
