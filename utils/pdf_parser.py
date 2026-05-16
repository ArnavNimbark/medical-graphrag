import pdfplumber
import re

KNOWN_DRUGS = [

    "Warfarin",
    "Aspirin",
    "Ibuprofen",
    "Clopidogrel",
    "Alcohol",
    "Metformin",
    "Lisinopril",
    "Heparin",
    "Paracetamol",
    "Diazepam",
    "Morphine",
    "Tramadol",
    "Fluoxetine",
    "Nitroglycerin",
    "Sildenafil"
]

def extract_text_from_pdf(pdf_file):

    full_text = ""

    with pdfplumber.open(pdf_file) as pdf:

        for page in pdf.pages:

            text = page.extract_text()

            if text:

                full_text += text + "\n"

    return full_text


def detect_drugs(text):

    detected_drugs = []

    for drug in KNOWN_DRUGS:

        pattern = rf"\b{drug}\b"

        if re.search(pattern, text, re.IGNORECASE):

            detected_drugs.append(drug)

    return list(set(detected_drugs))