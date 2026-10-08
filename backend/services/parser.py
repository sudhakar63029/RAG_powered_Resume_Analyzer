from pypdf import PdfReader
import os

def extract_text_from_pdf(file_path:str) -> str:
    extract_text=""

    try:
        reader = PdfReader(file_path)

        for page in reader.pages:
            text=page.extract_text()
            if text:
                extract_text+=text+"\n"

    except Exception as e:
        raise Exception(f"Failed to parse pdf file : {str(e)}")

    return extract_text