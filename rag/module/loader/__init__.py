from .doc_loader import CustomizedOcrDocLoader
from .pdf_loader import CustomizedOcrPdfLoader

LOADER_MAPPING = {
    "CustomizedOcrPdfLoader": [".pdf"],
    "CustomizedOcrDocLoader": [".docx", ".doc"],
}