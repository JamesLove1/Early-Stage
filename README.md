# PDF Extraction Tool

A Python utility for extracting text, tables, and images/graphs from PDF files using pdfplumber, PyMuPDF, and OpenCV.

## Features
- **Text extraction**: Page-by-page text extraction with page numbers
- **Table extraction**: Enhanced table detection, saved as both CSV and Excel files
- **Image extraction**: Extracts embedded images from PDFs
- **Graph detection**: Automatically detects and extracts graphs/charts with padding to capture legends and axes
- **Detailed reports**: Summary report for each extraction

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface
```bash
# Extract everything from a PDF
python extract_pdf.py path/to/your.pdf

# Extract only text
python extract_pdf.py path/to/your.pdf --text-only

# Extract only tables
python extract_pdf.py path/to/your.pdf --tables-only

# Extract only images and graphs
python extract_pdf.py path/to/your.pdf --images-only

# Skip specific components
python extract_pdf.py path/to/your.pdf --no-images
python extract_pdf.py path/to/your.pdf --no-tables --no-images
```

### Python API
```python
from pdf_extractor import PDFExtractor

# Extract everything
extractor = PDFExtractor("path/to/your.pdf")
results = extractor.extract_all()

# Or extract specific components
extractor = PDFExtractor("path/to/your.pdf")
text = extractor.extract_text()
tables = extractor.extract_tables()
images_and_graphs = extractor.extract_images_and_graphs()
```

### Example Script
```bash
# Run the example script (processes PDFs in current directory)
python example.py
```

## Output Structure
```
output/
└── [pdf_name]/
    ├── text/
    │   └── extracted_text.txt
    ├── tables/
    │   ├── table_*.csv
    │   └── table_*.xlsx
    ├── images/
    │   ├── embedded_*.png
    │   └── graph_*.png
    └── extraction_report.txt
```

## Files
- `pdf_extractor.py` - Main extraction module with PDFExtractor class
- `extract_pdf.py` - Command-line interface
- `example.py` - Example usage script
- `requirements.txt` - Python dependencies

## Technical Details
- Uses **pdfplumber** for text and table extraction with enhanced settings
- Uses **PyMuPDF (fitz)** for embedded image extraction and page rendering
- Uses **OpenCV** for graph/chart detection with intelligent padding
- Tables are cleaned and saved in both CSV and Excel formats
- Graphs are extracted with 100px padding to ensure legends and axes are captured