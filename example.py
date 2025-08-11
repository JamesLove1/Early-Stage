from pdf_extractor import PDFExtractor
import os


def main():
    pdf_files = [
        "UK_Monthly_Index_July_2025.pdf",
        "uk-real-estate-market-outlook-mid-year-review-2025-report.pdf"
    ]
    
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            print(f"\n{'='*70}")
            print(f"Processing: {pdf_file}")
            print(f"{'='*70}")
            
            extractor = PDFExtractor(pdf_file)
            results = extractor.extract_all()
            
            print(f"\nAll extraction results saved in: output/{extractor.pdf_name}/")
            print(f"Check extraction_report.txt for detailed summary")
            
        else:
            print(f"PDF file not found: {pdf_file}")


if __name__ == "__main__":
    main()