#!/usr/bin/env python3
import argparse
import sys
from pdf_extractor import PDFExtractor


def main():
    parser = argparse.ArgumentParser(description='Extract text, tables, and images from PDF files')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--text-only', action='store_true', help='Extract only text')
    parser.add_argument('--tables-only', action='store_true', help='Extract only tables')
    parser.add_argument('--images-only', action='store_true', help='Extract only images')
    parser.add_argument('--no-text', action='store_true', help='Skip text extraction')
    parser.add_argument('--no-tables', action='store_true', help='Skip table extraction')
    parser.add_argument('--no-images', action='store_true', help='Skip image extraction')
    
    args = parser.parse_args()
    
    # Determine what to extract
    extract_text = True
    extract_tables = True
    extract_images = True
    
    if args.text_only:
        extract_tables = False
        extract_images = False
    elif args.tables_only:
        extract_text = False
        extract_images = False
    elif args.images_only:
        extract_text = False
        extract_tables = False
    else:
        if args.no_text:
            extract_text = False
        if args.no_tables:
            extract_tables = False
        if args.no_images:
            extract_images = False
    
    # Perform extraction
    try:
        extractor = PDFExtractor(args.pdf_path)
        
        if extract_text and extract_tables and extract_images:
            # Use extract_all for complete extraction
            results = extractor.extract_all()
        else:
            # Use selective extraction
            results = {}
            if extract_text:
                results['text'] = extractor.extract_text()
            if extract_tables:
                results['tables'] = extractor.extract_tables()
            if extract_images:
                results['images_and_graphs'] = extractor.extract_images_and_graphs()
            
            print(f"\nExtraction complete!")
            if extract_text and 'text' in results:
                print(f"- Pages with text: {len(results['text'])}")
            if extract_tables and 'tables' in results:
                print(f"- Tables found: {len(results['tables'])}")
            if extract_images and 'images_and_graphs' in results:
                print(f"- Images and graphs found: {len(results['images_and_graphs'])}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()