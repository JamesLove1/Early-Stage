import os
import pdfplumber
import pandas as pd
from PIL import Image
import fitz
from typing import List, Dict, Any, Optional, Tuple
import io
from pathlib import Path
import numpy as np
import cv2


class PDFExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.pdf_name = Path(pdf_path).stem
        self.output_base_dir = "output"
        
    def _ensure_output_dirs(self) -> Dict[str, str]:
        dirs = {
            'text': os.path.join(self.output_base_dir, self.pdf_name, 'text'),
            'tables': os.path.join(self.output_base_dir, self.pdf_name, 'tables'),
            'images': os.path.join(self.output_base_dir, self.pdf_name, 'images'),
            'page_images': os.path.join(self.output_base_dir, self.pdf_name, 'page_images')
        }
        
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            
        return dirs
    
    def extract_text(self, save_to_file: bool = True) -> Dict[int, str]:
        print(f"Extracting text from {self.pdf_path}...")
        text_by_page = {}
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    text = page.extract_text()
                    if text:
                        text_by_page[page_num] = text
                        
            if save_to_file and text_by_page:
                dirs = self._ensure_output_dirs()
                output_path = os.path.join(dirs['text'], 'extracted_text.txt')
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    for page_num, text in text_by_page.items():
                        f.write(f"\n{'='*50}\n")
                        f.write(f"PAGE {page_num}\n")
                        f.write(f"{'='*50}\n\n")
                        f.write(text)
                        f.write('\n')
                
                print(f"Text saved to: {output_path}")
                
        except Exception as e:
            print(f"Error extracting text: {e}")
            
        return text_by_page
    
    def extract_tables(self, save_to_csv: bool = True, use_enhanced: bool = True) -> List[Dict[str, Any]]:
        print(f"Extracting tables from {self.pdf_path}...")
        all_tables = []
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                table_count = 0
                
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    
                    if use_enhanced:
                        table_settings = {
                            "vertical_strategy": "lines",
                            "horizontal_strategy": "lines",
                            "explicit_vertical_lines": [],
                            "explicit_horizontal_lines": [],
                            "snap_tolerance": 3,
                            "snap_x_tolerance": 3,
                            "snap_y_tolerance": 3,
                            "join_tolerance": 3,
                            "edge_min_length": 3,
                            "min_words_vertical": 3,
                            "min_words_horizontal": 1,
                            "intersection_tolerance": 3,
                        }
                        tables = page.extract_tables(table_settings)
                    else:
                        tables = page.extract_tables()
                    
                    if tables:
                        for j, table in enumerate(tables):
                            table_count += 1
                            
                            # Clean the table data
                            cleaned_table = []
                            for row in table:
                                cleaned_row = []
                                for cell in row:
                                    if cell is None:
                                        cleaned_row.append('')
                                    else:
                                        cleaned_row.append(str(cell).strip())
                                cleaned_table.append(cleaned_row)
                            
                            # Filter out empty rows
                            cleaned_table = [row for row in cleaned_table if any(cell for cell in row)]
                            
                            if cleaned_table:
                                try:
                                    # Try to use first row as headers
                                    if len(cleaned_table) > 1:
                                        df = pd.DataFrame(cleaned_table[1:], columns=cleaned_table[0])
                                    else:
                                        df = pd.DataFrame(cleaned_table)
                                except:
                                    df = pd.DataFrame(cleaned_table)
                                
                                # Clean up the dataframe
                                df = df.replace('', pd.NA)
                                df = df.dropna(how='all')
                                df = df.dropna(axis=1, how='all')
                                
                                if not df.empty:
                                    table_info = {
                                        'page': page_num,
                                        'table_number': table_count,
                                        'data': df,
                                        'rows': len(df),
                                        'columns': len(df.columns)
                                    }
                                    
                                    all_tables.append(table_info)
                                    
                                    if save_to_csv:
                                        dirs = self._ensure_output_dirs()
                                        output_path = os.path.join(
                                            dirs['tables'], 
                                            f'table_{table_count}_page_{page_num}.csv'
                                        )
                                        df.to_csv(output_path, index=False)
                                        print(f"Table {table_count} saved to: {output_path}")
                                        
                                        # Also save as Excel for better formatting
                                        excel_path = output_path.replace('.csv', '.xlsx')
                                        df.to_excel(excel_path, index=False)
                                
        except Exception as e:
            print(f"Error extracting tables: {e}")
            
        return all_tables
    
    def extract_images_and_graphs(self, save_images: bool = True, dpi: int = 300, save_full_pages: bool = False) -> List[Dict[str, Any]]:
        """
        Enhanced image extraction that captures both embedded images and 
        renders pages to capture graphs/charts
        """
        print(f"Extracting images and graphs from {self.pdf_path}...")
        extracted_content = []
        
        try:
            pdf_document = fitz.open(self.pdf_path)
            dirs = self._ensure_output_dirs()
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # 1. Extract embedded images (original method)
                image_list = page.get_images()
                
                if image_list:
                    print(f"Found {len(image_list)} embedded images on page {page_num + 1}")
                    
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        
                        try:
                            base_image = pdf_document.extract_image(xref)
                            image_bytes = base_image["image"]
                            
                            image = Image.open(io.BytesIO(image_bytes))
                            
                            image_info = {
                                'type': 'embedded_image',
                                'page': page_num + 1,
                                'index': img_index + 1,
                                'width': image.width,
                                'height': image.height,
                                'mode': image.mode,
                                'format': base_image.get("ext", "png")
                            }
                            
                            if save_images:
                                output_path = os.path.join(
                                    dirs['images'],
                                    f'embedded_page_{page_num + 1}_img_{img_index + 1}.png'
                                )
                                
                                if image.mode == "CMYK":
                                    image = image.convert("RGB")
                                    
                                image.save(output_path)
                                image_info['saved_path'] = output_path
                                print(f"Embedded image saved to: {output_path}")
                                
                            extracted_content.append(image_info)
                            
                        except Exception as e:
                            print(f"Error extracting embedded image {img_index + 1} from page {page_num + 1}: {e}")
                
                # 2. Render page to detect and extract graphs/charts
                print(f"Analyzing page {page_num + 1} for graphs/charts...")
                
                try:
                    # Render page at high resolution
                    mat = fitz.Matrix(dpi/72.0, dpi/72.0)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    
                    # Convert to PIL Image
                    img_data = pix.pil_tobytes(format="PNG")
                    page_image = Image.open(io.BytesIO(img_data))
                    
                    if save_images:
                        # Save full page only if requested
                        if save_full_pages:
                            full_page_path = os.path.join(
                                dirs['page_images'],
                                f'page_{page_num + 1}_full.png'
                            )
                            page_image.save(full_page_path)
                            print(f"Full page render saved to: {full_page_path}")
                        
                        # Try to detect and extract graphs/charts from the page
                        graphs = self._extract_graphs_from_page(page_image, page_num + 1, dirs['images'])
                        if graphs:
                            print(f"Found {len(graphs)} potential graphs on page {page_num + 1}")
                        extracted_content.extend(graphs)
                    
                except Exception as e:
                    print(f"Error rendering page {page_num + 1}: {e}")
                    
            pdf_document.close()
            
        except Exception as e:
            print(f"Error in enhanced image extraction: {e}")
            
        return extracted_content
    
    def _extract_graphs_from_page(self, page_image: Image, page_num: int, output_dir: str) -> List[Dict[str, Any]]:
        """
        Attempts to detect and extract graph/chart regions from a page image with padding
        """
        graphs = []
        
        try:
            # Convert PIL Image to OpenCV format
            img_array = np.array(page_image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Use multiple detection methods
            # Method 1: Edge detection for graphs with borders
            edges = cv2.Canny(gray, 30, 100)
            
            # Method 2: Threshold for graphs without clear borders
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            
            # Combine both methods
            combined = cv2.bitwise_or(edges, thresh)
            
            # Apply morphological operations to connect nearby components
            kernel = np.ones((5, 5), np.uint8)
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Parameters for graph detection
            min_area = 20000  # Increased minimum area
            max_graphs_per_page = 6  # Limit graphs per page
            graph_count = 0
            padding = 100  # Increased padding to capture all legends and axes
            
            # Track extracted regions to avoid overlaps
            extracted_regions = []
            
            for contour in contours:
                if graph_count >= max_graphs_per_page:
                    break
                    
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check aspect ratio (graphs are usually not too narrow)
                    aspect_ratio = w / h
                    if 0.2 < aspect_ratio < 5.0:  # More permissive ratio
                        # Add padding to capture legends and axes
                        x_pad = max(0, x - padding)
                        y_pad = max(0, y - padding)
                        w_pad = min(page_image.width - x_pad, w + 2 * padding)
                        h_pad = min(page_image.height - y_pad, h + 2 * padding)
                        
                        # Check if this region overlaps with already extracted regions
                        overlap = False
                        for ex_x, ex_y, ex_w, ex_h in extracted_regions:
                            if not (x_pad + w_pad < ex_x or ex_x + ex_w < x_pad or 
                                    y_pad + h_pad < ex_y or ex_y + ex_h < y_pad):
                                overlap = True
                                break
                        
                        if not overlap:
                            # Extract the region with padding
                            graph_region = page_image.crop((x_pad, y_pad, x_pad + w_pad, y_pad + h_pad))
                            
                            # Check if region has enough content (not just white space)
                            region_array = np.array(graph_region)
                            if region_array.std() > 15:  # Has some variation
                                # Additional check: ensure it's not mostly text
                                if self._is_likely_graph(region_array):
                                    graph_count += 1
                                    
                                    graph_info = {
                                        'type': 'detected_graph',
                                        'page': page_num,
                                        'index': graph_count,
                                        'x': x_pad,
                                        'y': y_pad,
                                        'width': w_pad,
                                        'height': h_pad
                                    }
                                    
                                    # Save the extracted graph
                                    graph_path = os.path.join(
                                        output_dir,
                                        f'graph_page_{page_num}_region_{graph_count}.png'
                                    )
                                    graph_region.save(graph_path)
                                    graph_info['saved_path'] = graph_path
                                    
                                    graphs.append(graph_info)
                                    extracted_regions.append((x_pad, y_pad, w_pad, h_pad))
                            
        except Exception as e:
            print(f"Error in graph detection for page {page_num}: {e}")
            
        return graphs
    
    def _is_likely_graph(self, img_array: np.ndarray) -> bool:
        """
        Simple heuristic to determine if an image region is likely a graph
        """
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Check for presence of lines (common in graphs)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        # Graphs typically have some edges but not too many (unlike dense text)
        return 0.01 < edge_ratio < 0.3
    
    def extract_all(self, extract_page_renders: bool = True) -> Dict[str, Any]:
        print(f"\nStarting enhanced extraction from: {self.pdf_path}\n")
        
        results = {
            'pdf_path': self.pdf_path,
            'pdf_name': self.pdf_name,
            'text': self.extract_text(),
            'tables': self.extract_tables(use_enhanced=True),
            'images_and_graphs': self.extract_images_and_graphs()
        }
        
        print(f"\nExtraction complete!")
        print(f"- Pages with text: {len(results['text'])}")
        print(f"- Tables found: {len(results['tables'])}")
        print(f"- Images and graphs found: {len(results['images_and_graphs'])}")
        
        embedded_count = sum(1 for item in results['images_and_graphs'] if item['type'] == 'embedded_image')
        page_renders = sum(1 for item in results['images_and_graphs'] if item['type'] == 'page_render')
        graph_count = sum(1 for item in results['images_and_graphs'] if item['type'] == 'detected_graph')
        
        print(f"  - Embedded images: {embedded_count}")
        print(f"  - Page renders: {page_renders}")
        print(f"  - Detected graphs: {graph_count}")
        
        self._generate_summary_report(results)
        
        return results
    
    def _generate_summary_report(self, results: Dict[str, Any]) -> None:
        dirs = self._ensure_output_dirs()
        report_path = os.path.join(self.output_base_dir, self.pdf_name, 'extraction_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Enhanced PDF Extraction Report\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"PDF File: {results['pdf_path']}\n")
            f.write(f"PDF Name: {results['pdf_name']}\n\n")
            
            f.write(f"Text Extraction:\n")
            f.write(f"- Total pages with text: {len(results['text'])}\n")
            f.write(f"- Pages: {', '.join(map(str, results['text'].keys()))}\n\n")
            
            f.write(f"Table Extraction:\n")
            f.write(f"- Total tables found: {len(results['tables'])}\n")
            for table in results['tables']:
                f.write(f"  - Table {table['table_number']}: Page {table['page']}, "
                       f"{table['rows']} rows × {table['columns']} columns\n")
            
            f.write(f"\nImage and Graph Extraction:\n")
            embedded_count = sum(1 for item in results['images_and_graphs'] if item['type'] == 'embedded_image')
            page_renders = sum(1 for item in results['images_and_graphs'] if item['type'] == 'page_render')
            graph_count = sum(1 for item in results['images_and_graphs'] if item['type'] == 'detected_graph')
            
            f.write(f"- Total items extracted: {len(results['images_and_graphs'])}\n")
            f.write(f"  - Embedded images: {embedded_count}\n")
            f.write(f"  - Full page renders: {page_renders}\n")
            f.write(f"  - Detected graphs/charts: {graph_count}\n")
            
            f.write(f"\nOutput Directories:\n")
            f.write(f"- Text: {dirs['text']}\n")
            f.write(f"- Tables: {dirs['tables']}\n")
            f.write(f"- Images: {dirs['images']}\n")
            f.write(f"- Page renders: {dirs['page_images']}\n")
                
        print(f"\nSummary report saved to: {report_path}")