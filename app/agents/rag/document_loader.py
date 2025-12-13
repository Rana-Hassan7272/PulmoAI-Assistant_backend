"""
Document Loader for RAG Pipeline

Supports loading documents from:
- PDF files
- Text files (.txt)
- Plain text strings
"""

import os
from typing import List, Dict, Optional
from pathlib import Path
import PyPDF2
import io


class DocumentLoader:
    """Load and parse documents from various sources."""
    
    def __init__(self):
        self.supported_extensions = {'.pdf', '.txt'}
    
    def load_pdf(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load text from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of documents with 'content' and 'metadata'
        """
        documents = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        documents.append({
                            'content': text,
                            'metadata': {
                                'source': file_path,
                                'page': page_num + 1,
                                'total_pages': num_pages,
                                'type': 'pdf'
                            }
                        })
        except Exception as e:
            raise Exception(f"Failed to load PDF {file_path}: {str(e)}")
        
        return documents
    
    def load_pdf_from_bytes(self, pdf_bytes: bytes, source_name: str = "uploaded_file") -> List[Dict[str, str]]:
        """
        Load text from PDF bytes (for file uploads).
        
        Args:
            pdf_bytes: PDF file content as bytes
            source_name: Name for the source document
            
        Returns:
            List of documents with 'content' and 'metadata'
        """
        documents = []
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            num_pages = len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    documents.append({
                        'content': text,
                        'metadata': {
                            'source': source_name,
                            'page': page_num + 1,
                            'total_pages': num_pages,
                            'type': 'pdf'
                        }
                    })
        except Exception as e:
            raise Exception(f"Failed to load PDF from bytes: {str(e)}")
        
        return documents
    
    def load_text_file(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load text from .txt file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of documents with 'content' and 'metadata'
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            return [{
                'content': content,
                'metadata': {
                    'source': file_path,
                    'type': 'text'
                }
            }]
        except Exception as e:
            raise Exception(f"Failed to load text file {file_path}: {str(e)}")
    
    def load_text_from_string(self, text: str, source_name: str = "text_input") -> List[Dict[str, str]]:
        """
        Load text from string.
        
        Args:
            text: Text content
            source_name: Name for the source document
            
        Returns:
            List of documents with 'content' and 'metadata'
        """
        return [{
            'content': text,
            'metadata': {
                'source': source_name,
                'type': 'text'
            }
        }]
    
    def load_file(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load document from file (auto-detect type).
        
        Args:
            file_path: Path to file
            
        Returns:
            List of documents with 'content' and 'metadata'
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self.load_pdf(str(file_path))
        elif extension == '.txt':
            return self.load_text_file(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {extension}. Supported: {self.supported_extensions}")
    
    def load_directory(self, directory_path: str, recursive: bool = True) -> List[Dict[str, str]]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to directory
            recursive: Whether to search recursively
            
        Returns:
            List of all documents
        """
        directory_path = Path(directory_path)
        all_documents = []
        
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    documents = self.load_file(str(file_path))
                    all_documents.extend(documents)
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")
                    continue
        
        return all_documents

