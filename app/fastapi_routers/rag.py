"""
FastAPI router for RAG (Retrieval-Augmented Generation) document management.

Endpoints for:
- Uploading documents (PDF, text)
- Managing knowledge base
- Querying RAG system
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional, List
from pydantic import BaseModel
from ..agents.rag.rag_agent import get_rag_agent
import os
from pathlib import Path

router = APIRouter(prefix="/rag", tags=["RAG Knowledge Base"])


class DocumentUploadResponse(BaseModel):
    message: str
    documents_added: int
    total_documents: int


class TextUploadRequest(BaseModel):
    text: str
    source_name: Optional[str] = "text_input"


class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = 5
    min_similarity: Optional[float] = 0.3


class QueryResponse(BaseModel):
    context: str
    documents_retrieved: int


class StatsResponse(BaseModel):
    total_documents: int
    embedding_dimension: int
    index_type: str


@router.post("/upload/pdf", response_model=DocumentUploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    source_name: Optional[str] = Form(None)
):
    """
    Upload a PDF document to the RAG knowledge base.
    
    Args:
        file: PDF file to upload
        source_name: Optional name for the document source
        
    Returns:
        Information about the upload
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Read file content
        pdf_bytes = await file.read()
        
        # Get RAG agent
        rag_agent = get_rag_agent()
        
        # Add document
        source = source_name or file.filename
        rag_agent.add_pdf_bytes(pdf_bytes, source_name=source)
        
        # Get stats
        stats = rag_agent.get_stats()
        
        return DocumentUploadResponse(
            message=f"PDF '{source}' uploaded and indexed successfully",
            documents_added=1,  # We don't track exact count per upload
            total_documents=stats.get('total_documents', 0)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload PDF: {str(e)}"
        )


@router.post("/upload/text", response_model=DocumentUploadResponse)
async def upload_text(request: TextUploadRequest):
    """
    Upload text content to the RAG knowledge base.
    
    Args:
        request: Text content and source name
        
    Returns:
        Information about the upload
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text content cannot be empty")
        
        # Get RAG agent
        rag_agent = get_rag_agent()
        
        # Add text
        rag_agent.add_text(request.text, request.source_name)
        
        # Get stats
        stats = rag_agent.get_stats()
        
        return DocumentUploadResponse(
            message=f"Text from '{request.source_name}' uploaded and indexed successfully",
            documents_added=1,
            total_documents=stats.get('total_documents', 0)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload text: {str(e)}"
        )


@router.post("/upload/directory")
async def upload_directory(directory_path: str):
    """
    Load all documents from a directory into the RAG knowledge base.
    
    Args:
        directory_path: Path to directory containing PDF/text files
        
    Returns:
        Information about the upload
    """
    try:
        if not os.path.exists(directory_path):
            raise HTTPException(status_code=400, detail=f"Directory not found: {directory_path}")
        
        if not os.path.isdir(directory_path):
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {directory_path}")
        
        # Get RAG agent
        rag_agent = get_rag_agent()
        
        # Load knowledge base
        rag_agent.load_knowledge_base(directory_path)
        
        # Get stats
        stats = rag_agent.get_stats()
        
        return DocumentUploadResponse(
            message=f"Directory '{directory_path}' loaded and indexed successfully",
            documents_added=stats.get('total_documents', 0),
            total_documents=stats.get('total_documents', 0)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load directory: {str(e)}"
        )


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG knowledge base.
    
    Args:
        request: Query text and retrieval parameters
        
    Returns:
        Retrieved context and number of documents
    """
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Get RAG agent
        rag_agent = get_rag_agent()
        
        # Retrieve context
        context = rag_agent.retrieve_context(
            query=request.query,
            k=request.k or 5,
            min_similarity=request.min_similarity or 0.3
        )
        
        # Count documents retrieved (rough estimate from context)
        documents_retrieved = context.count("[") if context else 0
        
        return QueryResponse(
            context=context,
            documents_retrieved=documents_retrieved
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query RAG: {str(e)}"
        )


@router.get("/stats", response_model=StatsResponse)
async def get_rag_stats():
    """
    Get statistics about the RAG knowledge base.
    
    Returns:
        Statistics about documents and index
    """
    try:
        rag_agent = get_rag_agent()
        stats = rag_agent.get_stats()
        
        return StatsResponse(
            total_documents=stats.get('total_documents', 0),
            embedding_dimension=stats.get('embedding_dimension', 0),
            index_type=stats.get('index_type', 'Unknown')
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )


@router.post("/reload")
async def reload_documents():
    """
    Reload documents from the default documents folder.
    
    Useful when new documents are added to backend/app/agents/rag/documents/
    """
    try:
        rag_agent = get_rag_agent(force_reload=True)
        rag_agent.reload_documents()
        
        stats = rag_agent.get_stats()
        return {
            "message": "Documents reloaded successfully",
            "total_documents": stats.get('total_documents', 0)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload documents: {str(e)}"
        )


@router.delete("/clear")
async def clear_rag():
    """
    Clear all documents from the RAG knowledge base.
    
    WARNING: This will delete all indexed documents!
    """
    try:
        rag_agent = get_rag_agent()
        rag_agent.retriever.vector_store.clear()
        rag_agent.retriever.save(rag_agent.vector_store_path)
        
        return {"message": "RAG knowledge base cleared successfully"}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear RAG: {str(e)}"
        )

