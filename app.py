"""
0BULLSHIT SaaS Backend Application
A memory-centric SaaS platform with Gemini LLM integration and investor matching capabilities.
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from uuid import UUID, uuid4
import httpx

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from supabase import create_client, Client
import google.generativeai as genai

# Document processing imports
import PyPDF2
import pdfplumber
from docx import Document

# Import investor matching algorithms
from investor_matching_algo1 import algorithm_1_enhanced_keyword_matching
from investor_matching_algo2 import algorithm_2_semantic_contextual_matching

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL environment variable is required")

SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
if not SUPABASE_SERVICE_KEY:
    raise ValueError("SUPABASE_SERVICE_ROLE_KEY environment variable is required")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

# Initialize logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="0BULLSHIT SaaS API",
    description="Backend API for memory-centric investor matching and outreach platform",
    version="0.1.0"
)

# CORS middleware configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {str(e)}")
    raise

# Initialize Gemini
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize Gemini: {str(e)}")
    raise

# Security
security = HTTPBearer()

# ===========================
# Pydantic Models
# ===========================

async def generate_and_store_embeddings(message_id: str, content: str):
    """Background task to generate and store message embeddings"""
    try:
        # Generate embedding
        try:
            embedding = await generate_embedding(content)
        except Exception as e:
            logger.error(f"Failed to generate embedding for message {message_id}: {str(e)}")
            return
        
        # Store embedding in database
        try:
            supabase.table('chat_messages').update({
                'embedding': embedding,
                'updated_at': datetime.utcnow().isoformat()
            }).eq('id', message_id).execute()
        except Exception as e:
            logger.error(f"Failed to store embedding for message {message_id}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in generate_and_store_embeddings for message {message_id}: {str(e)}")

async def generate_and_store_template_embedding(template_id: str, content: str):
    """Background task to generate and store template embedding"""
    try:
        # Generate embedding
        try:
            embedding = await generate_embedding(content)
        except Exception as e:
            logger.error(f"Failed to generate embedding for template {template_id}: {str(e)}")
            return
        
        # Store embedding in database
        try:
            supabase.table('templates').update({
                'embedding': embedding,
                'updated_at': datetime.utcnow().isoformat()
            }).eq('id', template_id).execute()
        except Exception as e:
            logger.error(f"Failed to store embedding for template {template_id}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in generate_and_store_template_embedding for template {template_id}: {str(e)}")

async def process_document(document_id: str, storage_path: str, project_variables: dict):
    """
    Background task to process a document:
    1. Extract text based on file type
    2. Send to Gemini for analysis
    3. Store results
    """
    try:
        # Update status
        supabase.table('processed_documents').update({'status': 'processing_text'}).eq('id', document_id).execute()
        
        # Get document info from database
        doc_info = supabase.table('processed_documents').select('file_type').eq('id', document_id).single().execute()
        file_type = doc_info.data['file_type'].lower()
        
        # Extract text based on file type
        try:
            if file_type == '.txt':
                with open(storage_path, 'r', encoding='utf-8') as file:
                    extracted_text = file.read()
            
            elif file_type == '.pdf':
                extracted_text = ""
                try:
                    # Try pdfplumber first (better for text-based PDFs)
                    with pdfplumber.open(storage_path) as pdf:
                        for page in pdf.pages:
                            extracted_text += page.extract_text() or ""
                except Exception as e:
                    logger.warning(f"pdfplumber failed, trying PyPDF2: {str(e)}")
                    # Fallback to PyPDF2 (better for scanned PDFs)
                    with open(storage_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        for page in reader.pages:
                            extracted_text += page.extract_text() or ""
                
                if not extracted_text.strip():
                    raise ValueError("No text could be extracted from PDF")
            
            elif file_type == '.docx':
                doc = Document(storage_path)
                extracted_text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            if not extracted_text or not extracted_text.strip():
                raise ValueError("No text could be extracted from the document")
            
        except Exception as e:
            logger.error(f"Text extraction failed for document {document_id}: {str(e)}")
            supabase.table('processed_documents').update({
                'status': 'error',
                'error_message': f"Text extraction failed: {str(e)}",
                'updated_at': datetime.utcnow().isoformat()
            }).eq('id', document_id).execute()
            return
        
        # Update with extracted text
        supabase.table('processed_documents').update({
            'extracted_text': extracted_text,
            'status': 'processing_gemini'
        }).eq('id', document_id).execute()
        
        # Prepare prompt for Gemini
        analysis_prompt = f"""
        Analyze the following document and extract key information:
        
        Project Context:
        {json.dumps(project_variables, indent=2)}
        
        Document Text:
        {extracted_text[:10000]}  # Limit text length for Gemini
        
        Please provide a structured analysis including:
        1. Summary (2-3 sentences)
        2. Key entities mentioned (people, companies, technologies)
        3. Main topics/themes
        4. Relevant insights for investor matching
        5. Sentiment/tone
        
        Format your response as JSON.
        """
        
        # Call Gemini
        response = await call_gemini(analysis_prompt)
        
        try:
            processed_data = json.loads(response)
        except:
            processed_data = {
                'summary': response[:500],
                'raw_analysis': response
            }
        
                # Generate embedding from summary
        try:
            embedding = await generate_embedding(
                processed_data.get('summary', extracted_text[:500])
            )
        except Exception as e:
            logger.error(f"Failed to generate embedding for document {document_id}: {str(e)}")
            embedding = None
        
        # Update document with results
        try:
            update_data = {
                'gemini_processed_data': processed_data,
                'status': 'processed',
                'updated_at': datetime.utcnow().isoformat()
            }
            if embedding is not None:
                update_data['embedding'] = embedding
            
            supabase.table('processed_documents').update(update_data).eq('id', document_id).execute()
        except Exception as e:
            logger.error(f"Failed to update document {document_id} with results: {str(e)}")
            raise
        
    except Exception as e:
        logger.error(f"Document processing failed for {document_id}: {str(e)}")
        supabase.table('processed_documents').update({
            'status': 'error',
            'error_message': str(e),
            'updated_at': datetime.utcnow().isoformat()
        }).eq('id', document_id).execute() 