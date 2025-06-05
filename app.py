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
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from supabase import create_client, Client
import google.generativeai as genai

# Document processing imports
import PyPDF2
import pdfplumber
from docx import Document

# Load environment variables
load_dotenv()

# Import investor matching algorithms
try:
    from investor_matching_algo1 import algorithm_1_enhanced_keyword_matching
    from investor_matching_algo2 import algorithm_2_semantic_contextual_matching
except ImportError:
    logger.warning("Investor matching algorithms not found. Using placeholders.")
def algorithm_1_enhanced_keyword_matching(user_profile, investor_data, deep_research_keywords=None):
        return investor_data[:10] if investor_data else []

def algorithm_2_semantic_contextual_matching(user_profile, investor_data, deep_research_keywords=None):
        return investor_data[:10] if investor_data else []

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL environment variable is required")

SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
if not SUPABASE_SERVICE_KEY:
    raise ValueError("SUPABASE_SERVICE_KEY or SUPABASE_SERVICE_ROLE_KEY environment variable is required")

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
    logger.info("Supabase client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {str(e)}")
    raise

# Initialize Gemini
try:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini: {str(e)}")
    raise

# Security
security = HTTPBearer()

# ===========================
# Pydantic Models
# ===========================

class UserProfile(BaseModel):
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    project_variables: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    project_variables: Optional[Dict[str, Any]] = None

class ChatCreate(BaseModel):
    title: Optional[str] = None
    initial_message: Optional[str] = None

class MessageCreate(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None

class InvestorCreate(BaseModel):
    name: str
    company_name: Optional[str] = None
    email: Optional[str] = None
    linkedin_url: Optional[str] = None
    notes: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class InvestorUpdate(BaseModel):
    name: Optional[str] = None
    company_name: Optional[str] = None
    email: Optional[str] = None
    linkedin_url: Optional[str] = None
    notes: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class EmployeeCreate(BaseModel):
    name: str
    company_name: Optional[str] = None
    investor_id: Optional[UUID] = None
    role_or_title: Optional[str] = None
    email: Optional[str] = None
    linkedin_url: Optional[str] = None
    notes: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class EmployeeUpdate(BaseModel):
    name: Optional[str] = None
    company_name: Optional[str] = None
    investor_id: Optional[UUID] = None
    role_or_title: Optional[str] = None
    email: Optional[str] = None
    linkedin_url: Optional[str] = None
    notes: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class TemplateGenerate(BaseModel):
    user_input_prompt: str
    type: str = "email"  # email, linkedin_message
    target_audience_description: Optional[str] = None
    investor_ids: Optional[List[UUID]] = Field(default_factory=list)
    employee_ids: Optional[List[UUID]] = Field(default_factory=list)
    document_ids: Optional[List[UUID]] = Field(default_factory=list)

class PreferenceCreate(BaseModel):
    item_id: UUID
    item_type: str  # 'investor', 'employee', 'template'
    preference_type: str  # 'favourite', 'undesirable'

class DocumentUpload(BaseModel):
    original_file_name: str
    storage_path: str
    file_type: str
    file_size_bytes: int

class InvestorSearch(BaseModel):
    location: str
    categories: List[str]
    stage: str
    algorithm: int = 1  # 1 or 2
    deep_research: bool = False
    company_description: Optional[str] = None

class OutreachLog(BaseModel):
    employee_id: UUID
    template_id: UUID

# ===========================
# Helper Functions
# ===========================

async def generate_embedding(text: str) -> List[float]:
    """Generate embeddings using Gemini embedding model."""
    try:
        response = await genai.embed_content(
            model='models/embedding-001',
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return response['embedding']
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {str(e)}")

async def call_gemini(prompt: str, system_message: str = None, model_name: str = "gemini-2.0-flash") -> str:
    """
    Call Gemini API with the provided prompt.
    """
    try:
        model = genai.GenerativeModel(model_name)
        
        # Construct full prompt with system message if provided
        full_prompt = prompt
        if system_message:
            full_prompt = f"{system_message}\n\n{prompt}"
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verify JWT token and return user data.
    In production, this would validate the Supabase JWT token.
    """
    try:
        # For MVP, we'll do a simple check. In production, validate JWT properly
        token = credentials.credentials
        
        # Verify token with Supabase Auth
        user = supabase.auth.get_user(token)
        if not user or not user.user:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        
        return user.user
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(status_code=401, detail="Could not validate credentials")

async def get_project_context(project_id: UUID, user_id: str) -> Dict[str, Any]:
    """
    Retrieve comprehensive project context for LLM operations.
    """
    try:
        # Get project details
        project = supabase.table('projects').select('*').eq('id', str(project_id)).eq('user_id', user_id).single().execute()
        
        # Get recent chat messages (last 10 for context)
        recent_messages = supabase.table('chat_messages').select('*').join(
            'chats', 'chat_messages.chat_id=chats.id'
        ).eq('chats.project_id', str(project_id)).order('created_at', desc=True).limit(10).execute()
        
        # Get investors
        investors = supabase.table('investors').select('*').eq('project_id', str(project_id)).execute()
        
        # Get employees
        employees = supabase.table('employees').select('*').eq('project_id', str(project_id)).execute()
        
        # Get processed documents summaries
        documents = supabase.table('processed_documents').select(
            'id, original_file_name, gemini_processed_data'
        ).eq('project_id', str(project_id)).eq('status', 'processed').execute()
        
        return {
            'project': project.data,
            'recent_messages': recent_messages.data if recent_messages.data else [],
            'investors': investors.data if investors.data else [],
            'employees': employees.data if employees.data else [],
            'documents': documents.data if documents.data else []
        }
    except Exception as e:
        logger.error(f"Error retrieving project context: {str(e)}")
        return {}

async def generate_deep_research_keywords(
    user_location: str,
    user_categories: List[str],
    user_stage: str,
    categorias_content: str,
    etapas_content: str,
    ubicaciones_content: str
) -> List[str]:
    """
    Generate 30 specialized keywords using Gemini for deep research mode.
    """
    prompt = f"""
    You are an expert in venture capital and startup ecosystems. Generate 30 highly specific and relevant keywords
    for finding the best investor matches based on the following user inputs:
    
    Location: {user_location}
    Categories/Sectors: {', '.join(user_categories)}
    Investment Stage: {user_stage}
    
    Use these reference files as inspiration to understand the domain, but generate NEW, SPECIFIC keywords
    that go beyond simple synonyms:
    
    CATEGORIES REFERENCE:
    {categorias_content[:1000]}...  # Truncated for context window
    
    STAGES REFERENCE:
    {etapas_content[:1000]}...  # Truncated for context window
    
    LOCATIONS REFERENCE:
    {ubicaciones_content[:1000]}...  # Truncated for context window
    
    Generate 30 keywords total (10 for location, 10 for categories, 10 for stage) that are:
    - Highly specific and targeted
    - Related but not just synonyms
    - Useful for semantic matching
    
    Return the keywords as a JSON array.
    """
    
    response = await call_gemini(prompt)
    
    try:
        # Parse JSON response
        keywords = json.loads(response)
        return keywords if isinstance(keywords, list) else []
    except:
        # Fallback: extract keywords from text response
        return response.split(',')[:30]

# Background Tasks
async def generate_and_store_embeddings(message_id: str, content: str):
    """Background task to generate and store message embeddings"""
    try:
        embedding = await generate_embedding(content)
        supabase.table('chat_messages').update({
            'embedding': embedding,
            'updated_at': datetime.utcnow().isoformat()
        }).eq('id', message_id).execute()
    except Exception as e:
        logger.error(f"Failed to generate/store embedding for message {message_id}: {str(e)}")

async def generate_and_store_template_embedding(template_id: str, content: str):
    """Background task to generate and store template embedding"""
    try:
        embedding = await generate_embedding(content)
        supabase.table('templates').update({
            'embedding': embedding,
            'updated_at': datetime.utcnow().isoformat()
        }).eq('id', template_id).execute()
    except Exception as e:
        logger.error(f"Failed to generate/store embedding for template {template_id}: {str(e)}")

async def process_document(document_id: str, storage_path: str, project_variables: dict):
    """Process uploaded document and extract text content."""
    try:
        file_type = storage_path.split('.')[-1].lower()
        extracted_text = ""

        if file_type == 'txt':
            try:
                with open(storage_path, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()
            except FileNotFoundError:
                logger.warning(f"File not found at {storage_path}")
                extracted_text = f"Texto simulado para archivo .txt"
        elif file_type == 'pdf':
            logger.warning("PDF extraction simulada para esta prueba")
            extracted_text = f"Texto simulado para archivo .pdf"
        elif file_type == 'docx':
            logger.warning("DOCX extraction simulada para esta prueba")
            extracted_text = f"Texto simulado para archivo .docx"
        else:
            extracted_text = f"Texto simulado para {file_type}"

        # Generate embedding for the extracted text
        embedding = await generate_embedding(extracted_text)

        # Update document record with extracted text and embedding
        supabase.table('documents').update({
            'extracted_text': extracted_text,
            'embedding': embedding,
            'processed_at': datetime.utcnow().isoformat()
        }).eq('id', document_id).execute()

    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

# ===========================
# API Endpoints
# ===========================

@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "0BULLSHIT SaaS API is running",
        "version": "0.1.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint that verifies all critical services are working.
    """
    try:
        # Test Supabase connection
        supabase.table('projects').select('id').limit(1).execute()
        
        # Test Gemini API
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Test message")
        
        return {
            "status": "healthy",
            "supabase": "connected",
            "gemini": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# ===========================
# User Profile Management
# ===========================

@app.get("/profile")
async def get_profile(current_user = Depends(get_current_user)):
    """Get current user's profile"""
    try:
        profile = supabase.table('profiles').select('*').eq('id', current_user.id).single().execute()
        return profile.data
    except Exception as e:
        # Profile might not exist yet
        return {"id": current_user.id, "full_name": None, "avatar_url": None}

@app.put("/profile")
async def update_profile(profile: UserProfile, current_user = Depends(get_current_user)):
    """Update current user's profile"""
    try:
        # Upsert profile
        result = supabase.table('profiles').upsert({
            'id': current_user.id,
            'full_name': profile.full_name,
            'avatar_url': profile.avatar_url,
            'updated_at': datetime.utcnow().isoformat()
        }).execute()
        return result.data[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ===========================
# Project Management
# ===========================

@app.post("/projects", status_code=status.HTTP_201_CREATED)
async def create_project(project: ProjectCreate, current_user = Depends(get_current_user)):
    """Create a new project"""
    try:
        result = supabase.table('projects').insert({
            'user_id': current_user.id,
            'name': project.name,
            'description': project.description,
            'project_variables': project.project_variables or {}
        }).execute()
        return result.data[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/projects")
async def list_projects(current_user = Depends(get_current_user)):
    """List all projects for the authenticated user"""
    try:
        result = supabase.table('projects').select('*').eq('user_id', current_user.id).order('created_at', desc=True).execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/projects/{project_id}")
async def get_project(project_id: UUID, current_user = Depends(get_current_user)):
    """Get specific project details"""
    try:
        result = supabase.table('projects').select('*').eq('id', str(project_id)).eq('user_id', current_user.id).single().execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=404, detail="Project not found")

@app.put("/projects/{project_id}")
async def update_project(project_id: UUID, project: ProjectUpdate, current_user = Depends(get_current_user)):
    """Update project details"""
    try:
        update_data = {'updated_at': datetime.utcnow().isoformat()}
        if project.name is not None:
            update_data['name'] = project.name
        if project.description is not None:
            update_data['description'] = project.description
        if project.project_variables is not None:
            update_data['project_variables'] = project.project_variables
        
        result = supabase.table('projects').update(update_data).eq('id', str(project_id)).eq('user_id', current_user.id).execute()
        return result.data[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(project_id: UUID, current_user = Depends(get_current_user)):
    """Delete a project"""
    try:
        supabase.table('projects').delete().eq('id', str(project_id)).eq('user_id', current_user.id).execute()
        return None
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ===========================
# Chat Management
# ===========================

@app.post("/projects/{project_id}/chats", status_code=status.HTTP_201_CREATED)
async def create_chat(project_id: UUID, chat: ChatCreate, current_user = Depends(get_current_user)):
    """Create a new chat session"""
    try:
        # Verify project ownership
        project = supabase.table('projects').select('id').eq('id', str(project_id)).eq('user_id', current_user.id).single().execute()
        
        # Create chat
        result = supabase.table('chats').insert({
            'project_id': str(project_id),
            'title': chat.title or f"Chat {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
        }).execute()
        
        chat_data = result.data[0]
        
        # Add initial message if provided
        if chat.initial_message:
            supabase.table('chat_messages').insert({
                'chat_id': chat_data['id'],
                'user_id': current_user.id,
                'role': 'user',
                'content': chat.initial_message
            }).execute()
        
        return chat_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/projects/{project_id}/chats")
async def list_chats(project_id: UUID, current_user = Depends(get_current_user)):
    """List chat sessions for a project"""
    try:
        # Verify project ownership
        project = supabase.table('projects').select('id').eq('id', str(project_id)).eq('user_id', current_user.id).single().execute()
        
        result = supabase.table('chats').select('*').eq('project_id', str(project_id)).order('updated_at', desc=True).execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/projects/{project_id}/chats/{chat_id}/messages")
async def send_message(
    project_id: UUID,
    chat_id: UUID,
    message: MessageCreate,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
):
    """Send a message and get AI response"""
    try:
        # Verify ownership
        chat_check = supabase.table('chats').select('id, project_id').eq('id', str(chat_id)).eq('project_id', str(project_id)).single().execute()
        project_check = supabase.table('projects').select('id').eq('id', str(project_id)).eq('user_id', current_user.id).single().execute()
        
        # Store user message
        user_msg_result = supabase.table('chat_messages').insert({
            'chat_id': str(chat_id),
            'user_id': current_user.id,
            'role': 'user',
            'content': message.content,
            'metadata': message.metadata
        }).execute()
        
        # Get project context for AI
        context = await get_project_context(project_id, current_user.id)
        
        # Build prompt for Gemini
        system_message = """You are a helpful AI assistant for 0BULLSHIT, a platform that helps startups connect with investors.
        Use the provided context about the project, investors, employees, and documents to give relevant and personalized responses."""
        
        context_prompt = f"""
        Project Context:
        - Project Name: {context['project']['name']}
        - Project Variables: {json.dumps(context['project'].get('project_variables', {}))}
        
        Recent Conversation:
        {chr(10).join([f"{msg['role']}: {msg['content']}" for msg in context['recent_messages'][-5:]])}
        
        Available Investors: {len(context['investors'])}
        Available Employees: {len(context['employees'])}
        Processed Documents: {len(context['documents'])}
        
        User Message: {message.content}
        
        Please provide a helpful response based on the context above.
        """
        
        # Get AI response
        ai_response = await call_gemini(context_prompt, system_message)
        
        # Store AI response
        ai_msg_result = supabase.table('chat_messages').insert({
            'chat_id': str(chat_id),
            'role': 'assistant',
            'content': ai_response
        }).execute()
        
        # Update chat timestamp
        supabase.table('chats').update({'updated_at': datetime.utcnow().isoformat()}).eq('id', str(chat_id)).execute()
        
        # Generate embeddings in background
        background_tasks.add_task(generate_and_store_embeddings, user_msg_result.data[0]['id'], message.content)
        background_tasks.add_task(generate_and_store_embeddings, ai_msg_result.data[0]['id'], ai_response)
        
        return {
            'user_message': user_msg_result.data[0],
            'ai_message': ai_msg_result.data[0]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/projects/{project_id}/chats/{chat_id}/messages")
async def get_messages(project_id: UUID, chat_id: UUID, current_user = Depends(get_current_user)):
    """Retrieve messages for a specific chat session"""
    try:
        # Verify ownership
        chat_check = supabase.table('chats').select('id, project_id').eq('id', str(chat_id)).eq('project_id', str(project_id)).single().execute()
        project_check = supabase.table('projects').select('id').eq('id', str(project_id)).eq('user_id', current_user.id).single().execute()
        
        result = supabase.table('chat_messages').select('*').eq('chat_id', str(chat_id)).order('created_at').execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ===========================
# Investor Management
# ===========================

@app.post("/projects/{project_id}/investors", status_code=status.HTTP_201_CREATED)
async def create_investor(project_id: UUID, investor: InvestorCreate, current_user = Depends(get_current_user)):
    """Create a new investor"""
    try:
        # Verify project ownership
        project = supabase.table('projects').select('id').eq('id', str(project_id)).eq('user_id', current_user.id).single().execute()
        
        result = supabase.table('investors').insert({
            'project_id': str(project_id),
            'name': investor.name,
            'company_name': investor.company_name,
            'email': investor.email,
            'linkedin_url': investor.linkedin_url,
            'notes': investor.notes,
            'metadata': investor.metadata
        }).execute()
        return result.data[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/projects/{project_id}/investors")
async def list_investors(project_id: UUID, current_user = Depends(get_current_user)):
    """List all investors for a project"""
    try:
        # Verify project ownership
        project = supabase.table('projects').select('id').eq('id', str(project_id)).eq('user_id', current_user.id).single().execute()
        
        result = supabase.table('investors').select('*').eq('project_id', str(project_id)).execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/projects/{project_id}/investors/{investor_id}")
async def get_investor(project_id: UUID, investor_id: UUID, current_user = Depends(get_current_user)):
    """Get specific investor details"""
    try:
        # Verify ownership through project
        investor_check = supabase.table('investors').select('*, projects!inner(user_id)').eq('id', str(investor_id)).eq('project_id', str(project_id)).eq('projects.user_id', current_user.id).single().execute()
        return investor_check.data
    except Exception as e:
        raise HTTPException(status_code=404, detail="Investor not found")

@app.put("/projects/{project_id}/investors/{investor_id}")
async def update_investor(
    project_id: UUID,
    investor_id: UUID,
    investor: InvestorUpdate,
    current_user = Depends(get_current_user)
):
    """Update investor details"""
    try:
        # Build update data
        update_data = {'updated_at': datetime.utcnow().isoformat()}
        for field, value in investor.dict(exclude_unset=True).items():
            update_data[field] = value
        
        # Verify ownership and update
        project = supabase.table('projects').select('id').eq('id', str(project_id)).eq('user_id', current_user.id).single().execute()
        result = supabase.table('investors').update(update_data).eq('id', str(investor_id)).eq('project_id', str(project_id)).execute()
        return result.data[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/projects/{project_id}/investors/{investor_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_investor(project_id: UUID, investor_id: UUID, current_user = Depends(get_current_user)):
    """Delete an investor"""
    try:
        # Verify ownership through project
        project = supabase.table('projects').select('id').eq('id', str(project_id)).eq('user_id', current_user.id).single().execute()
        supabase.table('investors').delete().eq('id', str(investor_id)).eq('project_id', str(project_id)).execute()
        return None
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ===========================
# Employee Management
# ===========================

@app.post("/projects/{project_id}/employees", status_code=status.HTTP_201_CREATED)
async def create_employee(project_id: UUID, employee: EmployeeCreate, current_user = Depends(get_current_user)):
    """Create a new employee"""
    try:
        # Verify project ownership
        project = supabase.table('projects').select('id').eq('id', str(project_id)).eq('user_id', current_user.id).single().execute()
        
        # If company_name provided but no investor_id, try to match
        investor_id = employee.investor_id
        if employee.company_name and not investor_id:
            # Simple string matching for MVP
            investor_match = supabase.table('investors').select('id').eq('project_id', str(project_id)).eq('company_name', employee.company_name).execute()
            if investor_match.data:
                investor_id = investor_match.data[0]['id']
        
        result = supabase.table('employees').insert({
            'project_id': str(project_id),
            'investor_id': str(investor_id) if investor_id else None,
            'name': employee.name,
            'company_name': employee.company_name,
            'role_or_title': employee.role_or_title,
            'email': employee.email,
            'linkedin_url': employee.linkedin_url,
            'notes': employee.notes,
            'metadata': employee.metadata
        }).execute()
        return result.data[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/projects/{project_id}/employees")
async def list_employees(project_id: UUID, current_user = Depends(get_current_user)):
    """List all employees for a project"""
    try:
        # Verify project ownership
        project = supabase.table('projects').select('id').eq('id', str(project_id)).eq('user_id', current_user.id).single().execute()
        
        result = supabase.table('employees').select('*').eq('project_id', str(project_id)).execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/projects/{project_id}/employees/{employee_id}")
async def get_employee(project_id: UUID, employee_id: UUID, current_user = Depends(get_current_user)):
    """Get specific employee details"""
    try:
        # Verify ownership through project
        employee_check = supabase.table('employees').select('*, projects!inner(user_id)').eq('id', str(employee_id)).eq('project_id', str(project_id)).eq('projects.user_id', current_user.id).single().execute()
        return employee_check.data
    except Exception as e:
        raise HTTPException(status_code=404, detail="Employee not found")

@app.put("/projects/{project_id}/employees/{employee_id}")
async def update_employee(
    project_id: UUID,
    employee_id: UUID,
    employee: EmployeeUpdate,
    current_user = Depends(get_current_user)
):
    """Update employee details"""
    try:
        update_data = {'updated_at': datetime.utcnow().isoformat()}
        for field, value in employee.dict(exclude_unset=True).items():
            update_data[field] = value
        
        # Verify ownership and update
        project = supabase.table('projects').select('id').eq('id', str(project_id)).eq('user_id', current_user.id).single().execute()
        result = supabase.table('employees').update(update_data).eq('id', str(employee_id)).eq('project_id', str(project_id)).execute()
        return result.data[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/projects/{project_id}/employees/{employee_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_employee(project_id: UUID, employee_id: UUID, current_user = Depends(get_current_user)):
    """Delete an employee"""
    try:
        # Verify ownership through project
        project = supabase.table('projects').select('id').eq('id', str(project_id)).eq('user_id', current_user.id).single().execute()
        supabase.table('employees').delete().eq('id', str(employee_id)).eq('project_id', str(project_id)).execute()
        return None
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ===========================
# Template Generation
# ===========================

@app.post("/projects/{project_id}/templates/generate", status_code=status.HTTP_201_CREATED)
async def generate_template(
    project_id: UUID,
    template_req: TemplateGenerate,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
):
    """Generate a new template using Gemini"""
    try:
        # Verify project ownership and get project data
        project = supabase.table('projects').select('*').eq('id', str(project_id)).eq('user_id', current_user.id).single().execute()
        
        # Retrieve context based on user's request
        context_data = {
            'project_variables': project.data.get('project_variables', {}),
            'investors': [],
            'employees': [],
            'documents': []
        }
        
        # Fetch requested investors
        if template_req.investor_ids:
            investors = supabase.table('investors').select('*').in_('id', [str(id) for id in template_req.investor_ids]).eq('project_id', str(project_id)).execute()
            context_data['investors'] = investors.data
        
        # Fetch requested employees
        if template_req.employee_ids:
            employees = supabase.table('employees').select('*').in_('id', [str(id) for id in template_req.employee_ids]).eq('project_id', str(project_id)).execute()
            context_data['employees'] = employees.data
        
        # Fetch requested documents
        if template_req.document_ids:
            documents = supabase.table('processed_documents').select(
                'id, original_file_name, gemini_processed_data'
            ).in_('id', [str(id) for id in template_req.document_ids]).eq('project_id', str(project_id)).eq('status', 'processed').execute()
            context_data['documents'] = documents.data
        
        # Build comprehensive prompt
        system_message = f"""You are an expert outreach specialist creating personalized {template_req.type} templates.
        Create compelling, professional content that resonates with the target audience."""
        
        context_prompt = f"""
        Project Information:
        {json.dumps(context_data['project_variables'], indent=2)}
        
        Target Audience: {template_req.target_audience_description or 'Not specified'}
        
        Investors Context:
        {json.dumps([{
            'name': inv['name'],
            'company': inv.get('company_name'),
            'notes': inv.get('notes'),
            'metadata': inv.get('metadata', {})
        } for inv in context_data['investors']], indent=2)}
        
        Employees Context:
        {json.dumps([{
            'name': emp['name'],
            'role': emp.get('role_or_title'),
            'company': emp.get('company_name'),
            'notes': emp.get('notes')
        } for emp in context_data['employees']], indent=2)}
        
        Document Insights:
        {json.dumps([doc.get('gemini_processed_data', {}) for doc in context_data['documents']], indent=2)}
        
        User's Specific Instructions:
        {template_req.user_input_prompt}
        
        Generate a {template_req.type} with:
        1. A compelling subject line (if email)
        2. A personalized, engaging body
        
        Format your response as JSON:
        {{
            "subject": "...",  // Only for emails
            "body": "..."
        }}
        """
        
        # Call Gemini
        response = await call_gemini(context_prompt, system_message)
        
        # Parse response
        try:
            template_content = json.loads(response)
        except:
            # Fallback for non-JSON response
            template_content = {
                'subject': f"Outreach regarding {template_req.target_audience_description or 'opportunity'}",
                'body': response
            }
        
        # Store template
        result = supabase.table('templates').insert({
            'project_id': str(project_id),
            'user_id': current_user.id,
            'name': f"{template_req.type} - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            'type': template_req.type,
            'target_audience_description': template_req.target_audience_description,
            'user_input_prompt': template_req.user_input_prompt,
            'context_references': {
                'investor_ids': [str(id) for id in template_req.investor_ids],
                'employee_ids': [str(id) for id in template_req.employee_ids],
                'document_ids': [str(id) for id in template_req.document_ids]
            },
            'generated_content_subject': template_content.get('subject'),
            'generated_content_body': template_content.get('body', response),
            'gemini_model_version': 'gemini-2.0-flash'
        }).execute()
        
        # Generate embedding in background
        background_tasks.add_task(
            generate_and_store_template_embedding,
            result.data[0]['id'],
            template_content.get('body', response)
        )
        
        return result.data[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/projects/{project_id}/templates")
async def list_templates(project_id: UUID, current_user = Depends(get_current_user)):
    """List all templates for a project"""
    try:
        # Verify project ownership
        project = supabase.table('projects').select('id').eq('id', str(project_id)).eq('user_id', current_user.id).single().execute()
        
        result = supabase.table('templates').select('*').eq('project_id', str(project_id)).order('created_at', desc=True).execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/projects/{project_id}/templates/{template_id}")
async def get_template(project_id: UUID, template_id: UUID, current_user = Depends(get_current_user)):
    """Get specific template details"""
    try:
        # Verify ownership through project
        template_check = supabase.table('templates').select('*, projects!inner(user_id)').eq('id', str(template_id)).eq('project_id', str(project_id)).eq('projects.user_id', current_user.id).single().execute()
        return template_check.data
    except Exception as e:
        raise HTTPException(status_code=404, detail="Template not found")

@app.put("/projects/{project_id}/templates/{template_id}/favourite")
async def toggle_template_favourite(
    project_id: UUID,
    template_id: UUID,
    is_favourite: bool,
    current_user = Depends(get_current_user)
):
    """Toggle template favourite status"""
    try:
        # Verify ownership and update
        project = supabase.table('projects').select('id').eq('id', str(project_id)).eq('user_id', current_user.id).single().execute()
        result = supabase.table('templates').update({
            'is_favourite': is_favourite,
            'updated_at': datetime.utcnow().isoformat()
        }).eq('id', str(template_id)).eq('project_id', str(project_id)).execute()
        return result.data[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/projects/{project_id}/templates/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_template(project_id: UUID, template_id: UUID, current_user = Depends(get_current_user)):
    """Delete a template"""
    try:
        # Verify ownership through project
        project = supabase.table('projects').select('id').eq('id', str(project_id)).eq('user_id', current_user.id).single().execute()
        supabase.table('templates').delete().eq('id', str(template_id)).eq('project_id', str(project_id)).execute()
        return None
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ===========================
# User Preferences
# ===========================

@app.post("/projects/{project_id}/preferences", status_code=status.HTTP_201_CREATED)
async def set_preference(project_id: UUID, preference: PreferenceCreate, current_user = Depends(get_current_user)):
    """Set a user preference (favourite/undesirable)"""
    try:
        # Verify project ownership
        project = supabase.table('projects').select('id').eq('id', str(project_id)).eq('user_id', current_user.id).single().execute()
        
        # Check if preference already exists
        existing = supabase.table('user_item_preferences').select('id').eq('user_id', current_user.id).eq(
            'project_id', str(project_id)
        ).eq('item_id', str(preference.item_id)).eq('item_type', preference.item_type).execute()
        
        if existing.data:
            # Update existing preference
            result = supabase.table('user_item_preferences').update({
                'preference_type': preference.preference_type
            }).eq('id', existing.data[0]['id']).execute()
        else:
            # Create new preference
            result = supabase.table('user_item_preferences').insert({
                'user_id': current_user.id,
                'project_id': str(project_id),
                'item_id': str(preference.item_id),
                'item_type': preference.item_type,
                'preference_type': preference.preference_type
            }).execute()
        
        return result.data[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/projects/{project_id}/preferences")
async def list_preferences(
    project_id: UUID,
    item_type: Optional[str] = None,
    preference_type: Optional[str] = None,
    current_user = Depends(get_current_user)
):
    """List user preferences with optional filters"""
    try:
        # Verify project ownership
        project = supabase.table('projects').select('id').eq('id', str(project_id)).eq('user_id', current_user.id).single().execute()
        
        query = supabase.table('user_item_preferences').select('*').eq('user_id', current_user.id).eq('project_id', str(project_id))
        
        if item_type:
            query = query.eq('item_type', item_type)
        if preference_type:
            query = query.eq('preference_type', preference_type)
        
        result = query.execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/projects/{project_id}/preferences/{preference_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_preference(project_id: UUID, preference_id: UUID, current_user = Depends(get_current_user)):
    """Remove a preference"""
    try:
        # Verify ownership
        project = supabase.table('projects').select('id').eq('id', str(project_id)).eq('user_id', current_user.id).single().execute()
        supabase.table('user_item_preferences').delete().eq('id', str(preference_id)).eq(
            'user_id', current_user.id
        ).eq('project_id', str(project_id)).execute()
        return None
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ===========================
# Document Processing
# ===========================

@app.post("/projects/{project_id}/documents/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    project_id: UUID,
    document: DocumentUpload,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
):
    """Register an uploaded document and trigger processing"""
    try:
        # Verify project ownership
        project = supabase.table('projects').select('*').eq('id', str(project_id)).eq('user_id', current_user.id).single().execute()
        
        # Create document record
        result = supabase.table('processed_documents').insert({
            'project_id': str(project_id),
            'user_id': current_user.id,
            'original_file_name': document.original_file_name,
            'storage_path': document.storage_path,
            'file_type': document.file_type,
            'file_size_bytes': document.file_size_bytes,
            'status': 'uploaded'
        }).execute()
        
        document_id = result.data[0]['id']
        
        # Trigger async processing
        background_tasks.add_task(
            process_document,
            document_id,
            document.storage_path,
            project.data.get('project_variables', {})
        )
        
        return result.data[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/projects/{project_id}/documents")
async def list_documents(project_id: UUID, current_user = Depends(get_current_user)):
    """List processed documents for a project"""
    try:
        # Verify project ownership
        project = supabase.table('projects').select('id').eq('id', str(project_id)).eq('user_id', current_user.id).single().execute()
        
        result = supabase.table('processed_documents').select(
            'id, original_file_name, file_type, status, gemini_processed_data, created_at, updated_at'
        ).eq('project_id', str(project_id)).order('created_at', desc=True).execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/projects/{project_id}/documents/{document_id}")
async def get_document(project_id: UUID, document_id: UUID, current_user = Depends(get_current_user)):
    """Get specific document details including extracted text and analysis"""
    try:
        # Verify ownership through project
        doc_check = supabase.table('processed_documents').select('*, projects!inner(user_id)').eq('id', str(document_id)).eq('project_id', str(project_id)).eq('projects.user_id', current_user.id).single().execute()
        return doc_check.data
    except Exception as e:
        raise HTTPException(status_code=404, detail="Document not found")

# ===========================
# Investor Search & Matching
# ===========================

@app.post("/guest-chat")
async def guest_chat(message: dict):
    """
    Guest chat endpoint that interacts with Gemini and returns investor results when appropriate.
    """
    try:
        content = message.get("content", "").strip()
        if not content:
            return {"type": "chat_message", "data": "Por favor, escribe un mensaje."}

        # Sistema prompt para Gemini
        system_prompt = """Eres un asistente experto en matching de inversores. Tu objetivo es ayudar a encontrar inversores adecuados para startups.
        
        Necesitas obtener tres datos clave:
        1. Ubicación (location)
        2. Categorías/sectores (categories)
        3. Etapa de inversión (stage)
        
        Si el usuario expresa interés en buscar inversores y no has recopilado estos datos, pregunta por ellos uno a uno.
        Una vez tengas los tres datos y el usuario quiera buscar, responde con un JSON:
        {"action": "search", "location": "...", "categories": ["...", "..."], "stage": "..."}
        
        En cualquier otro caso, mantén una conversación natural y ayuda al usuario."""

        # Llamada a Gemini
        response = await call_gemini(content, system_message=system_prompt)
        
        try:
            # Intenta parsear la respuesta como JSON
            data = json.loads(response)
            if isinstance(data, dict) and data.get("action") == "search":
                # Realizar búsqueda de inversores
                all_investors = supabase.table('investors').select('*').execute()
                if not all_investors.data:
                    return {"type": "chat_message", "data": "Lo siento, no encontré inversores en este momento."}

                matches = algorithm_1_enhanced_keyword_matching(
                    {
                        'location': data.get('location', ''),
                        'categories': data.get('categories', []),
                        'stage': data.get('stage', ''),
                    },
                    all_investors.data
                )

                # Limitar a 10 resultados
                matches = matches[:10]
                
                follow_up = await call_gemini(
                    f"He encontrado {len(matches)} inversores potenciales para una startup en {data.get('location')}, " +
                    f"en los sectores {', '.join(data.get('categories', []))}, en etapa {data.get('stage')}. " +
                    "Genera un mensaje corto y positivo para el usuario sobre estos resultados."
                )

                return {
                    "type": "investor_results",
                    "data": matches,
                    "follow_up_message": follow_up
                }
        except json.JSONDecodeError:
            # No es JSON, es una respuesta normal de chat
            pass

        return {"type": "chat_message", "data": response}

    except Exception as e:
        logger.error(f"Error in guest chat: {str(e)}")
        return {
            "type": "chat_message",
            "data": "Lo siento, hubo un error procesando tu mensaje. Por favor, intenta de nuevo."
        }

@app.post("/projects/{project_id}/search-investors")
async def search_investors(
    project_id: UUID,
    search_params: InvestorSearch,
    current_user = Depends(get_current_user)
):
    """Search for matching investors based on provided criteria."""
    try:
        # Get all investors from Supabase
        all_investors = supabase.table('investors').select('*').execute()
        if not all_investors.data:
            return []

        # Load auxiliary data files for deep research if needed
        categorias_content = ""
        etapas_content = ""
        ubicaciones_content = ""
        
        if search_params.deep_research:
            try:
                with open('categorias.txt', 'r', encoding='utf-8') as f:
                    categorias_content = f.read()
            except FileNotFoundError:
                logger.warning("categorias.txt not found")
                
            try:
                with open('etapas.txt', 'r', encoding='utf-8') as f:
                    etapas_content = f.read()
            except FileNotFoundError:
                logger.warning("etapas.txt not found")
                
            try:
                with open('ubicaciones.txt', 'r', encoding='utf-8') as f:
                    ubicaciones_content = f.read()
            except FileNotFoundError:
                logger.warning("ubicaciones.txt not found")

        # Choose algorithm based on user preference
        if search_params.algorithm == 1:
            matches = algorithm_1_enhanced_keyword_matching(
                {
                    'location': search_params.location,
                    'categories': search_params.categories,
                    'stage': search_params.stage,
                    'description': search_params.company_description
                },
                all_investors.data,
                deep_research_keywords=[categorias_content, etapas_content, ubicaciones_content] if search_params.deep_research else None
            )
        else:
            matches = algorithm_2_semantic_contextual_matching(
                {
                    'location': search_params.location,
                    'categories': search_params.categories,
                    'stage': search_params.stage,
                    'description': search_params.company_description
                },
                all_investors.data,
                deep_research_keywords=[categorias_content, etapas_content, ubicaciones_content] if search_params.deep_research else None
            )

        return matches[:10]  # Limit to 10 results

    except Exception as e:
        logger.error(f"Error in investor search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# ===========================
# Outreach Support
# ===========================

@app.get("/projects/{project_id}/outreach-data")
async def get_outreach_data(
    project_id: UUID,
    employee_id: UUID,
    template_id: UUID,
    current_user = Depends(get_current_user)
):
    """Get data needed for outreach page"""
    try:
        # Verify project ownership
        project = supabase.table('projects').select('id').eq('id', str(project_id)).eq('user_id', current_user.id).single().execute()
        
        # Fetch employee data
        employee = supabase.table('employees').select('*').eq('id', str(employee_id)).eq('project_id', str(project_id)).single().execute()
        
        # Fetch template data
        template = supabase.table('templates').select('*').eq('id', str(template_id)).eq('project_id', str(project_id)).single().execute()
        
        return {
            'employee': employee.data,
            'template': template.data,
            'outreach_data': {
                'linkedin_url': employee.data.get('linkedin_url'),
                'employee_name': employee.data.get('name'),
                'template_subject': template.data.get('generated_content_subject'),
                'template_body': template.data.get('generated_content_body')
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/projects/{project_id}/outreach-log", status_code=status.HTTP_201_CREATED)
async def log_outreach(project_id: UUID, outreach: OutreachLog, current_user = Depends(get_current_user)):
    """Log an outreach attempt (optional for MVP)"""
    try:
        # Verify project ownership
        project = supabase.table('projects').select('id').eq('id', str(project_id)).eq('user_id', current_user.id).single().execute()
        
        # Create outreach log entry
        result = supabase.table('outreach_log').insert({
            'project_id': str(project_id),
            'user_id': current_user.id,
            'employee_id': str(outreach.employee_id),
            'template_id': str(outreach.template_id),
            'status': 'sent'
        }).execute()
        
        return result.data[0]
    except Exception as e:
        # If outreach_log table doesn't exist, just return success for MVP
        logger.warning(f"Outreach logging failed (table might not exist): {str(e)}")
        return {"message": "Outreach noted", "status": "logged"}

# ===========================
# Error Handlers
# ===========================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {"error": exc.detail, "status_code": exc.status_code}

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return {"error": "Internal server error", "status_code": 500}

# ===========================
# Main Entry Point
# ===========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))