# -*- coding: utf-8 -*-
"""
0Bullshit Backend v2.0 - Sistema Gamificado con 60 Bots
Sistema de créditos, suscripciones y memoria neuronal
"""

# ==============================================================================
#           IMPORTS
# ==============================================================================

print("1. Loading libraries...")
from flask import Flask, request, jsonify, session
from flask_cors import CORS
import pandas as pd
import google.generativeai as genai
import json
import ast
import re
import warnings
import time
import os
import sqlalchemy
from datetime import datetime, timedelta
import uuid
from sqlalchemy import text
import secrets
import stripe
import jwt
from functools import wraps
import hashlib
import bcrypt

# ==============================================================================
#           CONFIGURATION
# ==============================================================================

print("2. Configuring application...")
app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = secrets.token_hex(16)
warnings.filterwarnings('ignore')

# Environment Variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY")  # Para Opus4
DATABASE_URL = os.environ.get("DATABASE_URL")
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")
JWT_SECRET = os.environ.get("JWT_SECRET", secrets.token_hex(32))
UNIPILE_API_KEY = os.environ.get("UNIPILE_API_KEY")  # Para Pro plan

if not GEMINI_API_KEY:
    print("❌ FATAL: GEMINI_API_KEY not found.")
if not DATABASE_URL:
    print("❌ FATAL: DATABASE_URL not found.")

# Configure AI APIs
try:
    genai.configure(api_key=GEMINI_API_KEY)
    MODEL_NAME = "gemini-2.0-flash"
    print("✅ Gemini API configured.")
except Exception as e:
    print(f"❌ ERROR configuring Gemini: {e}")

# Configure Stripe
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY
    print("✅ Stripe configured.")

# Connect to Supabase
try:
    engine = sqlalchemy.create_engine(DATABASE_URL)
    print("✅ Supabase connection established.")
except Exception as e:
    print(f"❌ ERROR connecting to Supabase: {e}")
    engine = None

# ==============================================================================
#           CONSTANTS AND CONFIGURATIONS
# ==============================================================================

# Credit costs por acción
CREDIT_COSTS = {
    # Bots básicos (todos los planes)
    "basic_bot": 5,
    "advanced_bot": 15,
    "expert_bot": 25,
    "document_generation": 50,
    
    # Growth plan only
    "investor_search_result": 10,
    "employee_search_result": 8,
    
    # Pro plan only
    "template_generation": 20,
    "unipile_message": 5,
    "automated_sequence": 50,
    
    # Premium features
    "market_analysis": 100,
    "business_model": 150,
    "pitch_deck": 200
}

# Planes de suscripción
SUBSCRIPTION_PLANS = {
    "free": {
        "name": "Free",
        "price": 0,
        "credits_monthly": 100,
        "launch_credits": 100,
        "features": {
            "bots_access": True,
            "investor_search": False,
            "employee_search": False,
            "outreach_templates": False,
            "unlimited_docs": False,
            "neural_memory": False
        }
    },
    "growth": {
        "name": "Growth",
        "price": 20,
        "stripe_price_id": "price_growth_monthly",
        "credits_monthly": 10000,
        "launch_credits": 100000,
        "features": {
            "bots_access": True,
            "investor_search": True,
            "employee_search": True,
            "outreach_templates": False,
            "unlimited_docs": True,
            "neural_memory": True
        }
    },
    "pro": {
        "name": "Pro Outreach",
        "price": 89,
        "stripe_price_id": "price_pro_monthly",
        "credits_monthly": 50000,
        "launch_credits": 1000000,
        "features": {
            "bots_access": True,
            "investor_search": True,
            "employee_search": True,
            "outreach_templates": True,
            "unlimited_docs": True,
            "neural_memory": True,
            "unipile_integration": True
        }
    }
}

# ==============================================================================
#           AUTHENTICATION & USER MANAGEMENT
# ==============================================================================

def hash_password(password):
    """Hash password usando bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed):
    """Verifica password contra hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def generate_jwt_token(user_id):
    """Genera JWT token para el usuario"""
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(days=7),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

def verify_jwt_token(token):
    """Verifica y decodifica JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        return payload['user_id']
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def require_auth(f):
    """Decorator para endpoints que requieren autenticación"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No authorization token provided'}), 401
        
        if token.startswith('Bearer '):
            token = token[7:]
        
        user_id = verify_jwt_token(token)
        if not user_id:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        # Obtener usuario de la base de datos
        user = get_user_by_id(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 401
        
        # Pasar usuario al endpoint
        return f(user, *args, **kwargs)
    
    return decorated_function

def require_plan(required_plan):
    """Decorator para verificar plan de suscripción"""
    def decorator(f):
        @wraps(f)
        def decorated_function(user, *args, **kwargs):
            user_plan = user.get('plan', 'free')
            
            # Verificar acceso por plan
            plan_hierarchy = {'free': 0, 'growth': 1, 'pro': 2}
            required_level = plan_hierarchy.get(required_plan, 0)
            user_level = plan_hierarchy.get(user_plan, 0)
            
            if user_level < required_level:
                return jsonify({
                    'error': 'Plan upgrade required',
                    'required_plan': required_plan,
                    'current_plan': user_plan,
                    'upsell': True,
                    'upgrade_url': f'/upgrade/{required_plan}'
                }), 403
            
            return f(user, *args, **kwargs)
        return decorated_function
    return decorator

# ==============================================================================
#           DATABASE HELPERS
# ==============================================================================

def get_user_by_id(user_id):
    """Obtiene usuario por ID"""
    try:
        query = """
        SELECT u.*, s.status as subscription_status, s.stripe_subscription_id
        FROM users u
        LEFT JOIN subscriptions s ON u.id = s.user_id AND s.status = 'active'
        WHERE u.id = %s
        """
        result = pd.read_sql(query, engine, params=[user_id])
        if result.empty:
            return None
        return result.iloc[0].to_dict()
    except Exception as e:
        print(f"❌ ERROR getting user: {e}")
        return None

def get_user_by_email(email):
    """Obtiene usuario por email"""
    try:
        query = "SELECT * FROM users WHERE email = %s"
        result = pd.read_sql(query, engine, params=[email])
        if result.empty:
            return None
        return result.iloc[0].to_dict()
    except Exception as e:
        print(f"❌ ERROR getting user by email: {e}")
        return None

def create_user(email, password, first_name, last_name):
    """Crea nuevo usuario"""
    try:
        user_id = str(uuid.uuid4())
        hashed_password = hash_password(password)
        
        query = """
        INSERT INTO users (id, email, password_hash, first_name, last_name, plan, credits_balance, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        params = (
            user_id, email, hashed_password, first_name, last_name,
            'free', SUBSCRIPTION_PLANS['free']['launch_credits'],
            datetime.now(), datetime.now()
        )
        
        with engine.connect() as conn:
            conn.execute(text(query), params)
            conn.commit()
        
        # Inicializar memoria neuronal
        init_neural_memory(user_id)
        
        return user_id
    except Exception as e:
        print(f"❌ ERROR creating user: {e}")
        return None

def get_user_credits(user_id):
    """Obtiene créditos del usuario"""
    try:
        query = "SELECT credits_balance FROM users WHERE id = %s"
        result = pd.read_sql(query, engine, params=[user_id])
        if result.empty:
            return 0
        return result.iloc[0]['credits_balance']
    except Exception as e:
        print(f"❌ ERROR getting credits: {e}")
        return 0

def charge_credits(user_id, amount):
    """Cobra créditos al usuario"""
    try:
        query = """
        UPDATE users 
        SET credits_balance = credits_balance - %s, updated_at = %s
        WHERE id = %s AND credits_balance >= %s
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(query), (amount, datetime.now(), user_id, amount))
            conn.commit()
            
            if result.rowcount == 0:
                return False  # Créditos insuficientes
        
        # Log de transacción
        log_credit_transaction(user_id, -amount, 'charge', 'Bot usage')
        return True
    except Exception as e:
        print(f"❌ ERROR charging credits: {e}")
        return False

def add_credits(user_id, amount, reason='purchase'):
    """Añade créditos al usuario"""
    try:
        query = """
        UPDATE users 
        SET credits_balance = credits_balance + %s, updated_at = %s
        WHERE id = %s
        """
        
        with engine.connect() as conn:
            conn.execute(text(query), (amount, datetime.now(), user_id))
            conn.commit()
        
        log_credit_transaction(user_id, amount, 'add', reason)
        return True
    except Exception as e:
        print(f"❌ ERROR adding credits: {e}")
        return False

def log_credit_transaction(user_id, amount, transaction_type, description):
    """Log de transacciones de créditos"""
    try:
        query = """
        INSERT INTO credit_transactions (id, user_id, amount, transaction_type, description, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        params = (
            str(uuid.uuid4()), user_id, amount, transaction_type, description, datetime.now()
        )
        
        with engine.connect() as conn:
            conn.execute(text(query), params)
            conn.commit()
    except Exception as e:
        print(f"❌ ERROR logging transaction: {e}")

# ==============================================================================
#           GEMINI ARMY - 60 BOTS SYSTEM
# ==============================================================================

# Importar el ejército de bots del archivo separado
from bots.gemini_army import GEMINI_ARMY, execute_gemini_bot

class BotManager:
    def __init__(self):
        self.bots = GEMINI_ARMY
        self.router = GeminiRouter()
    
    def process_user_request(self, user_input, user_context, user_id):
        """Procesa solicitud del usuario con IA router"""
        try:
            # Router decide qué bot usar
            selected_bot_id = self.router.select_optimal_bot(user_input, user_context)
            
            if not selected_bot_id or selected_bot_id not in self.bots:
                selected_bot_id = "general_consultant"  # Bot por defecto
            
            selected_bot = self.bots[selected_bot_id]
            
            # Verificar créditos
            required_credits = selected_bot["credit_cost"]
            user_credits = get_user_credits(user_id)
            
            if user_credits < required_credits:
                return {
                    "error": "insufficient_credits",
                    "required": required_credits,
                    "available": user_credits,
                    "upsell": True
                }
            
            # Ejecutar bot
            result = execute_gemini_bot(selected_bot_id, user_context, user_input, user_credits)
            
            if result.get("success"):
                # Cobrar créditos
                charge_credits(user_id, required_credits)
                
                # Guardar en memoria neuronal
                save_neural_interaction(user_id, {
                    "input": user_input,
                    "bot_used": selected_bot_id,
                    "output": result["response"],
                    "credits_charged": required_credits,
                    "context": user_context
                })
            
            return result
            
        except Exception as e:
            print(f"❌ ERROR in bot manager: {e}")
            return {"error": "Error processing request"}

class GeminiRouter:
    """Router inteligente que decide qué bot usar"""
    
    def select_optimal_bot(self, user_input, user_context):
        """Usa Gemini para seleccionar el bot óptimo"""
        try:
            router_prompt = f"""
            Eres un router inteligente que decide qué bot especializado usar.
            
            Input del usuario: "{user_input}"
            Contexto: {user_context}
            
            Bots disponibles: {list(GEMINI_ARMY.keys())}
            
            Responde SOLO con el ID del bot más apropiado.
            
            Ejemplos:
            - "necesito un pitch deck" → "pitch_deck_master"
            - "buscar inversores" → "investor_researcher" 
            - "estrategia de producto" → "product_visionary"
            - "análisis financiero" → "cfo_virtual"
            
            Bot ID:"""
            
            model = genai.GenerativeModel(MODEL_NAME)
            response = model.generate_content(router_prompt)
            
            selected_bot = response.text.strip().replace('"', '').replace("'", "")
            
            if selected_bot in GEMINI_ARMY:
                return selected_bot
            
            return "general_consultant"  # Fallback
            
        except Exception as e:
            print(f"❌ ERROR in router: {e}")
            return "general_consultant"

# ==============================================================================
#           NEURAL MEMORY SYSTEM
# ==============================================================================

def init_neural_memory(user_id):
    """Inicializa memoria neuronal para nuevo usuario"""
    try:
        memory_id = str(uuid.uuid4())
        query = """
        INSERT INTO neural_memory (id, user_id, memory_type, memory_data, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        initial_memory = {
            "interactions_count": 0,
            "preferred_bots": {},
            "success_patterns": {},
            "startup_context": {},
            "learning_preferences": {}
        }
        
        params = (
            memory_id, user_id, 'main_memory', 
            json.dumps(initial_memory), datetime.now(), datetime.now()
        )
        
        with engine.connect() as conn:
            conn.execute(text(query), params)
            conn.commit()
            
    except Exception as e:
        print(f"❌ ERROR initializing neural memory: {e}")

def save_neural_interaction(user_id, interaction_data):
    """Guarda interacción en memoria neuronal"""
    try:
        interaction_id = str(uuid.uuid4())
        query = """
        INSERT INTO neural_interactions (id, user_id, bot_used, user_input, bot_output, 
                                       credits_charged, context_data, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        params = (
            interaction_id, user_id, interaction_data["bot_used"],
            interaction_data["input"], interaction_data["output"],
            interaction_data["credits_charged"], json.dumps(interaction_data["context"]),
            datetime.now()
        )
        
        with engine.connect() as conn:
            conn.execute(text(query), params)
            conn.commit()
            
        # Actualizar memoria principal
        update_neural_memory(user_id, interaction_data)
        
    except Exception as e:
        print(f"❌ ERROR saving neural interaction: {e}")

def update_neural_memory(user_id, interaction_data):
    """Actualiza memoria neuronal principal"""
    try:
        # Obtener memoria actual
        query = "SELECT memory_data FROM neural_memory WHERE user_id = %s AND memory_type = 'main_memory'"
        result = pd.read_sql(query, engine, params=[user_id])
        
        if result.empty:
            init_neural_memory(user_id)
            return
        
        memory = json.loads(result.iloc[0]['memory_data'])
        
        # Actualizar con nueva interacción
        memory["interactions_count"] += 1
        
        bot_used = interaction_data["bot_used"]
        if bot_used not in memory["preferred_bots"]:
            memory["preferred_bots"][bot_used] = 0
        memory["preferred_bots"][bot_used] += 1
        
        # Guardar memoria actualizada
        update_query = """
        UPDATE neural_memory 
        SET memory_data = %s, updated_at = %s 
        WHERE user_id = %s AND memory_type = 'main_memory'
        """
        
        with engine.connect() as conn:
            conn.execute(text(update_query), (json.dumps(memory), datetime.now(), user_id))
            conn.commit()
            
    except Exception as e:
        print(f"❌ ERROR updating neural memory: {e}")

def get_neural_memory(user_id):
    """Obtiene memoria neuronal del usuario"""
    try:
        query = "SELECT memory_data FROM neural_memory WHERE user_id = %s AND memory_type = 'main_memory'"
        result = pd.read_sql(query, engine, params=[user_id])
        
        if result.empty:
            return {}
        
        return json.loads(result.iloc[0]['memory_data'])
    except Exception as e:
        print(f"❌ ERROR getting neural memory: {e}")
        return {}

# ==============================================================================
#           SEARCH ALGORITHMS (Growth/Pro only)
# ==============================================================================

def intelligent_keyword_extraction(query, user_context):
    """Extrae keywords inteligentemente usando Gemini"""
    try:
        prompt = f"""
        Convierte esta consulta en keywords precisas para búsqueda de inversores:
        
        Query: "{query}"
        Context del usuario: {user_context}
        
        Extrae:
        1. Ubicaciones (países, ciudades, regiones)
        2. Etapas de inversión (seed, series A, growth, etc.)
        3. Categorías/sectores (fintech, healthtech, AI, etc.)
        
        Responde en JSON:
        {{
            "ubicacion": ["keyword1", "keyword2"],
            "etapa": ["keyword1", "keyword2"], 
            "categoria": ["keyword1", "keyword2"]
        }}
        """
        
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        
        # Extraer JSON de la respuesta
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        
        return {"ubicacion": [], "etapa": [], "categoria": []}
        
    except Exception as e:
        print(f"❌ ERROR in keyword extraction: {e}")
        return {"ubicacion": [], "etapa": [], "categoria": []}

def ml_investor_search(query, user_preferences, max_results=20):
    """Búsqueda de inversores con ML scoring"""
    if not engine:
        return {"error": "No database connection"}
    
    try:
        # Extraer keywords inteligentemente
        keywords = intelligent_keyword_extraction(query, user_preferences.get('context', {}))
        
        # Cargar inversores de la BD
        investors_query = """
        SELECT id, "Company_Name", "Company_Description", "Investing_Stage",
               "Company_Location", "Investment_Categories", "Company_Linkedin",
               "Keywords_Ubicacion_Adicionales", "Keywords_Etapas_Adicionales", 
               "Keywords_Categorias_Adicionales"
        FROM investors
        LIMIT 1000
        """
        
        investors_df = pd.read_sql(investors_query, engine)
        
        if investors_df.empty:
            return {"error": "No investors found"}
        
        # Aplicar ML scoring
        scored_investors = apply_ml_scoring(investors_df, keywords, user_preferences)
        
        # Ordenar y limitar resultados
        top_investors = scored_investors.head(max_results)
        
        return {
            "search_type": "ml_powered",
            "results": top_investors.to_dict('records'),
            "total_found": len(top_investors),
            "keywords_used": keywords
        }
        
    except Exception as e:
        print(f"❌ ERROR in ML investor search: {e}")
        return {"error": f"Search error: {e}"}

def apply_ml_scoring(investors_df, keywords, user_preferences):
    """Aplica scoring ML con pesos configurables"""
    try:
        # Parsear keywords de inversores
        investors_df['ubicacion_list'] = investors_df['Keywords_Ubicacion_Adicionales'].apply(parse_keywords)
        investors_df['etapa_list'] = investors_df['Keywords_Etapas_Adicionales'].apply(parse_keywords)
        investors_df['categoria_list'] = investors_df['Keywords_Categorias_Adicionales'].apply(parse_keywords)
        
        # Obtener pesos del usuario (por defecto: etapa 40%, categoría 40%, ubicación 20%)
        weights = user_preferences.get('weights', {
            'etapa': 0.4,
            'categoria': 0.4,
            'ubicacion': 0.2
        })
        
        # Calcular scores
        def calculate_score(row):
            ubicacion_score = calculate_match_score(row['ubicacion_list'], keywords['ubicacion'])
            etapa_score = calculate_match_score(row['etapa_list'], keywords['etapa'])
            categoria_score = calculate_match_score(row['categoria_list'], keywords['categoria'])
            
            total_score = (
                ubicacion_score * weights['ubicacion'] +
                etapa_score * weights['etapa'] +
                categoria_score * weights['categoria']
            ) * 100
            
            return total_score
        
        investors_df['ml_score'] = investors_df.apply(calculate_score, axis=1)
        
        # Filtrar solo los que tienen score > 0 y ordenar
        return investors_df[investors_df['ml_score'] > 0].sort_values('ml_score', ascending=False)
        
    except Exception as e:
        print(f"❌ ERROR in ML scoring: {e}")
        return investors_df

def calculate_match_score(investor_keywords, query_keywords):
    """Calcula score de matching entre keywords"""
    if not investor_keywords or not query_keywords:
        return 0
    
    investor_set = set(str(k).lower() for k in investor_keywords)
    query_set = set(str(k).lower() for k in query_keywords)
    
    intersection = investor_set.intersection(query_set)
    union = investor_set.union(query_set)
    
    if not union:
        return 0
    
    return len(intersection) / len(union)

def parse_keywords(value):
    """Parsea keywords de la BD"""
    if pd.isna(value) or str(value).strip() in ['[]', '']:
        return []
    try:
        result = ast.literal_eval(str(value))
        if isinstance(result, list):
            return [str(item).strip().lower() for item in result if str(item).strip()]
        else:
            return [str(result).strip().lower()] if str(result).strip() else []
    except (ValueError, SyntaxError):
        return [k.strip().lower() for k in str(value).split(',') if k.strip()]

# ==============================================================================
#           API ROUTES
# ==============================================================================

print("5. Defining API routes...")

@app.route('/')
def home():
    """Health check"""
    return jsonify({
        "status": "0Bullshit Backend v2.0 Ready! 🚀",
        "timestamp": datetime.now().isoformat(),
        "features": ["60 Bots", "Credits System", "Neural Memory", "3 Plans"]
    })

# ==================== AUTHENTICATION ROUTES ====================

@app.route('/auth/register', methods=['POST'])
def register():
    """Registro de usuario"""
    try:
        data = request.json
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        first_name = data.get('first_name', '').strip()
        last_name = data.get('last_name', '').strip()
        
        if not all([email, password, first_name, last_name]):
            return jsonify({"error": "All fields are required"}), 400
        
        # Verificar si el usuario ya existe
        existing_user = get_user_by_email(email)
        if existing_user:
            return jsonify({"error": "User already exists"}), 409
        
        # Crear usuario
        user_id = create_user(email, password, first_name, last_name)
        if not user_id:
            return jsonify({"error": "Error creating user"}), 500
        
        # Generar token
        token = generate_jwt_token(user_id)
        
        return jsonify({
            "message": "User registered successfully",
            "token": token,
            "user": {
                "id": user_id,
                "email": email,
                "first_name": first_name,
                "last_name": last_name,
                "plan": "free",
                "credits": SUBSCRIPTION_PLANS['free']['launch_credits']
            }
        })
        
    except Exception as e:
        print(f"❌ ERROR in register: {e}")
        return jsonify({"error": "Registration failed"}), 500

@app.route('/auth/login', methods=['POST'])
def login():
    """Login de usuario"""
    try:
        data = request.json
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400
        
        # Verificar usuario
        user = get_user_by_email(email)
        if not user or not verify_password(password, user['password_hash']):
            return jsonify({"error": "Invalid credentials"}), 401
        
        # Generar token
        token = generate_jwt_token(user['id'])
        
        return jsonify({
            "message": "Login successful",
            "token": token,
            "user": {
                "id": user['id'],
                "email": user['email'],
                "first_name": user['first_name'],
                "last_name": user['last_name'],
                "plan": user['plan'],
                "credits": user['credits_balance']
            }
        })
        
    except Exception as e:
        print(f"❌ ERROR in login: {e}")
        return jsonify({"error": "Login failed"}), 500

# ==================== USER & CREDITS ROUTES ====================

@app.route('/user/profile', methods=['GET'])
@require_auth
def get_profile(user):
    """Obtiene perfil del usuario"""
    try:
        neural_memory = get_neural_memory(user['id'])
        
        return jsonify({
            "user": {
                "id": user['id'],
                "email": user['email'],
                "first_name": user['first_name'],
                "last_name": user['last_name'],
                "plan": user['plan'],
                "credits": user['credits_balance'],
                "created_at": user['created_at']
            },
            "stats": {
                "interactions_count": neural_memory.get('interactions_count', 0),
                "preferred_bots": neural_memory.get('preferred_bots', {}),
                "plan_features": SUBSCRIPTION_PLANS[user['plan']]['features']
            }
        })
        
    except Exception as e:
        print(f"❌ ERROR getting profile: {e}")
        return jsonify({"error": "Error getting profile"}), 500

@app.route('/credits/balance', methods=['GET'])
@require_auth
def get_credits_balance(user):
    """Obtiene balance de créditos"""
    current_credits = get_user_credits(user['id'])
    
    return jsonify({
        "credits_balance": current_credits,
        "plan": user['plan'],
        "monthly_credits": SUBSCRIPTION_PLANS[user['plan']].get('credits_monthly', 0)
    })

@app.route('/credits/purchase', methods=['POST'])
@require_auth
def purchase_credits(user):
    """Compra de créditos con Stripe"""
    try:
        data = request.json
        credit_package = data.get('package')  # 'small', 'medium', 'large'
        
        credit_packages = {
            'small': {'credits': 1000, 'price': 1000},  # $10.00
            'medium': {'credits': 5000, 'price': 4000},  # $40.00 
            'large': {'credits': 20000, 'price': 12000}  # $120.00
        }
        
        if credit_package not in credit_packages:
            return jsonify({"error": "Invalid credit package"}), 400
        
        package_info = credit_packages[credit_package]
        
        # Crear Stripe Checkout Session
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'eur',
                    'product_data': {
                        'name': f'{package_info["credits"]} Credits',
                        'description': f'0Bullshit Credit Package - {credit_package.title()}'
                    },
                    'unit_amount': package_info['price'],
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=f"{request.host_url}payment/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{request.host_url}payment/cancel",
            metadata={
                'user_id': user['id'],
                'credit_amount': package_info['credits'],
                'package_type': credit_package
            }
        )
        
        return jsonify({
            "checkout_url": checkout_session.url,
            "session_id": checkout_session.id
        })
        
    except Exception as e:
        print(f"❌ ERROR creating checkout: {e}")
        return jsonify({"error": "Error creating payment"}), 500

# ==================== BOTS & CHAT ROUTES ====================

@app.route('/chat/bot', methods=['POST'])
@require_auth
def chat_with_bot(user):
    """Chat con sistema de 60 bots"""
    try:
        data = request.json
        user_input = data.get('message', '')
        context = data.get('context', {})
        
        if not user_input:
            return jsonify({"error": "Message is required"}), 400
        
        # Añadir contexto del usuario
        user_context = {
            **context,
            "user_id": user['id'],
            "user_plan": user['plan'],
            "user_credits": user['credits_balance'],
            "neural_memory": get_neural_memory(user['id'])
        }
        
        # Procesar con bot manager
        bot_manager = BotManager()
        result = bot_manager.process_user_request(user_input, user_context, user['id'])
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ ERROR in bot chat: {e}")
        return jsonify({"error": "Error processing message"}), 500

@app.route('/bots/available', methods=['GET'])
@require_auth
def get_available_bots(user):
    """Lista de bots disponibles"""
    try:
        user_plan = user['plan']
        plan_features = SUBSCRIPTION_PLANS[user_plan]['features']
        
        available_bots = {}
        for bot_id, bot_info in GEMINI_ARMY.items():
            # Todos los usuarios tienen acceso a los bots básicos
            if plan_features['bots_access']:
                available_bots[bot_id] = {
                    "name": bot_info["name"],
                    "description": bot_info["description"],
                    "credit_cost": bot_info["credit_cost"],
                    "category": bot_info.get("category", "general")
                }
        
        return jsonify({
            "available_bots": available_bots,
            "total_bots": len(available_bots),
            "user_plan": user_plan,
            "plan_features": plan_features
        })
        
    except Exception as e:
        print(f"❌ ERROR getting bots: {e}")
        return jsonify({"error": "Error getting available bots"}), 500

# ==================== SEARCH ROUTES (Growth/Pro only) ====================

@app.route('/search/investors', methods=['POST'])
@require_auth
@require_plan('growth')
def search_investors(user):
    """Búsqueda de inversores con ML"""
    try:
        data = request.json
        query = data.get('query', '')
        preferences = data.get('preferences', {})
        max_results = data.get('max_results', 20)
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        # Verificar créditos necesarios
        estimated_cost = max_results * CREDIT_COSTS['investor_search_result']
        user_credits = get_user_credits(user['id'])
        
        if user_credits < estimated_cost:
            return jsonify({
                "error": "insufficient_credits",
                "required": estimated_cost,
                "available": user_credits,
                "upsell": True
            }), 402
        
        # Ejecutar búsqueda ML
        results = ml_investor_search(query, preferences, max_results)
        
        if 'error' not in results:
            # Cobrar créditos por resultados encontrados
            actual_cost = len(results['results']) * CREDIT_COSTS['investor_search_result']
            charge_credits(user['id'], actual_cost)
            
            results['credits_charged'] = actual_cost
        
        return jsonify(results)
        
    except Exception as e:
        print(f"❌ ERROR in investor search: {e}")
        return jsonify({"error": "Search failed"}), 500

@app.route('/search/employees', methods=['POST'])
@require_auth
@require_plan('growth')
def search_employees(user):
    """Búsqueda de empleados en fondos"""
    try:
        data = request.json
        query = data.get('query', '')
        max_employees = data.get('max_employees', 50)
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        # Verificar créditos
        estimated_cost = max_employees * CREDIT_COSTS['employee_search_result']
        user_credits = get_user_credits(user['id'])
        
        if user_credits < estimated_cost:
            return jsonify({
                "error": "insufficient_credits", 
                "required": estimated_cost,
                "available": user_credits,
                "upsell": True
            }), 402
        
        # Ejecutar búsqueda (implementar lógica similar a inversores)
        # Por ahora placeholder
        results = {
            "search_type": "employees",
            "results": [],
            "total_found": 0,
            "message": "Employee search functionality coming soon"
        }
        
        return jsonify(results)
        
    except Exception as e:
        print(f"❌ ERROR in employee search: {e}")
        return jsonify({"error": "Search failed"}), 500

# ==================== OUTREACH ROUTES (Pro only) ====================

@app.route('/outreach/generate-template', methods=['POST'])
@require_auth
@require_plan('pro')
def generate_outreach_template(user):
    """Genera template de outreach personalizado"""
    try:
        data = request.json
        target_id = data.get('target_id')
        target_type = data.get('target_type')  # 'investor' or 'employee'
        platform = data.get('platform', 'email')
        instructions = data.get('instructions', '')
        
        if not target_id or not target_type:
            return jsonify({"error": "Target ID and type required"}), 400
        
        # Verificar créditos
        user_credits = get_user_credits(user['id'])
        cost = CREDIT_COSTS['template_generation']
        
        if user_credits < cost:
            return jsonify({
                "error": "insufficient_credits",
                "required": cost,
                "available": user_credits
            }), 402
        
        # Generar template (implementar lógica)
        template_result = {
            "template_id": str(uuid.uuid4()),
            "content": f"Generated template for {target_type} {target_id}",
            "platform": platform,
            "target_type": target_type,
            "credits_charged": cost
        }
        
        # Cobrar créditos
        charge_credits(user['id'], cost)
        
        return jsonify(template_result)
        
    except Exception as e:
        print(f"❌ ERROR generating template: {e}")
        return jsonify({"error": "Template generation failed"}), 500

# ==================== SUBSCRIPTION & PAYMENT ROUTES ====================

@app.route('/subscription/upgrade', methods=['POST'])
@require_auth
def upgrade_subscription(user):
    """Upgrade de plan con Stripe"""
    try:
        data = request.json
        target_plan = data.get('plan')  # 'growth' or 'pro'
        
        if target_plan not in ['growth', 'pro']:
            return jsonify({"error": "Invalid plan"}), 400
        
        plan_info = SUBSCRIPTION_PLANS[target_plan]
        
        # Crear Stripe Checkout Session para suscripción
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': plan_info['stripe_price_id'],
                'quantity': 1,
            }],
            mode='subscription',
            success_url=f"{request.host_url}subscription/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{request.host_url}subscription/cancel",
            metadata={
                'user_id': user['id'],
                'plan': target_plan
            }
        )
        
        return jsonify({
            "checkout_url": checkout_session.url,
            "session_id": checkout_session.id
        })
        
    except Exception as e:
        print(f"❌ ERROR creating subscription: {e}")
        return jsonify({"error": "Error creating subscription"}), 500

@app.route('/webhook/stripe', methods=['POST'])
def stripe_webhook():
    """Webhook de Stripe para events"""
    try:
        payload = request.get_data(as_text=True)
        sig_header = request.headers.get('Stripe-Signature')
        
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
        
        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']
            
            if session['mode'] == 'payment':
                # Compra de créditos
                handle_credit_purchase(session)
            elif session['mode'] == 'subscription':
                # Suscripción nueva
                handle_subscription_created(session)
        
        elif event['type'] == 'invoice.payment_succeeded':
            # Renovación de suscripción
            handle_subscription_renewal(event['data']['object'])
        
        return jsonify({"status": "success"})
        
    except Exception as e:
        print(f"❌ ERROR in webhook: {e}")
        return jsonify({"error": "Webhook failed"}), 400

def handle_credit_purchase(session):
    """Procesa compra de créditos"""
    try:
        user_id = session['metadata']['user_id']
        credit_amount = int(session['metadata']['credit_amount'])
        
        # Añadir créditos al usuario
        add_credits(user_id, credit_amount, 'purchase')
        
        print(f"✅ Added {credit_amount} credits to user {user_id}")
        
    except Exception as e:
        print(f"❌ ERROR handling credit purchase: {e}")

def handle_subscription_created(session):
    """Procesa nueva suscripción"""
    try:
        user_id = session['metadata']['user_id']
        plan = session['metadata']['plan']
        subscription_id = session['subscription']
        
        # Actualizar plan del usuario
        update_user_plan(user_id, plan, subscription_id)
        
        # Añadir créditos de lanzamiento
        launch_credits = SUBSCRIPTION_PLANS[plan]['launch_credits']
        add_credits(user_id, launch_credits, 'launch_bonus')
        
        print(f"✅ User {user_id} upgraded to {plan} with {launch_credits} bonus credits")
        
    except Exception as e:
        print(f"❌ ERROR handling subscription creation: {e}")

def update_user_plan(user_id, plan, subscription_id):
    """Actualiza plan del usuario"""
    try:
        # Actualizar usuario
        query = """
        UPDATE users 
        SET plan = %s, updated_at = %s
        WHERE id = %s
        """
        
        with engine.connect() as conn:
            conn.execute(text(query), (plan, datetime.now(), user_id))
            conn.commit()
        
        # Crear/actualizar suscripción
        sub_query = """
        INSERT INTO subscriptions (id, user_id, plan, stripe_subscription_id, status, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (user_id) DO UPDATE SET
        plan = EXCLUDED.plan,
        stripe_subscription_id = EXCLUDED.stripe_subscription_id,
        status = EXCLUDED.status,
        updated_at = EXCLUDED.updated_at
        """
        
        params = (
            str(uuid.uuid4()), user_id, plan, subscription_id,
            'active', datetime.now(), datetime.now()
        )
        
        with engine.connect() as conn:
            conn.execute(text(sub_query), params)
            conn.commit()
            
    except Exception as e:
        print(f"❌ ERROR updating user plan: {e}")

# ==================== ADMIN & ANALYTICS ROUTES ====================

@app.route('/admin/stats', methods=['GET'])
@require_auth
def get_admin_stats(user):
    """Estadísticas de admin (solo para admins)"""
    # Implementar verificación de admin
    return jsonify({"message": "Admin stats coming soon"})

# ==============================================================================
#           EXECUTION
# ==============================================================================

if __name__ == '__main__':
    print("Starting 0Bullshit Backend v2.0...")
    print("🚀 Features Ready:")
    print("  ✅ 60 Bot Army")
    print("  ✅ Credits System") 
    print("  ✅ Neural Memory")
    print("  ✅ 3-Tier Plans")
    print("  ✅ Stripe Integration")
    print("  ✅ ML Powered Search")
    print("  ✅ Authentication")
    
    app.run(host='0.0.0.0', port=8080, debug=True)
