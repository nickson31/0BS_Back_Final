# -*- coding: utf-8 -*-

"""
Enhanced Flask Backend API for 0Bullshit - Startup Investment Co-Pilot SaaS
Implements structured, proactive conversational AI with guided user experience
"""

# ==============================================================================
#           IMPORTS
# ==============================================================================

print("1. Loading libraries...")
from flask import Flask, request, jsonify
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
from datetime import datetime
import uuid
from sqlalchemy import text
import jwt
from functools import wraps

# ==============================================================================
#           CONFIGURATION
# ==============================================================================

print("2. Configuring application...")
app = Flask(__name__)
CORS(app)
warnings.filterwarnings('ignore')

# Environment Variables
API_KEY = os.environ.get("GEMINI_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")
JWT_SECRET = os.environ.get("JWT_SECRET", "your-secret-key")
SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET")

if not API_KEY:
    print("❌ FATAL: GEMINI_API_KEY not found.")
if not DATABASE_URL:
    print("❌ FATAL: DATABASE_URL not found.")

# Configure Gemini
try:
    genai.configure(api_key=API_KEY)
    MODEL_NAME = "gemini-2.0-flash"
    print("✅ Gemini API configured.")
except Exception as e:
    print(f"❌ ERROR configuring Gemini: {e}")
    exit()

# Connect to Supabase
try:
    engine = sqlalchemy.create_engine(DATABASE_URL)
    print("✅ Supabase connection established.")
except Exception as e:
    print(f"❌ ERROR connecting to Supabase: {e}")
    engine = None

# Context file paths
UBICACIONES_TXT_PATH = 'ubicaciones.txt'
ETAPAS_TXT_PATH = 'etapas.txt'
CATEGORIAS_TXT_PATH = 'categorias.txt'

# ==============================================================================
#           AUTHENTICATION MIDDLEWARE
# ==============================================================================

def verify_supabase_token(token):
    """Verify Supabase JWT token and return user data."""
    try:
        if not SUPABASE_JWT_SECRET:
            print("⚠️ WARNING: SUPABASE_JWT_SECRET not configured")
            return None
        
        # Decode JWT token
        payload = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=['HS256'])
        user_id = payload.get('sub')
        
        if not user_id:
            return None
        
        # Get user profile from database
        user_query = "SELECT * FROM profiles WHERE id = %s"
        user_df = pd.read_sql(user_query, engine, params=(user_id,))
        
        if user_df.empty:
            return None
        
        return user_df.iloc[0].to_dict()
        
    except jwt.ExpiredSignatureError:
        print("❌ Token expired")
        return None
    except jwt.InvalidTokenError:
        print("❌ Invalid token")
        return None
    except Exception as e:
        print(f"❌ Error verifying token: {e}")
        return None

def require_auth(f):
    """Decorator to require authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'No authorization token provided'}), 401
        
        token = auth_header.split(' ')[1]
        user_data = verify_supabase_token(token)
        
        if not user_data:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        # Add user data to request
        request.user = user_data
        return f(*args, **kwargs)
    
    return decorated_function

def get_user_project(user_id):
    """Get or check if user has a project."""
    try:
        project_query = "SELECT * FROM projects WHERE user_id = %s ORDER BY created_at DESC LIMIT 1"
        project_df = pd.read_sql(project_query, engine, params=(user_id,))
        
        if project_df.empty:
            return None
        
        return project_df.iloc[0].to_dict()
        
    except Exception as e:
        print(f"❌ ERROR getting user project: {e}")
        return None

# ==============================================================================
#           UTILITY FUNCTIONS
# ==============================================================================

print("3. Defining utility functions...")

def load_context_file(filepath):
    """Load content from a text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"  ❌ ERROR loading {filepath}: {e}")
        return ""

# Load context files
ubicaciones_context = load_context_file(UBICACIONES_TXT_PATH)
etapas_context = load_context_file(ETAPAS_TXT_PATH)
categorias_context = load_context_file(CATEGORIAS_TXT_PATH)
print("✅ Contexts loaded.")

def parse_keyword_list(value):
    """Convert a string (or NaN) to a list of keywords."""
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

def get_project_comprehensive_data(project_id):
    """Retrieve comprehensive project data including user plan and saved investors count."""
    try:
        # Get project data with user plan
        project_query = """
        SELECT p.id, p.user_id, p.project_name, p.project_description, 
               p.kpi_data, p.status, pr.plan
        FROM projects p
        JOIN profiles pr ON p.user_id = pr.id
        WHERE p.id = %s
        """
        # ✅ FIXED: Use tuple for params
        project_result = pd.read_sql(project_query, engine, params=(project_id,))
        
        if project_result.empty:
            return None, [], 0, "Free"
        
        project_data = project_result.iloc[0].to_dict()
        
        # Get conversation history
        conv_query = "SELECT history FROM project_conversations WHERE project_id = %s"
        # ✅ FIXED: Use tuple for params
        conv_result = pd.read_sql(conv_query, engine, params=(project_id,))
        chat_history = conv_result.iloc[0]['history'] if not conv_result.empty else []
        
        # Count saved investors with positive sentiment
        saved_count_query = """
        SELECT COUNT(*) as count 
        FROM project_saved_investors psi
        JOIN project_sentiments ps ON psi.investor_id = ps.entity_id 
        WHERE psi.project_id = %s AND ps.sentiment = 'like'
        """
        # ✅ FIXED: Use tuple for params
        saved_count_result = pd.read_sql(saved_count_query, engine, params=(project_id,))
        saved_investors_count = saved_count_result.iloc[0]['count'] if not saved_count_result.empty else 0
        
        return project_data, chat_history, saved_investors_count, project_data.get('plan', 'Free')
        
    except Exception as e:
        print(f"❌ ERROR retrieving comprehensive project data: {e}")
        return None, [], 0, "Free"

def save_conversation_message(project_id, user_message, bot_response):
    """Save conversation message to database."""
    try:
        # Get current conversation history
        _, chat_history, _, _ = get_project_comprehensive_data(project_id)
        
        # Ensure bot_response is string
        if not isinstance(bot_response, str):
            try:
                bot_response = json.dumps(bot_response)
            except (TypeError, ValueError):
                bot_response = str(bot_response)
        
        # Add new messages
        new_messages = [
            {"role": "user", "content": user_message, "timestamp": datetime.now().isoformat()},
            {"role": "assistant", "content": bot_response, "timestamp": datetime.now().isoformat()}
        ]
        
        if isinstance(chat_history, list):
            chat_history.extend(new_messages)
        else:
            chat_history = new_messages
        
        # Keep only last 20 messages
        chat_history = chat_history[-20:]
        
        # ✅ FIXED: Use positional parameters to avoid double % issues
        update_query = """
        INSERT INTO project_conversations (id, project_id, user_id, history, created_at, updated_at)
        VALUES (%s, %s, (SELECT user_id FROM projects WHERE id = %s), %s, %s, %s)
        ON CONFLICT (project_id)
        DO UPDATE SET history = EXCLUDED.history, updated_at = EXCLUDED.updated_at
        """
        
        params = (
            str(uuid.uuid4()),
            project_id,
            project_id,  # for the SELECT
            json.dumps(chat_history),
            datetime.now(),
            datetime.now()
        )
        
        with engine.connect() as conn:
            conn.execute(text(update_query), params)
            conn.commit()
            
    except Exception as e:
        print(f"❌ ERROR saving conversation: {e}")

def update_project_status(project_id, new_status):
    """Update project status in database."""
    try:
        update_query = "UPDATE projects SET status = %s WHERE id = %s"
        params = (new_status, project_id)
        
        with engine.connect() as conn:
            conn.execute(text(update_query), params)
            conn.commit()
    except Exception as e:
        print(f"❌ ERROR updating project status: {e}")

def update_project_kpi_data(project_id, kpi_data):
    """Update project KPI data in database."""
    try:
        update_query = "UPDATE projects SET kpi_data = %s WHERE id = %s"
        params = (json.dumps(kpi_data), project_id)
        
        with engine.connect() as conn:
            conn.execute(text(update_query), params)
            conn.commit()
    except Exception as e:
        print(f"❌ ERROR updating project KPI data: {e}")

def create_user_project(user_id, project_name):
    """Create a new project for the user."""
    try:
        project_id = str(uuid.uuid4())
        insert_query = """
        INSERT INTO projects (id, user_id, project_name, project_description, status, kpi_data, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        params = (
            project_id,
            user_id,
            project_name,
            "",  # empty description initially
            "ONBOARDING",
            json.dumps({}),  # empty KPI data
            datetime.now(),
            datetime.now()
        )
        
        with engine.connect() as conn:
            conn.execute(text(insert_query), params)
            conn.commit()
        
        return project_id
        
    except Exception as e:
        print(f"❌ ERROR creating project: {e}")
        return None

# ==============================================================================
#           SEARCH FUNCTIONS (LEGACY PRESERVED)
# ==============================================================================

def get_keywords_from_gemini_v2(query, u_ctx, e_ctx, c_ctx, model_name):
    """Get advanced keywords (primary and expanded) using Gemini."""
    print("  -> Calling Gemini (Advanced)...")
    start_time = time.time()
    response_text = "N/A"
    try:
        model = genai.GenerativeModel(model_name)
        prompt = f"""**Task:** Analyze the query and extract/expand keywords (Location, Stage, Categories).

**Query:** "{query}"

**Instructions:**
1. **Analyze** the query to identify Locations, Stages and Categories.
2. **Standardize & Infer:** Use contexts to standardize and infer primary keywords.
3. **Expand Keywords:** Generate 5-10 'expanded' keywords *highly relevant* for each 'primary'.
4. **Exact JSON Format:** {{"ubicacion": {{"primary": [...], "expanded": [...]}}, "etapa": {{"primary": [...], "expanded": [...]}}, "categoria": {{"primary": [...], "expanded": [...]}}}}

**Location Context (Ex.):**\n{u_ctx[:1000]}...

**Stage Context (Ex.):**\n{e_ctx[:1000]}...

**Category Context (Ex.):**\n{c_ctx[:1000]}...

**Required JSON Output:**\n```json\n{{ ... }}\n```"""

        response = model.generate_content(prompt)
        response_text = response.text

        json_match = re.search(r'```json\s*([\s\S]+?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_match_alt = re.search(r'({\s*"ubicacion":[\s\S]*})', response_text)
            if json_match_alt:
                json_str = json_match_alt.group(1).strip()
            else:
                print(f"  ⚠️ WARNING: No clear JSON found. Attempting to parse: {response_text}")
                json_str = response_text.strip()

        keywords = json.loads(json_str)
        final_keywords = {"ubicacion": {"primary": [], "expanded": []},
                          "etapa": {"primary": [], "expanded": []},
                          "categoria": {"primary": [], "expanded": []}}
        for cat in final_keywords.keys():
            if cat in keywords and isinstance(keywords[cat], dict):
                final_keywords[cat]["primary"] = [str(k).strip().lower() for k in keywords[cat].get("primary", [])]
                final_keywords[cat]["expanded"] = [str(k).strip().lower() for k in keywords[cat].get("expanded", [])]
        print(f"  ✅ Advanced keywords ({time.time() - start_time:.2f}s): {final_keywords}")
        return final_keywords
    except Exception as e:
        print(f"  ❌ ERROR with Gemini (Advanced): {e}\nResponse: {response_text}")
        return None

def calculate_match_score_v2(row, query_kws):
    """Calculate advanced score (normalized) and matching keywords."""
    investor_u = set(row['Ubicacion_List'])
    investor_e = set(row['Etapa_List'])
    investor_c = set(row['Categoria_List'])

    query_u_p = set(query_kws.get('ubicacion', {}).get('primary', []))
    query_u_e = set(query_kws.get('ubicacion', {}).get('expanded', []))
    query_e_p = set(query_kws.get('etapa', {}).get('primary', []))
    query_e_e = set(query_kws.get('etapa', {}).get('expanded', []))
    query_c_p = set(query_kws.get('categoria', {}).get('primary', []))
    query_c_e = set(query_kws.get('categoria', {}).get('expanded', []))

    matched_u_p = query_u_p.intersection(investor_u)
    matched_u_e = query_u_e.intersection(investor_u) - matched_u_p
    matched_e_p = query_e_p.intersection(investor_e)
    matched_e_e = query_e_e.intersection(investor_e) - matched_e_p
    matched_c_p = query_c_p.intersection(investor_c)
    matched_c_e = query_c_e.intersection(investor_c) - matched_c_p

    matched_u = list(matched_u_p.union(matched_u_e))
    matched_e = list(matched_e_p.union(matched_e_e))
    matched_c = list(matched_c_p.union(matched_c_e))

    WEIGHTS = {'ubicacion': 0.15, 'etapa': 0.45, 'categoria': 0.40}
    P_PRIMARY = 3.0
    P_EXPANDED = 1.0
    BONUS_PRIMARY_MATCH = 0.5

    score_u = (len(matched_u_p) * P_PRIMARY) + (len(matched_u_e) * P_EXPANDED)
    score_e = (len(matched_e_p) * P_PRIMARY) + (len(matched_e_e) * P_EXPANDED)
    score_c = (len(matched_c_p) * P_PRIMARY) + (len(matched_c_e) * P_EXPANDED)

    raw_score = (score_u * WEIGHTS['ubicacion'] +
                 score_e * WEIGHTS['etapa'] +
                 score_c * WEIGHTS['categoria'])
    raw_score += (len(matched_e_p) * BONUS_PRIMARY_MATCH) + (len(matched_c_p) * BONUS_PRIMARY_MATCH)

    max_u = (len(query_u_p) * P_PRIMARY) + (len(query_u_e) * P_EXPANDED)
    max_e = (len(query_e_p) * P_PRIMARY) + (len(query_e_e) * P_EXPANDED)
    max_c = (len(query_c_p) * P_PRIMARY) + (len(query_c_e) * P_EXPANDED)
    max_score = (max_u * WEIGHTS['ubicacion'] +
                 max_e * WEIGHTS['etapa'] +
                 max_c * WEIGHTS['categoria'])
    max_score += (len(query_e_p) * BONUS_PRIMARY_MATCH) + (len(query_c_p) * BONUS_PRIMARY_MATCH)

    normalized_score = (raw_score / max_score * 100) if max_score > 0 else 0.0

    return raw_score, normalized_score, matched_u, matched_e, matched_c

def run_investor_search(project_data):
    """Execute deep investor search based on project data."""
    if not engine:
        return {"error": "No database connection."}

    print("-> Starting deep investor search...")
    try:
        sql_query = """
        SELECT id, "Company_Name", "Company_Description", "Investing_Stage",
               "Investing_Type", "Company_Location", "Investment_Categories",
               "Company_Email", "Company_Phone", "Company_Linkedin", "Company_Website",
               "Keywords_Ubicacion_Adicionales", "Keywords_Etapas_Adicionales", 
               "Keywords_Categorias_Adicionales" 
        FROM investors
        """
        investors_df = pd.read_sql(sql_query, engine)
        print(f"  -> {len(investors_df)} investors loaded.")

        investors_df['Ubicacion_List'] = investors_df['Keywords_Ubicacion_Adicionales'].apply(parse_keyword_list)
        investors_df['Etapa_List'] = investors_df['Keywords_Etapas_Adicionales'].apply(parse_keyword_list)
        investors_df['Categoria_List'] = investors_df['Keywords_Categorias_Adicionales'].apply(parse_keyword_list)

        # Build intelligent query from project data
        query_parts = []
        if project_data.get('project_description'):
            query_parts.append(project_data['project_description'])
        
        kpi_data = project_data.get('kpi_data', {})
        if isinstance(kpi_data, str):
            kpi_data = json.loads(kpi_data)
        
        # Add relevant KPI context
        if kpi_data.get('ingresos_mensuales'):
            query_parts.append(f"Ingresos mensuales: {kpi_data['ingresos_mensuales']}")
        if kpi_data.get('runway'):
            query_parts.append(f"Runway: {kpi_data['runway']} meses")
        
        search_query = " ".join(query_parts)
        
        query_keywords = get_keywords_from_gemini_v2(
            search_query, ubicaciones_context, etapas_context, categorias_context, MODEL_NAME
        )

        if not query_keywords:
            return {"error": "Could not obtain keywords."}

        print("  -> Applying scoring...")
        
        score_results = investors_df.apply(lambda row: calculate_match_score_v2(row, query_keywords), axis=1, result_type='expand')
        score_results.columns = ['Score_Raw', 'Score', 'Matched_Ubicacion', 'Matched_Etapa', 'Matched_Categoria']
        investors_df = investors_df.join(score_results)
        results_df = investors_df[investors_df['Score'] > 0].sort_values(by='Score', ascending=False).head(50)

        # Columns to return to frontend
        cols_to_show = [
            'id', 'Company_Name', 'Company_Description', 'Investing_Stage',
            'Company_Location', 'Investment_Categories',
            'Company_Email', 'Company_Phone', 'Company_Linkedin', 'Company_Website',
            'Score'
        ]

        results_df['Score'] = results_df['Score'].map('{:,.1f}'.format)
        results_df = results_df.fillna('-')
        print(f"  ✅ Deep search completed, {len(results_df)} results.")
        
        return results_df[[c for c in cols_to_show if c in results_df.columns]].to_dict('records')

    except Exception as e:
        print(f"  ❌ ERROR in investor search: {e}")
        return {"error": f"Error in search: {e}"}

# ==============================================================================
#           0BULLSHIT MASTER PROMPT AND AI ORCHESTRATOR
# ==============================================================================

def get_zero_bullshit_master_prompt():
    """Return the 0Bullshit master prompt for conversational AI orchestration."""
    return """## ROL Y PERSONALIDAD

Eres "0Bullshit", un co-piloto de IA experto en fundraising para fundadores de startups y pymes. Tu misión es democratizar el acceso a capital y know-how, eliminando la "bullshit" (paja, rodeos, información inútil) del proceso. 

Tu tono es:
- **Directo y Rápido:** Vas al grano para ahorrar tiempo. Usas frases como "¡Genial!", "Anotado.", "Perfecto." 
- **Proactivo y Guía:** No esperas a que el usuario te pida las cosas. Tú inicias el proceso de recolección de datos y sugieres los siguientes pasos. 
- **Orientado a Valor:** Justificas tus recomendaciones con datos (ej. "elevan 90% su probabilidad de cerrar ronda"). 

## CONTEXTO DE LA CONVERSACIÓN

1. **Datos del Proyecto:** {project_data}
2. **Historial del Chat:** {chat_history}
3. **Plan del Usuario:** {user_plan}
4. **Inversores Guardados (Liked):** {saved_investors_count}
5. **Mensaje del Usuario:** {user_message}

## FLUJO DE TRABAJO ESTRUCTURADO Y REGLAS

Tu comportamiento principal depende del `status` del proyecto:

**A. SI `status` es 'ONBOARDING':**
Tu objetivo es rellenar los `kpi_data`. Revisa los datos del proyecto y haz la siguiente pregunta lógica en orden:
1. Si `project_description` está vacío: Pregunta por el problema que resuelven.
2. Si falta el "tuit": Pide la propuesta en <=140 caracteres.
3. Si faltan `ingresos_mensuales`, `margen_bruto` o `runway`: Pídelos. 
4. Si faltan `CAC`, `repeat_rate`, etc.: Pídelos. 
5. **REGLA DE UPSELL (GROWTH):** Antes de pedir los últimos KPIs (CAC, etc.), si el plan del usuario es 'Free', DEBES usar la herramienta `trigger_growth_upsell`. No hagas la pregunta. 

**B. SI `status` es 'SEARCH_READY':**
1. Si el usuario da una señal para buscar (ej. "listo", "busca ahora", "ok, sigamos"), tu DEBER es usar la herramienta `run_investor_search`. 
2. Si el usuario indica "Like" o "Dislike" sobre un inversor, usa `set_entity_sentiment`.
3. **REGLA DE UPSELL (OUTREACH):** Si el `saved_investors_count` llega a 3 y el plan del usuario no es 'Outreach', DEBES usar la herramienta `trigger_outreach_upsell`. 

**C. SI `status` es 'OUTREACH_READY':**
1. Si el usuario pide un ejemplo de mensaje o quiere empezar a contactar, usa `generate_outreach_example`. 
2. Si el usuario dice "adelante" o "activa outreach", usa `confirm_outreach_activation`.

## HERRAMIENTAS Y FORMATO DE SALIDA

Tu respuesta debe ser SIEMPRE un único objeto JSON, sin texto adicional.

**FORMATO OBLIGATORIO:**
{{
  "action": "nombre_de_la_herramienta",
  "parameters": {{ ... }}
}}

**LISTA DE HERRAMIENTAS:**
- `ask_onboarding_question(question_text: str)`: Para hacer la siguiente pregunta del proceso de onboarding.
- `trigger_growth_upsell()`: Muestra el mensaje de venta del plan Growth.
- `confirm_plan_activation(plan_name: str)`: Confirma que un plan ha sido activado.
- `run_investor_search()`: Inicia la búsqueda profunda de inversores.
- `set_entity_sentiment(entity_id: str, sentiment: str)`: Para guardar un "like" o "dislike".
- `trigger_outreach_upsell()`: Muestra el mensaje de venta del plan Outreach.
- `generate_outreach_example(target_investor_id: str)`: Genera un borrador de mensaje.
- `answer_general_chat(response_text: str)`: Para conversación que no encaje en el flujo principal.

**EJEMPLO DE RAZONAMIENTO:**
- **Usuario:** "Reducimos la factura eléctrica de pequeños comercios..." 
- **Mi estado mental:** OK, `status` es `ONBOARDING`. `project_description` está relleno. Lo siguiente es el "tuit".
- **Mi salida JSON:**
  ```json
  {{
    "action": "ask_onboarding_question",
    "parameters": {{
      "question_text": "Perfecto: 'Ahorro energético para comercio minorista'. Ahora condensa tu propuesta en <=140 caracteres (un tuit rápido)."
    }}
  }}
  ```

**INSTRUCCIÓN FINAL:**
Analiza el contexto y devuelve el JSON de la acción apropiada. No agregues texto antes o después del JSON."""

def process_zero_bullshit_request(project_id, user_message):
    """Process request using 0Bullshit orchestrator."""
    try:
        # Get comprehensive project context
        project_data, chat_history, saved_investors_count, user_plan = get_project_comprehensive_data(project_id)
        
        if not project_data:
            return {
                "action": "answer_general_chat",
                "parameters": {
                    "response_text": "Lo siento, no pude encontrar tu proyecto. Por favor, verifica el ID del proyecto."
                }
            }
        
        # Prepare the master prompt with context
        master_prompt = get_zero_bullshit_master_prompt().format(
            project_data=json.dumps(project_data, ensure_ascii=False),
            chat_history=json.dumps(chat_history[-10:] if chat_history else [], ensure_ascii=False),
            user_plan=user_plan,
            saved_investors_count=saved_investors_count,
            user_message=user_message
        )
        
        # Call Gemini
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(master_prompt)
        response_text = response.text.strip()
        
        # Extract JSON from response
        json_match = re.search(r'```json\s*([\s\S]+?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try to find JSON without code blocks
            json_match = re.search(r'(\{[\s\S]*\})', response_text)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                json_str = response_text
        
        # Parse JSON
        action_data = json.loads(json_str)
        return action_data
        
    except Exception as e:
        print(f"❌ ERROR processing 0Bullshit request: {e}")
        return {
            "action": "answer_general_chat",
            "parameters": {
                "response_text": "Disculpa, tuve un error procesando tu solicitud. Inténtalo de nuevo o reformula tu pregunta."
            }
        }

# ==============================================================================
#           ACTION EXECUTION FUNCTIONS
# ==============================================================================

def execute_zero_bullshit_action(action_data, project_id):
    """Execute the determined action for 0Bullshit workflow."""
    action = action_data.get("action")
    parameters = action_data.get("parameters", {})
    
    if action == "ask_onboarding_question":
        question_text = parameters.get("question_text", "¿Puedes contarme más sobre tu proyecto?")
        return {
            "type": "onboarding_question",
            "content": question_text
        }
    
    elif action == "trigger_growth_upsell":
        upsell_message = """
🚀 **¡Momento de crecer!** 

Para acceder a búsquedas avanzadas y métricas detalladas, necesitas el plan **Growth**:

✅ Búsquedas ilimitadas de inversores
✅ Análisis de compatibilidad avanzado  
✅ Exportar resultados a Excel
✅ Soporte prioritario

**$29/mes** - Cancela cuando quieras

¿Activamos Growth para desbloquear todo el potencial de tu búsqueda?
"""
        return {
            "type": "plan_upsell",
            "content": upsell_message,
            "plan": "Growth",
            "price": "$29/mes"
        }
    
    elif action == "confirm_plan_activation":
        plan_name = parameters.get("plan_name", "Growth")
        return {
            "type": "plan_confirmed",
            "content": f"¡Perfecto! Plan {plan_name} activado. Ahora tienes acceso completo. ¡Sigamos!"
        }
    
    elif action == "run_investor_search":
        # Get project data for search
        project_data, _, _, _ = get_project_comprehensive_data(project_id)
        if not project_data:
            return {
                "type": "error",
                "content": "No se pudo acceder a los datos del proyecto."
            }
        
        search_results = run_investor_search(project_data)
        
        if isinstance(search_results, dict) and "error" in search_results:
            return {
                "type": "error",
                "content": search_results["error"]
            }
        
        # Update project status to SEARCH_READY if not already
        if project_data.get('status') == 'ONBOARDING':
            update_project_status(project_id, 'SEARCH_READY')
        
        return {
            "type": "investor_results",
            "content": search_results,
            "message": f"¡Encontré {len(search_results)} inversores potenciales! Revisa los resultados y marca como 'Like' los que más te interesen."
        }
    
    elif action == "set_entity_sentiment":
        entity_id = parameters.get("entity_id")
        sentiment = parameters.get("sentiment", "like").lower()
        
        if not entity_id:
            return {
                "type": "error",
                "content": "Se requiere el ID de la entidad."
            }
        
        try:
            # Add to saved investors if liked
            if sentiment == "like":
                save_query = """
                INSERT INTO project_saved_investors (project_id, investor_id, added_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (project_id, investor_id) DO NOTHING
                """
                params = (project_id, entity_id, datetime.now())
                
                with engine.connect() as conn:
                    conn.execute(text(save_query), params)
                    conn.commit()
            
            # Add sentiment
            sentiment_query = """
            INSERT INTO project_sentiments (id, project_id, user_id, entity_id, entity_type, sentiment, created_at)
            VALUES (%s, %s, (SELECT user_id FROM projects WHERE id = %s), %s, 'investor', %s, %s)
            ON CONFLICT (project_id, entity_id, entity_type) 
            DO UPDATE SET sentiment = EXCLUDED.sentiment
            """
            
            params = (
                str(uuid.uuid4()),
                project_id,
                project_id,
                entity_id,
                sentiment,
                datetime.now()
            )
            
            with engine.connect() as conn:
                conn.execute(text(sentiment_query), params)
                conn.commit()
            
            return {
                "type": "sentiment_saved",
                "content": f"¡Anotado! Inversor marcado como '{sentiment}'"
            }
            
        except Exception as e:
            print(f"❌ ERROR saving sentiment: {e}")
            return {
                "type": "error",
                "content": "No se pudo guardar la preferencia."
            }
    
    elif action == "trigger_outreach_upsell":
        upsell_message = """
💼 **¡Es hora del Outreach!**

Has guardado varios inversores interesantes. Para contactarlos de forma profesional, necesitas el plan **Outreach**:

✅ Plantillas de email personalizadas
✅ Mensajes de LinkedIn optimizados
✅ Seguimiento automático
✅ Analytics de respuestas
✅ CRM integrado

**$99/mes** - ROI garantizado

¿Activamos Outreach para empezar a contactar inversores de forma efectiva?
"""
        return {
            "type": "plan_upsell",
            "content": upsell_message,
            "plan": "Outreach",
            "price": "$99/mes"
        }
    
    elif action == "generate_outreach_example":
        target_investor_id = parameters.get("target_investor_id")
        
        if not target_investor_id:
            # Get first liked investor
            try:
                investor_query = """
                SELECT i.id, i."Company_Name", i."Company_Description"
                FROM investors i
                JOIN project_saved_investors psi ON i.id = psi.investor_id
                JOIN project_sentiments ps ON i.id = ps.entity_id
                WHERE psi.project_id = %s AND ps.sentiment = 'like'
                ORDER BY psi.added_at ASC
                LIMIT 1
                """
                # ✅ FIXED: Use tuple for params
                result = pd.read_sql(investor_query, engine, params=(project_id,))
                if not result.empty:
                    target_investor_id = result.iloc[0]['id']
                else:
                    return {
                        "type": "error",
                        "content": "No tienes inversores guardados aún. Marca algunos como 'Like' primero."
                    }
            except Exception as e:
                return {
                    "type": "error",
                    "content": "No se pudo acceder a tus inversores guardados."
                }
        
        # Generate outreach template
        try:
            project_data, _, _, _ = get_project_comprehensive_data(project_id)
            result = generate_outreach_template_content(
                target_investor_id, "investor", "email", 
                "Genera un email profesional y personalizado", project_id
            )
            
            if "error" in result:
                return {
                    "type": "error",
                    "content": result["error"]
                }
            
            # Update status to OUTREACH_READY
            update_project_status(project_id, 'OUTREACH_READY')
            
            return {
                "type": "outreach_template",
                "content": result,
                "message": "¡Aquí tienes un borrador personalizado! Puedes editarlo antes de enviarlo."
            }
            
        except Exception as e:
            print(f"❌ ERROR generating outreach: {e}")
            return {
                "type": "error",
                "content": "No se pudo generar el borrador del mensaje."
            }
    
    elif action == "confirm_outreach_activation":
        return {
            "type": "outreach_activated",
            "content": "¡Perfecto! Sistema de outreach activado. Ahora puedes enviar mensajes profesionales a todos tus inversores guardados.",
            "next_steps": [
                "Revisa y personaliza cada mensaje",
                "Programa envíos escalonados",
                "Monitorea respuestas en tiempo real"
            ]
        }
    
    elif action == "answer_general_chat":
        response_text = parameters.get("response_text", "¡Hola! Estoy aquí para ayudarte con tu fundraising.")
        return {
            "type": "text_response",
            "content": response_text
        }
    
    else:
        return {
            "type": "error",
            "content": "Acción no reconocida."
        }

def generate_outreach_template_content(target_entity_id, target_entity_type, platform, user_instructions, project_id):
    """Generate outreach template using Gemini with 0Bullshit style."""
    try:
        # Get project description for context
        project_data, _, _, _ = get_project_comprehensive_data(project_id)
        
        # Get entity information
        if target_entity_type == 'investor':
            entity_query = """
            SELECT "Company_Name", "Company_Description", "Investing_Stage", 
                   "Investment_Categories", "Company_Location"
            FROM investors WHERE id = %s
            """
        else:  # employee
            entity_query = """
            SELECT "fullName", "headline", "Company_Name", 
                   "location", "about", "current_job_title"
            FROM employees WHERE id = %s
            """
        
        # ✅ FIXED: Use tuple for params
        entity_df = pd.read_sql(entity_query, engine, params=(target_entity_id,))
        
        if entity_df.empty:
            return {"error": f"Could not find {target_entity_type} with ID {target_entity_id}"}
        
        entity_info = entity_df.iloc[0].to_dict()
        
        # Prepare context for template generation
        if target_entity_type == 'investor':
            entity_context = f"""
            Fondo: {entity_info.get('Company_Name', 'N/A')}
            Descripción: {entity_info.get('Company_Description', 'N/A')}
            Etapa de Inversión: {entity_info.get('Investing_Stage', 'N/A')}
            Categorías: {entity_info.get('Investment_Categories', 'N/A')}
            Ubicación: {entity_info.get('Company_Location', 'N/A')}
            """
        else:
            entity_context = f"""
            Nombre: {entity_info.get('fullName', 'N/A')}
            Título: {entity_info.get('current_job_title', 'N/A')}
            Empresa: {entity_info.get('Company_Name', 'N/A')}
            Headline: {entity_info.get('headline', 'N/A')}
            Ubicación: {entity_info.get('location', 'N/A')}
            """
        
        # Get project KPIs for context
        kpi_data = project_data.get('kpi_data', {})
        if isinstance(kpi_data, str):
            kpi_data = json.loads(kpi_data)
        
        template_prompt = f"""
**Tarea:** Genera un email profesional y directo (estilo "0Bullshit") para solicitar una reunión de inversión.

**Información del Fondo/Inversor:**
{entity_context}

**Mi Startup:**
Descripción: {project_data.get('project_description', 'N/A')}
Ingresos mensuales: {kpi_data.get('ingresos_mensuales', 'N/A')}
Runway: {kpi_data.get('runway', 'N/A')} meses
Margen bruto: {kpi_data.get('margen_bruto', 'N/A')}%

**Estilo "0Bullshit" - Requisitos:**
1. **Directo y sin rodeos** - Máximo 150 palabras
2. **Asunto claro** - Menciona la startup y el stage
3. **Párrafo 1:** Quién soy y qué hacemos (1-2 líneas)
4. **Párrafo 2:** Métricas clave que demuestren tracción (2-3 líneas)
5. **Párrafo 3:** Por qué este fondo específicamente (1-2 líneas mencionando su portfolio/especialización)
6. **Call-to-action:** Solicitud directa de 15-20 min de reunión
7. **Tono:** Profesional pero humano, confiado sin ser arrogante

**Evitar:**
- Frases genéricas o marketing speak
- Emails largos o con mucho texto
- Adjuntos en el primer contacto

**Formato de salida:**
Asunto: [asunto aquí]

[email body aquí]

Saludos,
[Firma]
"""

        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(template_prompt)
        template_content = response.text.strip()
        
        # Save to database
        template_id = str(uuid.uuid4())
        insert_query = """
        INSERT INTO template_generators (id, project_id, user_id, target_investor_id, platform, conversation_history, generated_template, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        params = (
            template_id,
            project_id,
            project_data['user_id'],
            target_entity_id,
            platform,
            json.dumps([]),
            template_content,
            datetime.now(),
            datetime.now()
        )
        
        with engine.connect() as conn:
            conn.execute(text(insert_query), params)
            conn.commit()
        
        return {
            "template_id": template_id,
            "content": template_content,
            "platform": platform,
            "target_entity_type": target_entity_type,
            "target_info": entity_info
        }
        
    except Exception as e:
        print(f"❌ ERROR generating outreach template: {e}")
        return {"error": "No se pudo generar la plantilla de outreach"}

# ==============================================================================
#           API ROUTES
# ==============================================================================

print("5. Defining API routes...")

@app.route('/')
def home():
    """Main route to verify the API is working."""
    return "<h1>0Bullshit - Enhanced Investor Finder API - OK</h1>"

@app.route('/auth/check', methods=['GET'])
@require_auth
def check_auth():
    """Check if user is authenticated and has a project."""
    try:
        user_id = request.user['id']
        user_plan = request.user.get('plan', 'Free')
        
        # Check if user has a project
        project = get_user_project(user_id)
        
        if not project:
            return jsonify({
                "authenticated": True,
                "has_project": False,
                "user": request.user,
                "message": "No project found. Create one to continue."
            }), 200
        
        return jsonify({
            "authenticated": True,
            "has_project": True,
            "user": request.user,
            "project": project
        }), 200
        
    except Exception as e:
        print(f"❌ ERROR in auth check: {e}")
        return jsonify({"error": "Could not verify authentication"}), 500

@app.route('/projects', methods=['POST'])
@require_auth
def create_project():
    """Create a new project for the authenticated user."""
    try:
        data = request.json
        project_name = data.get('project_name')
        
        if not project_name or len(project_name.strip()) < 3:
            return jsonify({"error": "El nombre del proyecto debe tener al menos 3 caracteres"}), 400
        
        user_id = request.user['id']
        
        # Check if user already has a project (for Free users, limit to 1)
        existing_project = get_user_project(user_id)
        user_plan = request.user.get('plan', 'Free')
        
        if existing_project and user_plan == 'Free':
            return jsonify({
                "error": "Los usuarios Free solo pueden tener un proyecto. Actualiza tu plan para crear más proyectos."
            }), 403
        
        # Create the project
        project_id = create_user_project(user_id, project_name.strip())
        
        if not project_id:
            return jsonify({"error": "No se pudo crear el proyecto"}), 500
        
        # Get the created project data
        project_data, _, _, _ = get_project_comprehensive_data(project_id)
        
        return jsonify({
            "message": "Proyecto creado exitosamente",
            "project": project_data
        }), 201
        
    except Exception as e:
        print(f"❌ ERROR creating project: {e}")
        return jsonify({"error": "No se pudo crear el proyecto"}), 500

@app.route('/chat', methods=['POST'])
@require_auth
def zero_bullshit_chat_endpoint():
    """Main 0Bullshit conversational endpoint."""
    print("\n--- Request /chat (0Bullshit) ---")
    data = request.json
    user_message = data.get('message')
    project_id = data.get('project_id')

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    print(f"  -> Message: '{user_message[:50]}...'")

    try:
        user_id = request.user['id']
        
        # If no project_id provided, get user's current project
        if not project_id:
            project = get_user_project(user_id)
            if not project:
                return jsonify({
                    "type": "need_project",
                    "content": "Para continuar, necesitas crear un proyecto. Ve a la sección 'Crear Proyecto' y dale un nombre a tu startup."
                }), 200
            project_id = project['id']
        
        # Verify project belongs to user
        project_data, _, _, _ = get_project_comprehensive_data(project_id)
        if not project_data or project_data['user_id'] != user_id:
            return jsonify({"error": "Proyecto no encontrado o no tienes permisos"}), 404

        # Process the 0Bullshit conversational request
        action_data = process_zero_bullshit_request(project_id, user_message)
        
        # Execute the determined action
        result = execute_zero_bullshit_action(action_data, project_id)
        
        # Save conversation to database
        save_conversation_message(project_id, user_message, result.get("content", ""))
        
        print("--- Request /chat (0Bullshit) completed ---")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ ERROR in 0Bullshit chat endpoint: {e}")
        return jsonify({
            "type": "error",
            "content": "Hubo un error procesando tu solicitud. Inténtalo de nuevo."
        }), 500

@app.route('/projects/<project_id>/status', methods=['PUT'])
@require_auth
def update_project_status_endpoint(project_id):
    """Update project status."""
    try:
        # Verify project belongs to user
        project_data, _, _, _ = get_project_comprehensive_data(project_id)
        if not project_data or project_data['user_id'] != request.user['id']:
            return jsonify({"error": "Proyecto no encontrado o no tienes permisos"}), 404
        
        data = request.json
        new_status = data.get('status')
        
        if not new_status:
            return jsonify({"error": "status is required"}), 400
        
        update_project_status(project_id, new_status)
        return jsonify({"message": f"Project status updated to {new_status}"})
        
    except Exception as e:
        print(f"❌ ERROR updating project status: {e}")
        return jsonify({"error": "Could not update project status"}), 500

@app.route('/projects/<project_id>/kpi', methods=['PUT'])
@require_auth
def update_project_kpi_endpoint(project_id):
    """Update project KPI data."""
    try:
        # Verify project belongs to user
        project_data, _, _, _ = get_project_comprehensive_data(project_id)
        if not project_data or project_data['user_id'] != request.user['id']:
            return jsonify({"error": "Proyecto no encontrado o no tienes permisos"}), 404
        
        data = request.json
        kpi_data = data.get('kpi_data')
        
        if not kpi_data:
            return jsonify({"error": "kpi_data is required"}), 400
        
        update_project_kpi_data(project_id, kpi_data)
        return jsonify({"message": "KPI data updated successfully"})
        
    except Exception as e:
        print(f"❌ ERROR updating KPI data: {e}")
        return jsonify({"error": "Could not update KPI data"}), 500

@app.route('/projects/<project_id>', methods=['GET'])
@require_auth
def get_project_endpoint(project_id):
    """Get comprehensive project data."""
    try:
        project_data, chat_history, saved_investors_count, user_plan = get_project_comprehensive_data(project_id)
        
        if not project_data:
            return jsonify({"error": "Project not found"}), 404
        
        # Verify project belongs to user
        if project_data['user_id'] != request.user['id']:
            return jsonify({"error": "No tienes permisos para acceder a este proyecto"}), 403
        
        return jsonify({
            "project": project_data,
            "chat_history": chat_history,
            "saved_investors_count": saved_investors_count,
            "user_plan": user_plan
        })
        
    except Exception as e:
        print(f"❌ ERROR getting project data: {e}")
        return jsonify({"error": "Could not retrieve project data"}), 500

@app.route('/projects/<project_id>/saved-investors', methods=['GET'])
@require_auth
def get_saved_investors_endpoint(project_id):
    """Get saved investors with sentiment for a project."""
    try:
        # Verify project belongs to user
        project_data, _, _, _ = get_project_comprehensive_data(project_id)
        if not project_data or project_data['user_id'] != request.user['id']:
            return jsonify({"error": "Proyecto no encontrado o no tienes permisos"}), 404
        
        query = """
        SELECT i.id, i."Company_Name", i."Company_Description", i."Investing_Stage",
               i."Company_Location", i."Investment_Categories", i."Company_Email",
               i."Company_Phone", i."Company_Linkedin", i."Company_Website",
               psi.added_at, ps.sentiment
        FROM project_saved_investors psi
        JOIN investors i ON psi.investor_id = i.id
        LEFT JOIN project_sentiments ps ON i.id = ps.entity_id AND ps.project_id = psi.project_id
        WHERE psi.project_id = %s
        ORDER BY psi.added_at DESC
        """
        
        # ✅ FIXED: Use tuple for params
        results_df = pd.read_sql(query, engine, params=(project_id,))
        saved_investors = results_df.fillna('-').to_dict('records')
        
        return jsonify({"saved_investors": saved_investors})
        
    except Exception as e:
        print(f"❌ ERROR getting saved investors: {e}")
        return jsonify({"error": "Could not retrieve saved investors"}), 500

@app.route('/projects/<project_id>/templates', methods=['GET'])
@require_auth
def get_project_templates_endpoint(project_id):
    """Get generated templates for a project."""
    try:
        # Verify project belongs to user
        project_data, _, _, _ = get_project_comprehensive_data(project_id)
        if not project_data or project_data['user_id'] != request.user['id']:
            return jsonify({"error": "Proyecto no encontrado o no tienes permisos"}), 404
        
        query = """
        SELECT tg.id, tg.target_investor_id, tg.platform, tg.generated_template, 
               tg.created_at, i."Company_Name" as target_name
        FROM template_generators tg
        LEFT JOIN investors i ON tg.target_investor_id = i.id
        WHERE tg.project_id = %s
        ORDER BY tg.created_at DESC
        """
        
        # ✅ FIXED: Use tuple for params
        results_df = pd.read_sql(query, engine, params=(project_id,))
        templates = results_df.fillna('-').to_dict('records')
        
        return jsonify({"templates": templates})
        
    except Exception as e:
        print(f"❌ ERROR getting templates: {e}")
        return jsonify({"error": "Could not retrieve templates"}), 500

# Legacy endpoint for backward compatibility
@app.route('/find_investors', methods=['POST'])
def find_investors_legacy_endpoint():
    """Legacy endpoint for backward compatibility."""
    print("\n--- Request /find_investors (Legacy) ---")
    data = request.json
    user_query = data.get('query')
    is_deep = data.get('deep_research', False)

    if not user_query:
        return jsonify({"error": "Query required"}), 400

    print(f"  -> Query: '{user_query[:50]}...', Deep: {is_deep}.")
    
    # Create mock project data for legacy support
    mock_project_data = {
        'project_description': user_query,
        'kpi_data': {}
    }
    
    results = run_investor_search(mock_project_data)
    print("--- Request /find_investors (Legacy) completed ---")
    return jsonify(results)

# ==============================================================================
#           EXECUTION
# ==============================================================================

if __name__ == '__main__':
    print("Starting Flask server for local testing...")
    print("🚀 0Bullshit Enhanced Backend READY! 🚀")
    app.run(host='0.0.0.0', port=8080, debug=True)
