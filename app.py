# -*- coding: utf-8 -*-

"""
0Bullshit - Backend Completo Sin Autenticación 
Sistema de sesiones temporales con funcionalidad completa
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
from datetime import datetime
import uuid
from sqlalchemy import text
import secrets

# ==============================================================================
#           CONFIGURATION
# ==============================================================================

print("2. Configuring application...")
app = Flask(__name__)
CORS(app, supports_credentials=True)  # Permite sesiones
app.secret_key = secrets.token_hex(16)  # Para sesiones Flask
warnings.filterwarnings('ignore')

# Environment Variables
API_KEY = os.environ.get("GEMINI_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")

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

def get_or_create_session_project():
    """Obtiene o crea un proyecto temporal para la sesión."""
    if 'project_id' not in session:
        # Crear proyecto temporal
        project_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())  # Usuario temporal
        
        try:
            # Crear usuario temporal en profiles
            insert_user_query = """
            INSERT INTO profiles (id, plan, created_at, updated_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """
            user_params = (user_id, 'Free', datetime.now(), datetime.now())
            
            # Crear proyecto temporal
            insert_project_query = """
            INSERT INTO projects (id, user_id, project_name, project_description, status, kpi_data, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """
            project_params = (
                project_id, user_id, "Sesión Temporal", "", "ACTIVE", 
                json.dumps({}), datetime.now(), datetime.now()
            )
            
            with engine.connect() as conn:
                conn.execute(text(insert_user_query), tuple(user_params))
                conn.execute(text(insert_project_query), tuple(project_params))
                conn.commit()
            
            session['project_id'] = project_id
            session['user_id'] = user_id
            print(f"✅ Created temporary session: project_id={project_id}")
            
        except Exception as e:
            print(f"❌ ERROR creating session project: {e}")
            # Fallback a IDs temporales sin BD
            session['project_id'] = project_id
            session['user_id'] = user_id
    
    return session['project_id'], session['user_id']

# ==============================================================================
#           ALGORITMOS DE MATCHING DE INVERSORES
# ==============================================================================

def get_keywords_from_gemini_v2(query, u_ctx, e_ctx, c_ctx, model_name):
    """Obtiene keywords avanzadas usando Gemini para matching de inversores."""
    print("  -> Calling Gemini for Investor Keywords...")
    start_time = time.time()
    response_text = "N/A"
    try:
        model = genai.GenerativeModel(model_name)
        prompt = f"""**Task:** Analiza la consulta y extrae keywords para búsqueda de inversores (Ubicación, Etapa, Categorías).

**Query:** "{query}"

**Instructions:**
1. **Analiza** la consulta para identificar Ubicaciones, Etapas y Categorías de inversión.
2. **Estandariza e Infiere:** Usa los contextos para estandarizar e inferir keywords primarias.
3. **Expande Keywords:** Genera 5-10 keywords 'expandidas' altamente relevantes para cada 'primaria'.
4. **Formato JSON Exacto:** {{"ubicacion": {{"primary": [...], "expanded": [...]}}, "etapa": {{"primary": [...], "expanded": [...]}}, "categoria": {{"primary": [...], "expanded": [...]}}}}

**Location Context:**\n{u_ctx[:1000]}...

**Stage Context:**\n{e_ctx[:1000]}...

**Category Context:**\n{c_ctx[:1000]}...

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
        print(f"  ✅ Investor keywords ({time.time() - start_time:.2f}s): {final_keywords}")
        return final_keywords
    except Exception as e:
        print(f"  ❌ ERROR with Gemini (Investor Keywords): {e}\nResponse: {response_text}")
        return None

def calculate_match_score_v2(row, query_kws):
    """Calculate advanced score (normalized) and matching keywords for investors."""
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

def run_investor_search_normal(query):
    """Ejecuta búsqueda NORMAL de inversores (50 mejores resultados)."""
    if not engine:
        return {"error": "No database connection."}

    print("-> Starting NORMAL investor search...")
    try:
        sql_query = """
        SELECT id, "Company_Name", "Company_Description", "Investing_Stage",
               "Company_Location", "Investment_Categories", "Company_Linkedin",
               "Keywords_Ubicacion_Adicionales", "Keywords_Etapas_Adicionales", 
               "Keywords_Categorias_Adicionales" 
        FROM investors
        """
        investors_df = pd.read_sql(sql_query, engine)
        print(f"  -> {len(investors_df)} investors loaded.")

        investors_df['Ubicacion_List'] = investors_df['Keywords_Ubicacion_Adicionales'].apply(parse_keyword_list)
        investors_df['Etapa_List'] = investors_df['Keywords_Etapas_Adicionales'].apply(parse_keyword_list)
        investors_df['Categoria_List'] = investors_df['Keywords_Categorias_Adicionales'].apply(parse_keyword_list)

        query_keywords = get_keywords_from_gemini_v2(
            query, ubicaciones_context, etapas_context, categorias_context, MODEL_NAME
        )

        if not query_keywords:
            return {"error": "Could not obtain keywords."}

        print("  -> Applying NORMAL scoring...")
        
        score_results = investors_df.apply(lambda row: calculate_match_score_v2(row, query_keywords), axis=1, result_type='expand')
        score_results.columns = ['Score_Raw', 'Score', 'Matched_Ubicacion', 'Matched_Etapa', 'Matched_Categoria']
        investors_df = investors_df.join(score_results)
        results_df = investors_df[investors_df['Score'] > 0].sort_values(by='Score', ascending=False).head(50)

        # Columnas específicas que quiere mostrar el usuario
        cols_to_show = [
            'id', 'Company_Name', 'Company_Description', 'Investing_Stage',
            'Company_Location', 'Investment_Categories', 'Company_Linkedin', 'Score'
        ]

        results_df['Score'] = results_df['Score'].map('{:,.1f}'.format)
        results_df = results_df.fillna('-')
        print(f"  ✅ NORMAL search completed, {len(results_df)} results.")
        
        return {
            "search_type": "normal",
            "results": results_df[[c for c in cols_to_show if c in results_df.columns]].to_dict('records'),
            "total_found": len(results_df)
        }

    except Exception as e:
        print(f"  ❌ ERROR in normal investor search: {e}")
        return {"error": f"Error in normal search: {e}"}

def run_investor_search_deep(query):
    """Ejecuta búsqueda DEEP RESEARCH de inversores."""
    if not engine:
        return {"error": "No database connection."}

    print("-> Starting DEEP RESEARCH investor search...")
    try:
        # Primero hacemos la búsqueda normal
        normal_results = run_investor_search_normal(query)
        if "error" in normal_results:
            return normal_results
        
        # Luego aplicamos análisis más profundo usando Gemini
        print("  -> Applying DEEP RESEARCH analysis...")
        
        model = genai.GenerativeModel(MODEL_NAME)
        deep_prompt = f"""
**DEEP RESEARCH TASK:** Analiza esta consulta de startup y proporciona insights avanzados para matching de inversores.

**Consulta:** "{query}"

**Tu tarea:**
1. Identifica el sector/industria exacto
2. Determina la etapa de inversión más probable
3. Identifica factores de riesgo y oportunidades
4. Sugiere tipos de inversores ideales
5. Calcula métricas de compatibilidad adicionales

**Responde en formato conciso (máximo 200 palabras):**
- Sector principal: [sector]
- Etapa recomendada: [etapa]
- Factores clave: [3-4 factores]
- Timing: [análisis de timing del mercado]
"""
        
        deep_analysis = model.generate_content(deep_prompt)
        
        print(f"  ✅ DEEP RESEARCH completed, {normal_results['total_found']} results with advanced analysis.")
        
        return {
            "search_type": "deep_research",
            "results": normal_results["results"],
            "total_found": normal_results["total_found"],
            "deep_analysis": deep_analysis.text,
            "insights": "Análisis profundo aplicado con factores de riesgo, oportunidades y métricas avanzadas"
        }

    except Exception as e:
        print(f"  ❌ ERROR in deep investor search: {e}")
        return {"error": f"Error in deep search: {e}"}

# ==============================================================================
#           ALGORITMO DE BÚSQUEDA DE EMPLEADOS DE FONDOS
# ==============================================================================

def find_employees_from_investors(investor_results, search_type="normal"):
    """
    LÓGICA CORRECTA: 
    1. Toma los 50 mejores fondos de inversión encontrados
    2. Extrae sus Company_Name 
    3. Busca TODOS los empleados que trabajan en esos fondos con decision_score > 44
    """
    if not engine:
        return {"error": "No database connection."}

    print(f"-> Finding employees from {search_type} investor search...")
    
    try:
        # Extraer los nombres de las empresas de inversión
        if "results" not in investor_results:
            return {"error": "No investor results provided"}
        
        investor_companies = []
        for investor in investor_results["results"]:
            company_name = investor.get("Company_Name", "").strip()
            if company_name and company_name != "-":
                investor_companies.append(company_name)
        
        if not investor_companies:
            return {"error": "No valid company names found in investor results"}
        
        print(f"  -> Searching employees in {len(investor_companies)} investment firms...")
        
        # Crear la consulta SQL para buscar empleados en esas empresas
        # Usar ILIKE para búsqueda case-insensitive y filtrar por decision_score > 44
        company_conditions = " OR ".join([f'"Company_Name" ILIKE %s' for _ in investor_companies])
        
        sql_query = f"""
        SELECT id, "fullName", "headline", "current_job_title", "location", 
               "linkedinUrl", "email", "profilePic", "Company_Name",
               "decision_score"
        FROM employees
        WHERE ({company_conditions}) AND "decision_score" > 44
        ORDER BY "decision_score" DESC, "Company_Name", "current_job_title"
        """
        
        # Preparar parámetros para la consulta (agregar % para búsqueda parcial)
        params = tuple(f"%{company}%" for company in investor_companies)
        
        employees_df = pd.read_sql(sql_query, engine, params=params)
        
        print(f"  -> Found {len(employees_df)} high-quality employees (score > 44) across {len(investor_companies)} investment firms")
        
        if employees_df.empty:
            return {
                "search_type": "fund_employees", 
                "from_search": search_type,
                "results": [],
                "total_found": 0,
                "searched_funds": investor_companies,
                "message": "No se encontraron empleados con score > 44 en los fondos seleccionados"
            }
        
        # Preparar columnas para mostrar (las que especificó el usuario)
        cols_to_show = [
            'id', 'fullName', 'headline', 'current_job_title', 'location',
            'linkedinUrl', 'email', 'profilePic'
        ]
        
        # Limpiar datos
        employees_df = employees_df.fillna('-')
        
        # Agrupar por empresa para mejor presentación
        employees_by_fund = {}
        for _, employee in employees_df.iterrows():
            fund_name = employee['Company_Name']
            if fund_name not in employees_by_fund:
                employees_by_fund[fund_name] = []
            employees_by_fund[fund_name].append(employee.to_dict())
        
        print(f"  ✅ Employee search completed: {len(employees_df)} high-quality employees from {len(employees_by_fund)} funds")
        
        return {
            "search_type": "fund_employees",
            "from_search": search_type,
            "results": employees_df[[c for c in cols_to_show if c in employees_df.columns]].to_dict('records'),
            "employees_by_fund": employees_by_fund,
            "total_found": len(employees_df),
            "funds_found": len(employees_by_fund),
            "searched_funds": investor_companies
        }

    except Exception as e:
        print(f"  ❌ ERROR in employee search from investors: {e}")
        return {"error": f"Error finding employees from investment firms: {e}"}

def run_employee_search(query, search_type="normal"):
    """
    Función principal para búsqueda de empleados:
    1. Primero busca los 50 mejores fondos con el query
    2. Luego encuentra TODOS los empleados de esos fondos con decision_score > 44
    """
    print(f"-> Starting {search_type} employee search for: '{query[:50]}...'")
    
    try:
        # PASO 1: Buscar los mejores fondos de inversión
        if search_type == "deep":
            investor_results = run_investor_search_deep(query)
        else:
            investor_results = run_investor_search_normal(query)
        
        if "error" in investor_results:
            return {
                "error": f"Error finding relevant investment funds: {investor_results['error']}"
            }
        
        print(f"  -> Found {investor_results.get('total_found', 0)} relevant investment funds")
        
        # PASO 2: Buscar empleados en esos fondos
        employee_results = find_employees_from_investors(investor_results, search_type)
        
        if "error" in employee_results:
            return employee_results
        
        # PASO 3: Combinar resultados
        combined_results = {
            "search_type": f"combined_{search_type}",
            "query": query,
            "relevant_funds": investor_results["results"][:10],  # Solo los top 10 fondos para contexto
            "total_funds_found": investor_results.get("total_found", 0),
            "employees": employee_results["results"],
            "employees_by_fund": employee_results.get("employees_by_fund", {}),
            "total_employees_found": employee_results.get("total_found", 0),
            "funds_with_employees": employee_results.get("funds_found", 0)
        }
        
        # Si es deep research, incluir el análisis
        if search_type == "deep" and "deep_analysis" in investor_results:
            combined_results["deep_analysis"] = investor_results["deep_analysis"]
            combined_results["insights"] = investor_results.get("insights", "")
        
        return combined_results

    except Exception as e:
        print(f"  ❌ ERROR in combined employee search: {e}")
        return {"error": f"Error in employee search: {e}"}

# ==============================================================================
#           FUNCIONES DE SESIÓN Y GESTIÓN DE DATOS
# ==============================================================================

def save_investor_to_session(project_id, user_id, investor_id):
    """Guarda un inversor en la sesión del proyecto."""
    try:
        insert_query = """
        INSERT INTO project_saved_investors (project_id, investor_id, added_at)
        VALUES (%s, %s, %s)
        ON CONFLICT (project_id, investor_id) DO NOTHING
        """
        params = (project_id, investor_id, datetime.now())
        
        with engine.connect() as conn:
            conn.execute(text(insert_query), params)
            conn.commit()
        
        return True
    except Exception as e:
        print(f"❌ ERROR saving investor to session: {e}")
        return False

def save_employee_to_session(project_id, user_id, employee_id):
    """Guarda un empleado en la sesión del proyecto."""
    try:
        insert_query = """
        INSERT INTO project_saved_employees (project_id, employee_id, added_at)
        VALUES (%s, %s, %s)
        ON CONFLICT (project_id, employee_id) DO NOTHING
        """
        params = (project_id, employee_id, datetime.now())
        
        with engine.connect() as conn:
            conn.execute(text(insert_query), params)
            conn.commit()
        
        return True
    except Exception as e:
        print(f"❌ ERROR saving employee to session: {e}")
        return False

def save_sentiment(project_id, user_id, entity_id, entity_type, sentiment):
    """Guarda el sentiment (like/dislike) de una entidad."""
    try:
        insert_query = """
        INSERT INTO project_sentiments (id, project_id, user_id, entity_id, entity_type, sentiment, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (project_id, entity_id, entity_type) 
        DO UPDATE SET sentiment = EXCLUDED.sentiment, created_at = EXCLUDED.created_at
        """
        
        params = (
            str(uuid.uuid4()),
            project_id,
            user_id,
            entity_id,
            entity_type,
            sentiment,
            datetime.now()
        )
        
        with engine.connect() as conn:
            conn.execute(text(insert_query), params)
            conn.commit()
        
        return True
    except Exception as e:
        print(f"❌ ERROR saving sentiment: {e}")
        return False

def get_saved_investors(project_id):
    """Obtiene los inversores guardados de la sesión."""
    try:
        query = """
        SELECT i.id, i."Company_Name", i."Company_Description", i."Investing_Stage",
               i."Company_Location", i."Investment_Categories", i."Company_Linkedin",
               psi.added_at, COALESCE(ps.sentiment, 'none') as sentiment
        FROM project_saved_investors psi
        JOIN investors i ON psi.investor_id = i.id
        LEFT JOIN project_sentiments ps ON i.id = ps.entity_id AND ps.project_id = psi.project_id
        WHERE psi.project_id = %s
        ORDER BY psi.added_at DESC
        """
        
        results_df = pd.read_sql(query, engine, params=(project_id,))
        return results_df.fillna('-').to_dict('records')
        
    except Exception as e:
        print(f"❌ ERROR getting saved investors: {e}")
        return []

def get_saved_employees(project_id):
    """Obtiene los empleados guardados de la sesión."""
    try:
        query = """
        SELECT e.id, e."fullName", e."headline", e."current_job_title", e."location",
               e."linkedinUrl", e."email", e."profilePic", e."Company_Name",
               pse.added_at, COALESCE(ps.sentiment, 'none') as sentiment
        FROM project_saved_employees pse
        JOIN employees e ON pse.employee_id = e.id
        LEFT JOIN project_sentiments ps ON e.id = ps.entity_id AND ps.project_id = pse.project_id
        WHERE pse.project_id = %s
        ORDER BY pse.added_at DESC
        """
        
        results_df = pd.read_sql(query, engine, params=(project_id,))
        return results_df.fillna('-').to_dict('records')
        
    except Exception as e:
        print(f"❌ ERROR getting saved employees: {e}")
        return []

# ==============================================================================
#           GENERACIÓN DE TEMPLATES CON GEMINI
# ==============================================================================

def generate_outreach_template(project_id, user_id, target_investor_id=None, target_employee_id=None, platform="email", user_instructions=""):
    """Genera template de outreach personalizado usando Gemini."""
    try:
        # Obtener información del target
        target_info = {}
        target_type = ""
        
        if target_investor_id:
            target_query = """
            SELECT "Company_Name", "Company_Description", "Investing_Stage", 
                   "Investment_Categories", "Company_Location"
            FROM investors WHERE id = %s
            """
            target_df = pd.read_sql(target_query, engine, params=(target_investor_id,))
            if not target_df.empty:
                target_info = target_df.iloc[0].to_dict()
                target_type = "investor"
        
        elif target_employee_id:
            target_query = """
            SELECT "fullName", "headline", "current_job_title", "Company_Name", 
                   "location", "linkedinUrl", "email"
            FROM employees WHERE id = %s
            """
            target_df = pd.read_sql(target_query, engine, params=(target_employee_id,))
            if not target_df.empty:
                target_info = target_df.iloc[0].to_dict()
                target_type = "employee"
        
        if not target_info:
            return {"error": "Target not found"}
        
        # Obtener información del proyecto (contexto de la startup)
        project_query = """
        SELECT project_name, project_description, kpi_data
        FROM projects WHERE id = %s
        """
        project_df = pd.read_sql(project_query, engine, params=(project_id,))
        project_context = {}
        if not project_df.empty:
            project_context = project_df.iloc[0].to_dict()
        
        # Preparar prompt para Gemini
        if target_type == "investor":
            target_context = f"""
            Fondo: {target_info.get('Company_Name', 'N/A')}
            Descripción: {target_info.get('Company_Description', 'N/A')}
            Etapa de Inversión: {target_info.get('Investing_Stage', 'N/A')}
            Categorías: {target_info.get('Investment_Categories', 'N/A')}
            Ubicación: {target_info.get('Company_Location', 'N/A')}
            """
        else:
            target_context = f"""
            Nombre: {target_info.get('fullName', 'N/A')}
            Título: {target_info.get('current_job_title', 'N/A')}
            Empresa: {target_info.get('Company_Name', 'N/A')}
            Headline: {target_info.get('headline', 'N/A')}
            Ubicación: {target_info.get('location', 'N/A')}
            """
        
        template_prompt = f"""
**Tarea:** Genera un {platform} profesional y directo para solicitar una reunión de inversión.

**Target ({target_type}):**
{target_context}

**Mi Startup:**
Nombre: {project_context.get('project_name', 'Mi Startup')}
Descripción: {project_context.get('project_description', 'Startup innovadora')}

**Instrucciones adicionales:** {user_instructions}

**Estilo "0Bullshit" - Requisitos:**
1. **Directo y sin rodeos** - Máximo 120 palabras
2. **Asunto claro** (si es email)
3. **Párrafo 1:** Quién soy y qué hacemos (1-2 líneas)
4. **Párrafo 2:** Por qué este {target_type} específicamente (1-2 líneas)
5. **Call-to-action:** Solicitud directa de 15-20 min de reunión
6. **Tono:** Profesional pero humano, confiado sin ser arrogante

**Formato de salida:**
{f"Asunto: [asunto aquí]" if platform == "email" else ""}

[mensaje aquí]

{"Saludos," if platform == "email" else ""}
[Nombre]
"""

        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(template_prompt)
        template_content = response.text.strip()
        
        # Guardar template en la base de datos
        template_id = str(uuid.uuid4())
        insert_query = """
        INSERT INTO template_generators (id, project_id, user_id, target_investor_id, target_employee_id, platform, conversation_history, generated_template, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        params = (
            template_id,
            project_id,
            user_id,
            target_investor_id,
            target_employee_id,
            platform,
            json.dumps([]),
            template_content,
            datetime.now(),
            datetime.now()
        )
        
        with engine.connect() as conn:
            conn.execute(text(insert_query), tuple(params))
            conn.commit()
        
        return {
            "template_id": template_id,
            "content": template_content,
            "platform": platform,
            "target_type": target_type,
            "target_info": target_info
        }
        
    except Exception as e:
        print(f"❌ ERROR generating outreach template: {e}")
        return {"error": "No se pudo generar la plantilla de outreach"}

# ==============================================================================
#           GEMINI MASTER AI - SUPERINTELIGENTE
# ==============================================================================

def get_master_ai_prompt():
    """Prompt principal para el AI superinteligente que decide qué hacer."""
    return """## ROL Y PERSONALIDAD

Eres "0Bullshit", un supermentor de IA para startups y emprendedores. Tu misión es ayudar con fundraising, búsqueda de talento, y estrategia de negocio sin rodeos.

Tu personalidad:
- **Directo y eficiente:** Vas directo al grano
- **Proactivo:** Sugieres acciones específicas
- **Inteligente:** Decides automáticamente qué herramientas usar
- **Orientado a resultados:** Siempre das next steps

## HERRAMIENTAS DISPONIBLES

Tienes acceso a estas funciones:
1. **investor_search_normal:** Búsqueda estándar de inversores (50 mejores)
2. **investor_search_deep:** Búsqueda avanzada con análisis profundo (50 mejores + insights)
3. **employee_search:** Búsqueda de contactos en fondos de inversión (score > 44)
4. **general_chat:** Conversación general, consejos, estrategia

## LÓGICA DE DECISIÓN

**USA investor_search_normal CUANDO:**
- El usuario pida "buscar inversores", "encontrar VCs", "fundraising"
- Mencione necesidades de capital simples
- Búsqueda rápida de inversores

**USA investor_search_deep CUANDO:**
- El usuario pida "análisis profundo", "deep research", "investigación avanzada"
- Mencione factores complejos como riesgo, mercado, timing
- Necesite insights estratégicos de inversión

**USA employee_search CUANDO:**
- El usuario pida "buscar contactos", "encontrar people", "networking"
- Quiera contactar personas específicas en fondos de inversión
- Mencione "associates", "partners", "analysts" de VCs
- Hable de warm introductions, contactos en fondos

**USA general_chat PARA TODO LO DEMÁS:**
- Preguntas generales de estrategia
- Consejos de negocio
- Dudas sobre startups
- Conversación normal

## CONTEXTO DE LA CONVERSACIÓN

**Mensaje del usuario:** {user_message}

## FORMATO DE RESPUESTA OBLIGATORIO

Responde SIEMPRE con un único objeto JSON:

```json
{{
  "action": "nombre_de_la_accion",
  "reasoning": "por qué elegiste esta acción",
  "response": "tu respuesta al usuario",
  "parameters": {{
    "query": "consulta procesada para la búsqueda (si aplica)"
  }}
}}
```

**ACCIONES DISPONIBLES:**
- "investor_search_normal"
- "investor_search_deep" 
- "employee_search"
- "general_chat"

## EJEMPLOS

**Usuario:** "Necesito inversores para mi fintech de pagos en México"
**Tu respuesta:**
```json
{{
  "action": "investor_search_normal",
  "reasoning": "Solicitud directa de inversores para fintech en México, búsqueda estándar es apropiada",
  "response": "¡Perfecto! Voy a buscar inversores especializados en fintech y pagos en México. Te traigo los 50 mejores matches.",
  "parameters": {{
    "query": "fintech pagos México seed series A venture capital"
  }}
}}
```

**Usuario:** "Busco contactos en fondos que inviertan en AI"
**Tu respuesta:**
```json
{{
  "action": "employee_search",
  "reasoning": "Solicita contactos específicos en fondos de inversión relevantes para AI",
  "response": "¡Genial! Primero busco los mejores fondos que invierten en AI, y luego te traigo TODOS los associates, partners y analysts que trabajan en esos fondos para networking directo.",
  "parameters": {{
    "query": "artificial intelligence AI machine learning deep tech venture capital"
  }}
}}
```

Analiza el mensaje del usuario y responde con el JSON apropiado."""

def process_master_ai_request(user_message):
    """Procesa la solicitud usando el AI master superinteligente."""
    try:
        # Preparar el prompt principal
        master_prompt = get_master_ai_prompt().format(user_message=user_message)
        
        # Llamar a Gemini
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(master_prompt)
        response_text = response.text.strip()
        
        # Extraer JSON de la respuesta
        json_match = re.search(r'```json\s*([\s\S]+?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Intentar encontrar JSON sin bloques de código
            json_match = re.search(r'(\{[\s\S]*\})', response_text)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                json_str = response_text
        
        # Parsear JSON
        action_data = json.loads(json_str)
        return action_data
        
    except Exception as e:
        print(f"❌ ERROR processing master AI request: {e}")
        return {
            "action": "general_chat",
            "reasoning": "Error en el procesamiento, defaulting a chat general",
            "response": "Disculpa, tuve un error procesando tu solicitud. ¿Puedes reformular tu pregunta?",
            "parameters": {}
        }

def execute_master_ai_action(action_data):
    """Ejecuta la acción determinada por el AI master."""
    action = action_data.get("action")
    parameters = action_data.get("parameters", {})
    ai_response = action_data.get("response", "")
    
    if action == "investor_search_normal":
        query = parameters.get("query", "")
        if not query:
            return {
                "type": "error",
                "content": "No se pudo generar la consulta de búsqueda."
            }
        
        search_results = run_investor_search_normal(query)
        
        if "error" in search_results:
            return {
                "type": "error",
                "content": search_results["error"]
            }
        
        return {
            "type": "investor_results_normal",
            "ai_response": ai_response,
            "search_results": search_results,
            "message": f"✅ Búsqueda normal completada: {search_results['total_found']} inversores encontrados"
        }
    
    elif action == "investor_search_deep":
        query = parameters.get("query", "")
        if not query:
            return {
                "type": "error",
                "content": "No se pudo generar la consulta de búsqueda profunda."
            }
        
        search_results = run_investor_search_deep(query)
        
        if "error" in search_results:
            return {
                "type": "error",
                "content": search_results["error"]
            }
        
        return {
            "type": "investor_results_deep",
            "ai_response": ai_response,
            "search_results": search_results,
            "message": f"🔍 Deep research completado: {search_results['total_found']} inversores + análisis avanzado"
        }
    
    elif action == "employee_search":
        query = parameters.get("query", "")
        if not query:
            return {
                "type": "error",
                "content": "No se pudo generar la consulta de búsqueda de empleados."
            }
        
        search_results = run_employee_search(query)
        
        if "error" in search_results:
            return {
                "type": "error",
                "content": search_results["error"]
            }
        
        return {
            "type": "employee_results",
            "ai_response": ai_response,
            "search_results": search_results,
            "message": f"👥 Encontré {search_results.get('total_employees_found', 0)} contactos (score > 44) en {search_results.get('funds_with_employees', 0)} fondos relevantes"
        }
    
    elif action == "general_chat":
        return {
            "type": "text_response",
            "content": ai_response,
            "ai_response": ai_response
        }
    
    else:
        return {
            "type": "error",
            "content": "Acción no reconocida."
        }

# ==============================================================================
#           API ROUTES - SISTEMA DE SESIONES TEMPORALES
# ==============================================================================

print("5. Defining API routes...")

@app.route('/')
def home():
    """Main route to verify the API is working."""
    return "<h1>🚀 0Bullshit - Backend Completo - READY! 🚀</h1>"

@app.route('/session/info', methods=['GET'])
def get_session_info():
    """Obtiene información de la sesión actual."""
    try:
        project_id, user_id = get_or_create_session_project()
        
        # Obtener estadísticas de la sesión
        saved_investors = get_saved_investors(project_id)
        saved_employees = get_saved_employees(project_id)
        
        return jsonify({
            "session_active": True,
            "project_id": project_id,
            "user_id": user_id,
            "saved_investors_count": len(saved_investors),
            "saved_employees_count": len(saved_employees)
        })
        
    except Exception as e:
        print(f"❌ ERROR getting session info: {e}")
        return jsonify({"error": "Could not get session info"}), 500

@app.route('/chat', methods=['POST'])
def master_chat_endpoint():
    """Endpoint principal para el chat con AI superinteligente."""
    print("\n--- Request /chat (Master AI) ---")
    data = request.json
    user_message = data.get('message')

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    print(f"  -> Message: '{user_message[:50]}...'")

    try:
        # Asegurarse de que hay una sesión activa
        project_id, user_id = get_or_create_session_project()
        
        # Procesar la solicitud con el AI master
        action_data = process_master_ai_request(user_message)
        
        # Ejecutar la acción determinada
        result = execute_master_ai_action(action_data)
        
        # Agregar info de sesión al resultado
        result["session_info"] = {
            "project_id": project_id,
            "user_id": user_id
        }
        
        print("--- Request /chat (Master AI) completed ---")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ ERROR in master chat endpoint: {e}")
        return jsonify({
            "type": "error",
            "content": "Hubo un error procesando tu solicitud. Inténtalo de nuevo."
        }), 500

@app.route('/search/investors', methods=['POST'])
def search_investors_endpoint():
    """Endpoint directo para búsqueda de inversores."""
    print("\n--- Request /search/investors ---")
    data = request.json
    query = data.get('query')
    search_type = data.get('type', 'normal')  # 'normal' o 'deep'

    if not query:
        return jsonify({"error": "query is required"}), 400

    print(f"  -> Query: '{query[:50]}...', Type: {search_type}")

    try:
        project_id, user_id = get_or_create_session_project()
        
        if search_type == 'deep':
            results = run_investor_search_deep(query)
        else:
            results = run_investor_search_normal(query)
        
        results["session_info"] = {"project_id": project_id, "user_id": user_id}
        
        print("--- Request /search/investors completed ---")
        return jsonify(results)
        
    except Exception as e:
        print(f"❌ ERROR in investor search endpoint: {e}")
        return jsonify({"error": f"Error en búsqueda de inversores: {e}"}), 500

@app.route('/search/employees', methods=['POST'])
def search_employees_endpoint():
    """Endpoint directo para búsqueda de empleados en fondos de inversión."""
    print("\n--- Request /search/employees ---")
    data = request.json
    query = data.get('query')
    search_type = data.get('type', 'normal')  # 'normal' o 'deep'

    if not query:
        return jsonify({"error": "query is required"}), 400

    print(f"  -> Query: '{query[:50]}...', Type: {search_type}")

    try:
        project_id, user_id = get_or_create_session_project()
        
        results = run_employee_search(query, search_type)
        results["session_info"] = {"project_id": project_id, "user_id": user_id}
        
        print("--- Request /search/employees completed ---")
        return jsonify(results)
        
    except Exception as e:
        print(f"❌ ERROR in employee search endpoint: {e}")
        return jsonify({"error": f"Error en búsqueda de empleados: {e}"}), 500

@app.route('/save/investor', methods=['POST'])
def save_investor_endpoint():
    """Guarda un inversor en la sesión."""
    try:
        data = request.json
        investor_id = data.get('investor_id')
        
        if not investor_id:
            return jsonify({"error": "investor_id is required"}), 400
        
        project_id, user_id = get_or_create_session_project()
        
        success = save_investor_to_session(project_id, user_id, investor_id)
        
        if success:
            return jsonify({"message": "Investor saved successfully"})
        else:
            return jsonify({"error": "Failed to save investor"}), 500
            
    except Exception as e:
        print(f"❌ ERROR saving investor: {e}")
        return jsonify({"error": "Could not save investor"}), 500

@app.route('/save/employee', methods=['POST'])
def save_employee_endpoint():
    """Guarda un empleado en la sesión."""
    try:
        data = request.json
        employee_id = data.get('employee_id')
        
        if not employee_id:
            return jsonify({"error": "employee_id is required"}), 400
        
        project_id, user_id = get_or_create_session_project()
        
        success = save_employee_to_session(project_id, user_id, employee_id)
        
        if success:
            return jsonify({"message": "Employee saved successfully"})
        else:
            return jsonify({"error": "Failed to save employee"}), 500
            
    except Exception as e:
        print(f"❌ ERROR saving employee: {e}")
        return jsonify({"error": "Could not save employee"}), 500

@app.route('/sentiment', methods=['POST'])
def save_sentiment_endpoint():
    """Guarda el sentiment (like/dislike) de una entidad."""
    try:
        data = request.json
        entity_id = data.get('entity_id')
        entity_type = data.get('entity_type')  # 'investor' o 'employee'
        sentiment = data.get('sentiment')  # 'like' o 'dislike'
        
        if not all([entity_id, entity_type, sentiment]):
            return jsonify({"error": "entity_id, entity_type, and sentiment are required"}), 400
        
        project_id, user_id = get_or_create_session_project()
        
        # Guardar en tabla correspondiente si es like
        if sentiment == 'like':
            if entity_type == 'investor':
                save_investor_to_session(project_id, user_id, entity_id)
            elif entity_type == 'employee':
                save_employee_to_session(project_id, user_id, entity_id)
        
        # Guardar sentiment
        success = save_sentiment(project_id, user_id, entity_id, entity_type, sentiment)
        
        if success:
            return jsonify({"message": f"Sentiment '{sentiment}' saved successfully"})
        else:
            return jsonify({"error": "Failed to save sentiment"}), 500
            
    except Exception as e:
        print(f"❌ ERROR saving sentiment: {e}")
        return jsonify({"error": "Could not save sentiment"}), 500

@app.route('/saved/investors', methods=['GET'])
def get_saved_investors_endpoint():
    """Obtiene los inversores guardados de la sesión."""
    try:
        project_id, user_id = get_or_create_session_project()
        
        saved_investors = get_saved_investors(project_id)
        
        return jsonify({
            "saved_investors": saved_investors,
            "total_count": len(saved_investors)
        })
        
    except Exception as e:
        print(f"❌ ERROR getting saved investors: {e}")
        return jsonify({"error": "Could not get saved investors"}), 500

@app.route('/saved/employees', methods=['GET'])
def get_saved_employees_endpoint():
    """Obtiene los empleados guardados de la sesión."""
    try:
        project_id, user_id = get_or_create_session_project()
        
        saved_employees = get_saved_employees(project_id)
        
        return jsonify({
            "saved_employees": saved_employees,
            "total_count": len(saved_employees)
        })
        
    except Exception as e:
        print(f"❌ ERROR getting saved employees: {e}")
        return jsonify({"error": "Could not get saved employees"}), 500

@app.route('/generate/template', methods=['POST'])
def generate_template_endpoint():
    """Genera template de outreach personalizado."""
    try:
        data = request.json
        target_investor_id = data.get('target_investor_id')
        target_employee_id = data.get('target_employee_id')
        platform = data.get('platform', 'email')  # 'email' o 'linkedin'
        user_instructions = data.get('instructions', '')
        
        if not target_investor_id and not target_employee_id:
            return jsonify({"error": "target_investor_id or target_employee_id is required"}), 400
        
        project_id, user_id = get_or_create_session_project()
        
        result = generate_outreach_template(
            project_id, user_id, target_investor_id, target_employee_id, platform, user_instructions
        )
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ ERROR generating template: {e}")
        return jsonify({"error": "Could not generate template"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected" if engine else "disconnected",
        "gemini": "configured" if API_KEY else "not_configured",
        "session_support": True
    })

# Endpoint legacy para compatibilidad
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
    
    try:
        project_id, user_id = get_or_create_session_project()
        
        if is_deep:
            results = run_investor_search_deep(user_query)
        else:
            results = run_investor_search_normal(user_query)
        
        results["session_info"] = {"project_id": project_id, "user_id": user_id}
        
        print("--- Request /find_investors (Legacy) completed ---")
        return jsonify(results)
        
    except Exception as e:
        print(f"❌ ERROR in legacy endpoint: {e}")
        return jsonify({"error": f"Error in search: {e}"}), 500

# ==============================================================================
#           EXECUTION
# ==============================================================================

if __name__ == '__main__':
    print("Starting Flask server for local testing...")
    print("🚀 0Bullshit Enhanced Backend READY! 🚀")
    print("🔓 MODO ABIERTO - Sesiones temporales activadas")
    print("🤖 AI Superinteligente activado")
    print("💼 Algoritmos de matching: Inversores + Empleados (score > 44)")
    print("👍 Sistema completo de like/dislike")
    print("📧 Generación de templates con Gemini")
    app.run(host='0.0.0.0', port=8080, debug=True)
