# --- CAMBIOS MÍNIMOS PARA ARREGLAR LOS ERRORES DE BD ---

# CAMBIO 1: En get_project_comprehensive_data() 
# Línea ~65: Cambiar params=(project_id,) por params=[project_id]
def get_project_comprehensive_data(project_id):
    if not engine:
        print("❌ ERROR: Database engine not initialized in get_project_comprehensive_data.")
        return None, [], 0, "Free"

    try:
        project_query = """
        SELECT p.id, p.user_id, p.project_name, p.project_description,
               p.kpi_data, p.status, pr.plan
        FROM projects p
        LEFT JOIN profiles pr ON p.user_id = pr.id
        WHERE p.id = %s
        """
        # FIX: Cambiar de tuple a list
        project_result_df = pd.read_sql(project_query, engine, params=[project_id])

        if project_result_df.empty:
            print(f"Project with ID {project_id} not found in 'projects' table or no matching profile.")
            return None, [], 0, "Free"

        project_data = project_result_df.iloc[0].to_dict()
        
        # Ensure kpi_data is a dictionary (it might be stored as JSON string)
        if isinstance(project_data.get('kpi_data'), str):
            try:
                project_data['kpi_data'] = json.loads(project_data['kpi_data'])
            except json.JSONDecodeError:
                project_data['kpi_data'] = {}
        elif project_data.get('kpi_data') is None:
             project_data['kpi_data'] = {}

        chat_history_query = """
        SELECT role, content, timestamp
        FROM project_conversations
        WHERE project_id = %s
        ORDER BY timestamp ASC
        LIMIT 50
        """
        # FIX: Cambiar de tuple a list
        chat_history_df = pd.read_sql(chat_history_query, engine, params=[project_id])
        chat_history_tuples = [
            {"role": row["role"], "parts": [{"text": row["content"]}]}
            for index, row in chat_history_df.iterrows()
        ]
        
        # Ensure user messages are "user" and model messages are "model" for Gemini
        for entry in chat_history_tuples:
            if entry["role"].lower() == "assistant":
                entry["role"] = "model"
            elif entry["role"].lower() != "user":
                entry["role"] = "user"

        saved_investors_query = """
        SELECT COUNT(*) as count
        FROM project_saved_investors
        WHERE project_id = %s AND sentiment = 'like'
        """
        # FIX: Cambiar de tuple a list
        saved_investors_df = pd.read_sql(saved_investors_query, engine, params=[project_id])
        saved_investors_count = saved_investors_df['count'].iloc[0] if not saved_investors_df.empty else 0
        
        user_plan = project_data.get('plan', 'Free')

        return project_data, chat_history_tuples, saved_investors_count, user_plan

    except Exception as e:
        print(f"❌ ERROR retrieving comprehensive project data: {e}")
        print(f"❌ ERROR retrieving comprehensive project data: {e}") 
        return None, [], 0, "Free"


# CAMBIO 2: En save_conversation_to_db()
# Línea ~119: Asegurar que content sea string y usar parámetros como dict
def save_conversation_to_db(project_id, user_id, role, content):
    if not engine:
        print("❌ ERROR: Database engine not initialized in save_conversation_to_db.")
        return

    # FIX: Ensure content is a string before saving
    content_to_save = content
    if not isinstance(content, str):
        try:
            content_to_save = json.dumps(content)
        except (TypeError, ValueError):
            content_to_save = str(content)

    insert_query = """
    INSERT INTO project_conversations (project_id, user_id, role, content, timestamp)
    VALUES (%(project_id)s, %(user_id)s, %(role)s, %(content)s, %(timestamp)s)
    """
    
    # FIX: Usar parámetros como diccionario en lugar de tuple
    params = {
        'project_id': project_id,
        'user_id': user_id,
        'role': role,
        'content': content_to_save,
        'timestamp': datetime.utcnow()
    }
    
    try:
        with engine.connect() as connection:
            connection.execute(text(insert_query), params)
            connection.commit()
        print(f"Conversation saved: {project_id}, Role: {role}")
    except Exception as e:
        print(f"❌ ERROR saving conversation: {e}")


# CAMBIO 3: En set_entity_sentiment_in_db() 
# Para consistencia, también cambiar el formato
def set_entity_sentiment_in_db(project_id, user_id, entity_id, entity_type, sentiment):
    if not engine:
        print("❌ ERROR: Database engine not initialized in set_entity_sentiment_in_db.")
        return False
    
    upsert_query = """
    INSERT INTO project_sentiments (project_id, user_id, entity_id, entity_type, sentiment, updated_at)
    VALUES (%(project_id)s, %(user_id)s, %(entity_id)s, %(entity_type)s, %(sentiment)s, %(updated_at)s)
    ON CONFLICT (project_id, user_id, entity_id, entity_type) 
    DO UPDATE SET sentiment = EXCLUDED.sentiment, updated_at = EXCLUDED.updated_at;
    """
    
    # FIX: Ya usa diccionario, está bien
    params = {
        "project_id": project_id,
        "user_id": user_id,
        "entity_id": entity_id,
        "entity_type": entity_type,
        "sentiment": sentiment,
        "updated_at": datetime.utcnow()
    }
    try:
        with engine.connect() as connection:
            connection.execute(text(upsert_query), params)
            connection.commit()
        print(f"Sentiment '{sentiment}' for {entity_type} '{entity_id}' saved for project '{project_id}'.")
        return True
    except Exception as e:
        print(f"❌ ERROR saving sentiment: {e}")
        return False
    if __name__ == '__main__':
    print("Starting Flask server for local testing...")
    print("🚀 0Bullshit Enhanced Backend READY! 🚀")
    app.run(host='0.0.0.0', port=8080, debug=True)
