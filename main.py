from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

import requests

# Remove FAISS imports
load_dotenv()
# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
METABASE_URL = os.getenv("METABASE_URL")
METABASE_USER = os.getenv("METABASE_USER")
METABASE_PASSWORD = os.getenv("METABASE_PASSWORD")
METABASE_DB_ID = int(os.getenv("METABASE_DB_ID"))
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")

# OpenAI init
client = OpenAI()

app = FastAPI()

print(OPENAI_API_KEY)
class QuestionRequest(BaseModel):
    question: str
    chart_type: Optional[str] = "bar"

def ask_openai(question: str) -> str:
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Return only valid SQL using the uploaded schema. Do not assume any columns."},
            {"role": "user", "content": "Return only valid SQL using the uploaded schema. Do not assume any columns. \n\n" + question}
        ],
        temperature=0
    )
    return res.choices[0].message.content.replace("```sql", "").replace("```", "").strip()

def get_metabase_session() -> str:
    """Authenticate to Metabase and return session token"""
    res = requests.post(f"{METABASE_URL}/api/session", json={
        "username": METABASE_USER,
        "password": METABASE_PASSWORD
    })
    res.raise_for_status()
    return res.json()["id"]

def create_metabase_card(sql: str, name: str = "OpenAI Chart", chart_type: str = "bar"):
    """Create a new chart in Metabase with dynamic axis settings"""
    session = get_metabase_session()
    headers = {"X-Metabase-Session": session}

    # Try to extract selected columns from SQL
    import re
    match = re.findall(r"SELECT\s+(.*?)\s+FROM", sql, re.IGNORECASE | re.DOTALL)
    x_axis = "x"
    y_axis = "y"
    if match:
        cols = match[0].split(",")
        if len(cols) >= 2:
            x_axis = cols[0].split("AS")[-1].strip().strip('"').strip()
            y_axis = cols[1].split("AS")[-1].strip().strip('"').strip()
        elif len(cols) == 1:
            x_axis = cols[0].split("AS")[-1].strip().strip('"').strip()

    payload = {
        "name": name,
        "description": "Generated via OpenAI",
        "display": chart_type,
        "dataset_query": {
            "type": "native",
            "native": {"query": sql},
            "database": METABASE_DB_ID
        },
        "visualization_settings": {
            "graph.x_axis": x_axis,
            "graph.y_axis": y_axis
        }
    }

    res = requests.post(f"{METABASE_URL}/api/card", json=payload, headers=headers)
    res.raise_for_status()
    return res.json()

def add_card_to_dashboard(card_id: int, dashboard_id: int, session: str):
    headers = {"X-Metabase-Session": session}

    # 1️⃣ Get existing dashboard layout
    res = requests.get(f"{METABASE_URL}/api/dashboard/{dashboard_id}", headers=headers)
    res.raise_for_status()
    dashboard = res.json()

    # 2️⃣ Reformat existing cards to layout format only
    existing_cards = []
    for card in dashboard.get("ordered_cards", []):
        if card.get("card") and card["card"].get("id"):
            existing_cards.append({
                "id": card["card"]["id"],
                "card_id": card["card"]["id"],
                "row": card.get("row", 0),
                "col": card.get("col", 0),
                "size_x": card.get("sizeX", 10),
                "size_y": card.get("sizeY", 10),
                "parameter_mappings": card.get("parameter_mappings", []),
                "series": card.get("series", [])
            })

    # 3️⃣ Add new card layout
    new_card = {
        "id": card_id,
        "card_id": card_id,
        "row": 0,
        "col": 0,
        "size_x": 10,
        "size_y": 10,
        "parameter_mappings": [],
        "series": []
    }

    payload = {"cards": existing_cards + [new_card]}

    import json
    print("🛠️ Sending payload to Metabase /dashboard/cards:", json.dumps(payload, indent=2))

    # 4️⃣ PUT updated layout
    res = requests.put(
        f"{METABASE_URL}/api/dashboard/{dashboard_id}/cards",
        json=payload,
        headers=headers
    )
    res.raise_for_status()
    return res.json()





# New endpoint: /responses
@app.post("/responses")
def get_response_from_vectorstore(request: QuestionRequest):
    try:
        search_resp = client.vector_stores.search(
            vector_store_id=VECTOR_STORE_ID,
            query=request.question,
            max_num_results=3
        )

        top_chunks = [
            "".join(segment.text for segment in chunk.content)
            for chunk in search_resp.data
        ]
        schema_context = "\n".join(top_chunks)

        # Generate SQL using OpenAI chat completion
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a data analyst.
Only output valid SQL.
Do not include any explanation, formatting, or comments.
Do not say anything other than SQL.
Schema:\n{schema_context}"""
                },
                {
                    "role": "user",
                    "content": f"Only output SQL. Do not explain. Question:\n{request.question}"
                }
            ],
            temperature=0
        )
        sql = res.choices[0].message.content.strip().replace("```sql", "").replace("```", "")

        # Create Metabase card and add to dashboard
        session = get_metabase_session()
        card = create_metabase_card(sql, name=request.question[:40], chart_type=request.chart_type)
        dashboard_id = int(os.getenv("METABASE_DASHBOARD_ID", "3"))
        dashboard_card = add_card_to_dashboard(card["id"], dashboard_id, session)

        return {
            "sql": sql,
            "metabase_card_id": card["id"],
            "dashboard_id": dashboard_id,
            "dashboard_card_id": dashboard_card["id"],
            "metabase_dashboard_url": f"{METABASE_URL}/dashboard/{dashboard_id}"
        }

    except OpenAIError as oe:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(oe)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
