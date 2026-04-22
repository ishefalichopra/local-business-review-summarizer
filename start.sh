#!/bin/bash
echo "Starting Local Business Review Summarizer..."

# Start containers
docker compose up -d
echo "Qdrant and n8n started"

# Start trigger server in background
source venv/bin/activate
python3 app/trigger.py &
echo "Trigger server started"

# Start streamlit
echo "Starting UI at http://localhost:8501"
streamlit run app/ui.py
