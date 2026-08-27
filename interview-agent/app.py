import json
import logging
from flask import Flask, render_template, request, Response

from dotenv import load_dotenv
load_dotenv()

from harness.logger import setup_logger
from llm.gemini_client import GeminiClient
from memory.memory import SemanticMemory
from agent.loop import AgentLoop
from main import load_config

app = Flask(__name__)
config = load_config("config.yaml")
logger = setup_logger(config)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.json
    jd_text = data.get("jd", "").strip()
    resume_text = data.get("resume", "").strip()

    if not jd_text or not resume_text:
        return {"error": "Both JD and Resume are required"}, 400

    def event_stream():
        gemini_client = GeminiClient(config)
        memory = SemanticMemory(gemini_client=gemini_client, config=config)
        loop = AgentLoop(gemini_client=gemini_client, memory=memory, config=config)
        
        for event in loop.run_stream(jd=jd_text, resume=resume_text):
            yield f"data: {json.dumps(event)}\n\n"
            
    return Response(event_stream(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
