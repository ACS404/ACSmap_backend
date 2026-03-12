from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
import google.generativeai as genai
from __init__ import app
import os

acs_chat_api = Blueprint('acs_chat_api', __name__, url_prefix='/api')
api = Api(acs_chat_api)

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class ACSChatAPI(Resource):
    def options(self):
        return {}, 200

    def post(self):
        try:
            data = request.get_json()

            if not data or 'message' not in data or 'type' not in data:
                return {"error": "Missing required fields: type and message"}, 400

            message = data['message']
            msg_type = data['type']

            if not GEMINI_API_KEY:
                return {"error": "Gemini API key not configured"}, 500

            if msg_type == 'hint':
                prompt = f"""You are a helpful assistant for the American Cancer Society body map tool.
A user is asking about cancer. Give them a SHORT helpful hint (2-3 sentences max) — 
not the full answer. Guide them toward learning more without overwhelming them.
Focus on symptoms, risk factors, or early detection tips.
Do NOT provide medical diagnoses or treatment advice.
Question: {message}"""
            else:
                prompt = f"""You are a knowledgeable, compassionate assistant for the American Cancer Society.
Answer this question about cancer clearly and accurately in plain English.
Include: what it is, key warning signs, main risk factors, and where to learn more (cancer.org).
Keep it under 150 words. Do NOT provide medical diagnoses or personalized treatment advice.
Always encourage consulting a doctor for personal health concerns.
Question: {message}"""

            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            response = model.generate_content(prompt)

            return {
                "success": True,
                "type": msg_type,
                "question": message,
                "answer": response.text
            }, 200

        except Exception as e:
            return {"error": "Internal server error", "details": str(e)}, 500

api.add_resource(ACSChatAPI, '/acs-chat')