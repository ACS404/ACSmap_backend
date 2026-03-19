from flask import Blueprint, request, jsonify
import google.generativeai as genai
import os

acs_chat_api = Blueprint('acs_chat_api', __name__, url_prefix='/api')

genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

SYSTEM_PROMPT = """You are a data-driven cancer risk assistant for the American Cancer Society (ACS), powered by ACS Cancer Facts & Figures 2026.

CORE RULE — answer the specific question asked. Do NOT default to a generic description of the cancer type's symptoms and treatments unless that is literally what was asked.

HOW TO ANSWER:
- Lead with a direct answer to the exact question.
- Back it up with specific numbers: relative risk multipliers, lifetime risk percentages, incidence rates, survival rates, or study findings. Use real ACS/NCI epidemiological data.
- If the message includes a user profile (age, smoking status, family history, BMI, prediction results, etc.), reference those specific details in your answer. Make the answer feel like it was written for that exact person, not a generic patient.
- If a relative risk or prediction result is included in the context, cite it directly (e.g. "Your model shows a 4.1× lung cancer RR — here's what drives that…").
- Do NOT pad the answer with a general overview of symptoms and treatment options unless asked.
- Do NOT end every response with a boilerplate "learn more at cancer.org" line — only include it when it's genuinely useful.
- Keep answers under 180 words. Be specific and direct, not reassuring and vague.
- Never provide a personal medical diagnosis. If the question is about a personal decision, give the data and then recommend a doctor.
"""

@acs_chat_api.route('/acs-chat', methods=['POST'])
def acs_chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Missing message field'}), 400

    message = data.get('message', '').strip()
    if not message:
        return jsonify({'error': 'Message cannot be empty'}), 400

    try:
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash-lite',
            system_instruction=SYSTEM_PROMPT
        )
        response = model.generate_content(message)
        return jsonify({'answer': response.text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500