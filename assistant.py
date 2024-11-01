import base64
import io
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
from PIL import Image
import numpy as np
import cv2
import sqlite3

def init_db():
    conn = sqlite3.connect('api_counts.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_counts (
            user_id TEXT PRIMARY KEY,
            api_count INTEGER NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

app = Flask(__name__)
CORS(app)

init_db()
class Assistant:
    def __init__(self):
        self.emotion_color_map = {
            "angry": "Red",
            "happy": "Green",
            "sad": "Blue",
            "fear": "Purple",
            "disgust": "Brown",
            "neutral": "Gray",
            "surprise": "Yellow"
        }

        # Replace 'model_name'
        # model_name = "google/gemma-2-2b-it"

        # try:
        #     self.model = AutoModelForCausalLM.from_pretrained(model_name)
        #     self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # except Exception as e:
        #     print("Error loading model:", e)
        #     print("Exiting...")
        #     exit()

    def answer(self, prompt, image_base64):
        if not prompt:
            return None, None
        
        # Check if "read my emotion" is in the prompt
        if "read my emotion" in prompt.lower():
            if not image_base64:
                return "No image provided for emotion detection.", "Unknown"
            # Decode image
            try:
                image_data = base64.b64decode(image_base64)
                image = Image.open(io.BytesIO(image_data))
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print("Error decoding image:", e)
                return "Failed to decode image.", "Unknown"

            # Detect emotion
            emotion_color = self.detect_emotion(frame)
            emotion_response = f"I detect that you are feeling {emotion_color['emotion']}. The color associated with this emotion is {emotion_color['color']}."
            print("Emotion Response:", emotion_response)
            return emotion_response, emotion_color['color']
        else:
            # For other prompts, directly use the prompt with the LLM
            response = self._generate_response(prompt)
            print("Response:", response)

            # Return color as "Unknown" since emotion detection isn't performed
            return response, "Unknown"
            
    def detect_emotion(self, frame):
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            print("DeepFace Result:", result)

            if isinstance(result, list):
                result = result[0]  # Get the first dictionary from the list

            emotion = result['dominant_emotion'].lower()
            print("Detected Emotion:", emotion)

            color = self.emotion_color_map.get(emotion, "Unknown")
            print("Mapped Color:", color)

            return {"emotion": emotion.capitalize(), "color": color}
        except Exception as e:
            print("Error in emotion detection:", e)
            return {"emotion": "Unknown", "color": "Unknown"}

    def _generate_response(self, prompt):
        # ollama_path = "/usr/local/bin/ollama" # to use in the local environment
        ollama_path = "/home/linuxbrew/.linuxbrew/bin/ollama"
        try:
            result = subprocess.run(
                [ollama_path, "run", "llama3.2:1b"],
                input=prompt,
                capture_output=True,
                text=True
            )
            response = result.stdout.strip()
            if result.returncode != 0:
                print("LLM Error:", result.stderr)
                return "I'm sorry, I'm unable to process your request at the moment."
            return response
        except Exception as e:
            print("Error calling LLaMA model:", e)
            return "I'm sorry, I'm unable to process your request at the moment."

assistant = Assistant()
user_api_counts = {}

@app.route('/process', methods=['POST'])
def process_request():
    data = request.get_json()
    user_id = data.get('user_id')
    prompt = data.get('text')
    image_base64 = data.get('image')
    
    if not user_id:
        return jsonify({'error': 'User ID not provided.'}), 400
    if not prompt:
        return jsonify({'error': 'No text provided.'}), 400

    # Update API call count for the user
    conn = sqlite3.connect('api_counts.db')
    cursor = conn.cursor()

    cursor.execute('SELECT api_count FROM api_counts WHERE user_id = ?', (user_id,))
    result = cursor.fetchone()

    if result:
        api_count = result[0] + 1
        cursor.execute('UPDATE api_counts SET api_count = ? WHERE user_id = ?', (api_count, user_id))
    else:
        api_count = 1
        cursor.execute('INSERT INTO api_counts (user_id, api_count) VALUES (?, ?)', (user_id, api_count))
    
    conn.commit()
    conn.close()

    # Check if user has exceeded free API calls
    if api_count > 20:
        max_reached = True
    else:
        max_reached = False

    response_text, color = assistant.answer(prompt, image_base64)

    return jsonify({
        'response': response_text,
        'color': color,
        'api_count': api_count,
        'max_reached': max_reached
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8282)
