import base64
import io
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

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
            # Use DeepFace to analyze the emotion
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            print("DeepFace Result:", result)  # Debugging statement

            # Check if result is a list
            if isinstance(result, list):
                result = result[0]  # Get the first dictionary from the list

            emotion = result['dominant_emotion'].lower()
            print("Detected Emotion:", emotion)  # Debugging statement

            # Map the emotion to a color
            color = self.emotion_color_map.get(emotion, "Unknown")
            print("Mapped Color:", color)  # Debugging statement

            return {"emotion": emotion.capitalize(), "color": color}
        except Exception as e:
            print("Error in emotion detection:", e)
            return {"emotion": "Unknown", "color": "Unknown"}

    def _generate_response(self, prompt):
        # Run Ollama command to get a response from the LLaMA model
        ollama_path = "/usr/local/bin/ollama"
        try:
            result = subprocess.run(
                [ollama_path, "run", "llama3.1"],
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
user_api_counts = {}  # Dictionary to track API counts per user

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
    user_api_counts[user_id] = user_api_counts.get(user_id, 0) + 1
    count = user_api_counts[user_id]

    # Check if user has exceeded free API calls
    if count > 20:
        max_reached = True
    else:
        max_reached = False

    # Process the request
    response_text, color = assistant.answer(prompt, image_base64)

    return jsonify({
        'response': response_text,
        'color': color,
        'api_count': count,
        'max_reached': max_reached
    })

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=8888)

