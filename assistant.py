import base64
import io
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

class Assistant:
    def __init__(self):
        # Initialize CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
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
            # Decode image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Detect emotion
            emotion_color = self.detect_emotion(frame)
            emotion_response = f"I detect that you are feeling {emotion_color['emotion']}. The color associated with this emotion is {emotion_color['color']}."
            print("Emotion Response:", emotion_response)
            return emotion_response, emotion_color['color']
        else:
            # For other prompts
            clip_description = self._get_clip_description(image_base64)
            full_prompt = f"{clip_description}. {prompt}"

            response = self._generate_response(full_prompt)
            print("Response:", response)

            return response, None
            
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
            print("Detected Emotion:", emotion)  # Debugging statement

            return {"emotion": emotion.capitalize(), "color": color}
        except Exception as e:
            print("Error in emotion detection:", e)
            return {"emotion": "Unknown", "color": "Unknown"}

    def _get_clip_description(self, image_base64):
        # Decode the image from base64
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))

        # Process the image with CLIP
        inputs = self.clip_processor(images=image, return_tensors="pt")
        image_features = self.clip_model.get_image_features(**inputs)

        # Placeholder for description generation
        return "Description of the image generated by CLIP."

    def _generate_response(self, full_prompt):
        # Run Ollama command to get a response from the LLaMA model
        result = subprocess.run(
            ["ollama", "run", "llama3.1", "--prompt", full_prompt],
            capture_output=True,
            text=True
        )
        response = result.stdout.strip()
        return response

@app.route('/process', methods=['POST'])
def process_request():
    data = request.get_json()
    prompt = data.get('text')
    image_base64 = data.get('image')

    response_text, color = assistant.answer(prompt, image_base64)

    return jsonify({
        'response': response_text,
        'color': color
    })

if __name__ == '__main__':
    # Initialize your LLM model
    assistant = Assistant()

    # Run the Flask app
    app.run(host='0.0.0.0', port=8888)