import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini API configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Single iteration workflow - no need for MAX_ITERATIONS

# Learning preferences
LEARNING_FORMATS = ["video", "text", "audio"]

# Model configuration
MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.7
MAX_TOKENS = 4000 