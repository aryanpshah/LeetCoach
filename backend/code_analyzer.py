import mediapipe as mp
import csv
import pandas as pd

from fastapi import FastAPI, UploadFile, File, Form
from typing import Annotated
from google import genai
import json
import re

app = FastAPI()

# Optional: Enable CORS if you're calling from a frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

@app.post("/analyze")
async def analyze_solution(question: str = Form(...), file: UploadFile = Form(...)):
    # Read the uploaded Python file
    code_bytes = await file.read()
    code_str = code_bytes.decode("utf-8")

    # Prepare prompt for Gemini
    client = genai.Client(api_key="AIzaSyBTyeY_i71cCx8CHNoN1uX9jvtHKS9VthA")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="You are a LeetCode interview assistant, who gives constructive feedback on leetcode coding solution and provides three scores from 0 to 100 for Accuracy, Efficiency, and Code Quality based on the following response: " + question + file.toString() + "Return your response as a JSON object using only the categories \"Accuracy\", \"Efficiency\", and \"Code Quality\".",
    )
    print(response.text)
    # Convert (parse) JSON string → Python dictionary
    match = re.search(r'\{.*?\}', response.text, re.DOTALL)
    if match:
        json_str = match.group(0)
        data = json.loads(json_str)
        print("Parsed scores:", data)
    else:
        print("No valid JSON found in response.")

    return {status: "success", "data": match}
