from fastapi import FastAPI
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
from google import genai
import json

app = FastAPI()

@app.get("/audio")
def audio():
    # Parameters
    fs = 44100  # Sample rate (samples per second)
    seconds = 5  # Duration of recording

    print("Recording...")
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")

    # Save as WAV file
    write("output.wav", fs, recording)
    print("Saved as output.wav")

    model = whisper.load_model("turbo")
    result = model.transcribe("output.wav")
    print(result["text"])

    client = genai.Client(api_key="AIzaSyBTyeY_i71cCx8CHNoN1uX9jvtHKS9VthA")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="You are a LeetCode interview assistant, who gives constructive feedback on the user's answer and provides three scores from 0 to 10 for communication, clarity, and accuracy based on the following response: " + result["text"] + "Return your response as a JSON object using only the categories \"general\", \"communication\", \"clarity\", and \"accuracy\".",
    )
    print(response.text)
    # Convert (parse) JSON string → Python dictionary
    data = json.loads(response.text)

    # Now you can use it like a normal Python dict
    return data