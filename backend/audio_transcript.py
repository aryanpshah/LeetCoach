import sounddevice as sd
from scipy.io.wavfile import write
import whisper

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