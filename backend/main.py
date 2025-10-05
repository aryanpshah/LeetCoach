import sounddevice as sd
from scipy.io.wavfile import write
import whisper
from google import genai
import json
import regex as re
import mediapipe as mp
import csv
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Annotated
import json
import re
import os
import io
import cv2

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.vercel.app", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health():
    return {"ok": True}

"""
def extract_landmarks(results, frame_idx):
    data = []
    for landmark_type, landmarks in {
        'pose': results.pose_landmarks,
        'face': results.face_landmarks,
        'left_hand': results.left_hand_landmarks,
        'right_hand': results.right_hand_landmarks
    }.items():
        if landmarks:
            for i, lm in enumerate(landmarks.landmark):
                data.append({
                    'frame': frame_idx,
                    'type': landmark_type,
                    'id': i,
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    'visibility': lm.visibility if hasattr(lm, 'visibility') else None
                })
    return data

def safe_merge(base_df, new_df):
    return base_df.merge(new_df, on='frame', how='left') if 'frame' in new_df.columns else base_df

def infer_sentiment(row):
    def safe(val):
        return 0 if pd.isna(val) or val is None else val

    smile = safe(row.get('smile_curvature'))
    eyebrow = safe(row.get('eyebrow_raise_avg'))
    tilt = safe(row.get('shoulder_tilt'))
    spine = safe(row.get('spine_alignment'))
    speed = safe(row.get('speed'))
    left_speed = safe(row.get('left_hand_speed'))
    right_speed = safe(row.get('right_hand_speed'))
    left_raised = bool(row.get('left_hand_raised', False))
    right_raised = bool(row.get('right_hand_raised', False))

    # Match the same classification logic
    if abs(smile) > 0.015 and abs(eyebrow) > 0.015 and abs(tilt) < 0.03 and abs(spine) < 0.03:
        return 'confident'
    elif abs(smile) < 0.2 and abs(eyebrow) < 0.003 and abs(tilt) > 0.075 and abs(spine) > 0.05:
        return 'tense/closed'
    elif left_raised and right_raised:
        return 'attentive'
    elif left_raised or right_raised:
        return 'engaged'
    elif abs(eyebrow) > 0.0475 and abs(smile) < 0.15:
        return 'surprised'
    elif (abs(left_speed) > 0.045 or abs(right_speed) > 0.045 or abs(speed) > 0.045) and abs(smile) < 0.15:
        return 'excited/anxious'
    elif abs(smile) > 0.08 and abs(eyebrow) < 0.07 and abs(speed) < 0.2:
        return 'friendly'
    else:
        return 'neutral'

# ------------------ POST /video ------------------
"""
"""
@app.post("/video")
def video(
    video: Annotated[UploadFile, File()],
    audio: Annotated[UploadFile, File()] = None,
    code: Annotated[UploadFile, File()] = None,
    problem_id: Annotated[str, Form()] = None,
    duration_sec: Annotated[str, Form()] = None
):
    try:
        # Save video to a temp file
        temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".webm").name
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Initialize MediaPipe
        mp_holistic = mp.solutions.holistic
        all_results = []
        cap = cv2.VideoCapture(temp_video_path)
        frame_idx = 0

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                all_results.append((frame_idx, results))
                frame_idx += 1

        cap.release()
        os.remove(temp_video_path)

        # Export landmarks to CSV
        csv_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['frame', 'type', 'id', 'x', 'y', 'z', 'visibility'])
            writer.writeheader()
            for frame_idx, results in all_results:
                writer.writerows(extract_landmarks(results, frame_idx))

        df = pd.read_csv(csv_path)
        os.remove(csv_path)

        # ------------------ Feature aggregation ------------------
        pose_df = df[df['type'] == 'pose']
        face_df = df[df['type'] == 'face']
        left_hand_df = df[df['type'] == 'left_hand']
        right_hand_df = df[df['type'] == 'right_hand']

        # Example: compute only shoulder tilt as demo
        tilt_data = []
        for frame in pose_df['frame'].unique():
            frame_data = pose_df[pose_df['frame'] == frame]
            left = frame_data[frame_data['id'] == 11]
            right = frame_data[frame_data['id'] == 12]
            if not left.empty and not right.empty:
                dy = left.iloc[0]['y'] - right.iloc[0]['y']
                dx = left.iloc[0]['x'] - right.iloc[0]['x']
                angle = abs(dy / dx) if dx != 0 else 0
                tilt_data.append({'frame': frame, 'shoulder_tilt': angle})
        features_df = pd.DataFrame(tilt_data)

        # Infer sentiment
        features_df['sentiment'] = features_df.apply(infer_sentiment, axis=1)

        # Return summary JSON
        sentiment_counts = features_df['sentiment'].value_counts(normalize=True) * 100
        sentiment_summary = sentiment_counts.reset_index()
        sentiment_summary.columns = ['sentiment', 'score']
        sentiment_summary['score'] = sentiment_summary['score'].round(2)

        return {"status": "success", "sentiment_summary": sentiment_summary.to_dict(orient="records")}

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"status": "error", "detail": str(e)}
"""

@app.post("/video")
def video(
    video: Annotated[UploadFile, File()],
    code: Annotated[UploadFile, File()] = None,
    problem_statement: Annotated[str, Form()] = None,
    duration_sec: Annotated[str, Form()] = None
):

    # Convert List to CSV for data analysis
    # Extract landmarks from MediaPipe results
    def extract_landmarks(results, frame_idx):
        data = []
        for landmark_type, landmarks in {
            'pose': results.pose_landmarks,
            'face': results.face_landmarks,
            'left_hand': results.left_hand_landmarks,
            'right_hand': results.right_hand_landmarks
        }.items():
            if landmarks:
                for i, lm in enumerate(landmarks.landmark):
                    data.append({
                        'frame': frame_idx,
                        'type': landmark_type,
                        'id': i,
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z,
                        'visibility': lm.visibility if hasattr(lm, 'visibility') else None
                    })
        return data

    # ------------------ Pose Metrics ------------------

    def compute_shoulder_tilt(pose_df):
        tilt_data = []
        for frame in pose_df['frame'].unique():
            frame_data = pose_df[pose_df['frame'] == frame]
            left = frame_data[frame_data['id'] == 11]
            right = frame_data[frame_data['id'] == 12]
            if not left.empty and not right.empty:
                dy = left.iloc[0]['y'] - right.iloc[0]['y']
                dx = left.iloc[0]['x'] - right.iloc[0]['x']
                angle = abs(dy / dx) if dx != 0 else 0
                tilt_data.append({'frame': frame, 'shoulder_tilt': angle})
        return pd.DataFrame(tilt_data)

    def compute_spine_alignment(pose_df):
        alignment_data = []
        for frame in pose_df['frame'].unique():
            frame_data = pose_df[pose_df['frame'] == frame]
            nose = frame_data[frame_data['id'] == 0]
            left_hip = frame_data[frame_data['id'] == 23]
            right_hip = frame_data[frame_data['id'] == 24]
            if not nose.empty and not left_hip.empty and not right_hip.empty:
                hip_x = (left_hip.iloc[0]['x'] + right_hip.iloc[0]['x']) / 2
                alignment = abs(nose.iloc[0]['x'] - hip_x)
                alignment_data.append({'frame': frame, 'spine_alignment': alignment})
        return pd.DataFrame(alignment_data)

    def detect_hand_raises(pose_df):
        raise_data = []
        for frame in pose_df['frame'].unique():
            frame_data = pose_df[pose_df['frame'] == frame]
            left_wrist = frame_data[frame_data['id'] == 15]
            right_wrist = frame_data[frame_data['id'] == 16]
            left_shoulder = frame_data[frame_data['id'] == 11]
            right_shoulder = frame_data[frame_data['id'] == 12]
            left_raised = not left_wrist.empty and not left_shoulder.empty and left_wrist.iloc[0]['y'] < left_shoulder.iloc[0]['y']
            right_raised = not right_wrist.empty and not right_shoulder.empty and right_wrist.iloc[0]['y'] < right_shoulder.iloc[0]['y']
            raise_data.append({'frame': frame, 'left_hand_raised': left_raised, 'right_hand_raised': right_raised})
        return pd.DataFrame(raise_data)

    def compute_movement_speed(pose_df, landmark_id):
        speed_data = []
        prev = None
        for frame in sorted(pose_df['frame'].unique()):
            current = pose_df[(pose_df['frame'] == frame) & (pose_df['id'] == landmark_id)]
            if not current.empty and prev is not None:
                dx = current.iloc[0]['x'] - prev['x']
                dy = current.iloc[0]['y'] - prev['y']
                speed = (dx**2 + dy**2)**0.5
                speed_data.append({'frame': frame, 'speed': speed})
            if not current.empty:
                prev = current.iloc[0]
        return pd.DataFrame(speed_data)

    # ------------------ Face Metrics ------------------

    def compute_smile_curvature(face_df):
        smile_data = []
        for frame in face_df['frame'].unique():
            frame_data = face_df[face_df['frame'] == frame]
            left_mouth = frame_data[frame_data['id'] == 61]
            right_mouth = frame_data[frame_data['id'] == 291]
            if not left_mouth.empty and not right_mouth.empty:
                dx = right_mouth.iloc[0]['x'] - left_mouth.iloc[0]['x']
                dy = right_mouth.iloc[0]['y'] - left_mouth.iloc[0]['y']
                curvature = dy / dx if dx != 0 else 0
                smile_data.append({'frame': frame, 'smile_curvature': curvature})
        return pd.DataFrame(smile_data)

    def compute_eyebrow_raise(face_df):
        raise_data = []
        for frame in face_df['frame'].unique():
            frame_data = face_df[face_df['frame'] == frame]
            brow_left = frame_data[frame_data['id'] == 70]
            eye_left = frame_data[frame_data['id'] == 159]
            brow_right = frame_data[frame_data['id'] == 300]
            eye_right = frame_data[frame_data['id'] == 386]
            if not brow_left.empty and not eye_left.empty and not brow_right.empty and not eye_right.empty:
                left_raise = brow_left.iloc[0]['y'] - eye_left.iloc[0]['y']
                right_raise = brow_right.iloc[0]['y'] - eye_right.iloc[0]['y']
                avg_raise = (left_raise + right_raise) / 2
                raise_data.append({'frame': frame, 'eyebrow_raise_avg': avg_raise})
        return pd.DataFrame(raise_data)

    # ------------------ Hand Metrics ------------------

    def compute_hand_raise(hand_df, pose_df, hand_type):
        raise_data = []
        wrist_id = 0
        shoulder_id = 11 if hand_type == 'left_hand' else 12
        for frame in hand_df['frame'].unique():
            hand_frame = hand_df[hand_df['frame'] == frame]
            pose_frame = pose_df[pose_df['frame'] == frame]
            wrist = hand_frame[hand_frame['id'] == wrist_id]
            shoulder = pose_frame[pose_frame['id'] == shoulder_id]
            if not wrist.empty and not shoulder.empty:
                raised = wrist.iloc[0]['y'] < shoulder.iloc[0]['y']
                raise_data.append({'frame': frame, f'{hand_type}_raised': raised})
        return pd.DataFrame(raise_data)

    def compute_hand_speed(hand_df, hand_type):
        speed_data = []
        prev = None
        for frame in sorted(hand_df['frame'].unique()):
            current = hand_df[(hand_df['frame'] == frame) & (hand_df['id'] == 0)]
            if not current.empty and prev is not None:
                dx = current.iloc[0]['x'] - prev['x']
                dy = current.iloc[0]['y'] - prev['y']
                speed = (dx**2 + dy**2)**0.5
                speed_data.append({'frame': frame, f'{hand_type}_speed': speed})
            if not current.empty:
                prev = current.iloc[0]
        return pd.DataFrame(speed_data)

    # Safe Merge Helper
    def safe_merge(base_df, new_df):
        return base_df.merge(new_df, on='frame', how='left') if 'frame' in new_df.columns else base_df

    #Sentiment analysis algo
    def infer_sentiment(row):
        # Handle missing values safely
        def safe(val):
            return 0 if pd.isna(val) or val is None else val

        smile = safe(row.get('smile_curvature'))
        eyebrow = safe(row.get('eyebrow_raise_avg'))
        tilt = safe(row.get('shoulder_tilt'))
        spine = safe(row.get('spine_alignment'))
        speed = safe(row.get('speed'))
        left_speed = safe(row.get('left_hand_speed'))
        right_speed = safe(row.get('right_hand_speed'))
        left_raised = bool(row.get('left_hand_raised', False))
        right_raised = bool(row.get('right_hand_raised', False))

        smile = safe(row.get('smile_curvature'))
        eyebrow = safe(row.get('eyebrow_raise_avg'))
        tilt = safe(row.get('shoulder_tilt'))
        spine = safe(row.get('spine_alignment'))
        speed = safe(row.get('speed'))
        left_speed = safe(row.get('left_hand_speed'))
        right_speed = safe(row.get('right_hand_speed'))
        left_raised = bool(row.get('left_hand_raised', False))
        right_raised = bool(row.get('right_hand_raised', False))

        # Match the same classification logic
        if abs(smile) > 0.015 and abs(eyebrow) > 0.015 and abs(tilt) < 0.03 and abs(spine) < 0.03:
            return 'confident'
        elif abs(smile) < 0.2 and abs(eyebrow) < 0.003 and abs(tilt) > 0.075 and abs(spine) > 0.05:
            return 'tense/closed'
        elif left_raised and right_raised:
            return 'attentive'
        elif left_raised or right_raised:
            return 'engaged'
        elif abs(eyebrow) > 0.0475 and abs(smile) < 0.15:
            return 'surprised'
        elif (abs(left_speed) > 0.045 or abs(right_speed) > 0.045 or abs(speed) > 0.045) and abs(smile) < 0.15:
            return 'excited/anxious'
        elif abs(smile) > 0.08 and abs(eyebrow) < 0.07 and abs(speed) < 0.2:
            return 'friendly'
        else:
            return 'neutral'
        
    # ------------------ MediaPipe Setup ------------------

    # Initialize MediaPipe
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    # Open webcam or video file
    import tempfile
    import shutil

    temp_audio_path = None
    audio_transcript = None
    audio_received = False

    def cleanup_audio():
        nonlocal temp_audio_path
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except OSError:
                pass
        temp_audio_path = None

    # Save uploaded file to a temporary location
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    if audio is not None:
        try:
            audio_suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
            audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix=audio_suffix)
            temp_audio_path = audio_temp.name
            audio_temp.close()
            with open(temp_audio_path, "wb") as audio_buffer:
                shutil.copyfileobj(audio.file, audio_buffer)
            audio_received = True
        except Exception as exc:
            print(f"Audio upload save failed: {exc}")
            cleanup_audio()
            audio_received = False

        if temp_audio_path:
            try:
                model = whisper.load_model("turbo")
                result = model.transcribe(temp_audio_path)
                audio_transcript = (result.get("text") or "").strip()
            except Exception as exc:
                print(f"Audio transcription failed: {exc}")
            finally:
                cleanup_audio()

    # Load video from saved file
    cap = cv2.VideoCapture(temp_video_path)

    # Store results for CSV export
    all_results = []
    frame_idx = 0

    print("before loop")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks
            if results.face_landmarks:
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            # if results.pose_landmarks:
            #     print("Pose", end=" ")
            # if results.face_landmarks:
            #     print("Face", end=" ")
            # if results.left_hand_landmarks:
            #     print("Left Hand", end=" ")
            # if results.right_hand_landmarks:
            #     print("Right Hand", end=" ")
            # print()

            # Save results for export
            all_results.append((frame_idx, results))


            # Show video
            #cv2.imshow('Full Body Detection', image)

            # Save results for export
            all_results.append((frame_idx, results))
            frame_idx += 1
            print(frame_idx)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Export to CSV
    with open('landmarks.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'type', 'id', 'x', 'y', 'z', 'visibility'])
        writer.writeheader()
        for frame_idx, results in all_results:
            writer.writerows(extract_landmarks(results, frame_idx))

    # Load and display the first few rows of the CSV
    df = pd.read_csv('landmarks.csv')
    df.head(100)  # Show first 10 rows

    # Conduct Sentiment Analysis Based on Landmarks 


    # ------------------ Feature Aggregation ------------------

    pose_df = df[df['type'] == 'pose']
    face_df = df[df['type'] == 'face']
    left_hand_df = df[df['type'] == 'left_hand']
    right_hand_df = df[df['type'] == 'right_hand']

    # Pose features
    shoulder_tilt_df = compute_shoulder_tilt(pose_df)
    spine_df = compute_spine_alignment(pose_df)
    hand_raises_df = detect_hand_raises(pose_df)
    speed_df = compute_movement_speed(pose_df, landmark_id=0)

    # Face and hand features
    smile_df = compute_smile_curvature(face_df)
    eyebrow_df = compute_eyebrow_raise(face_df)
    left_raise_df = compute_hand_raise(left_hand_df, pose_df, 'left_hand')
    right_raise_df = compute_hand_raise(right_hand_df, pose_df, 'right_hand')
    left_speed_df = compute_hand_speed(left_hand_df, 'left_hand')
    right_speed_df = compute_hand_speed(right_hand_df, 'right_hand')
    print(shoulder_tilt_df.head())
    print(shoulder_tilt_df.index.names)
    # Merge all features safely
    features_df = shoulder_tilt_df.merge(spine_df, on='frame') \
                                .merge(hand_raises_df, on='frame') \
                                .merge(speed_df, on='frame')

    features_df = safe_merge(features_df, smile_df)
    features_df = safe_merge(features_df, eyebrow_df)
    features_df = safe_merge(features_df, left_raise_df)
    features_df = safe_merge(features_df, right_raise_df)
    features_df = safe_merge(features_df, left_speed_df)
    features_df = safe_merge(features_df, right_speed_df)

    # Final result
    # print(features_df.head(100))

    features_df['sentiment'] = features_df.apply(infer_sentiment, axis=1)
    # print(features_df[['frame', 'sentiment']].head(20))
    # print(features_df['sentiment'].value_counts())

    features_df.to_csv("sentiment_analysis.csv", index=False)

    highlights = features_df[features_df['sentiment'] != 'neutral']
    # print(highlights[['frame', 'sentiment']])

    # Get percentage breakdown from non-neutral frames
    sentiment_counts = highlights['sentiment'].value_counts(normalize=True) * 100

    # Format into a readable table
    sentiment_summary = sentiment_counts.reset_index()
    sentiment_summary.columns = ['sentiment', 'score']
    sentiment_summary['score'] = sentiment_summary['score'].round(2)

    # Display the table
    #print(sentiment_summary.to_string(index=False))

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="You are a LeetCode interview assistant, who gives constructive feedback on the visual body language analysis and provides two scores from 0 to 100 for engagement and confidence based on the following response: " + sentiment_summary.to_string(index=False) + "Return your response as a JSON object using only the categories \"engagement\" and \"confidence\".",
    )
    print(response.text)
    # Convert (parse) JSON string → Python dictionary
    try:
        match = re.search(r'\{.*?\}', response.text, re.DOTALL)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
        else:
            cleanup_audio()
            return JSONResponse(status_code=400, content={"error": "Invalid JSON from Gemini"})
    except Exception as e:
        cleanup_audio()
        return JSONResponse(status_code=500, content={"error": "Failed to parse response"})


    # Now you can use it like a normal Python dict
    #return data

    # --- proceed with rest of your MediaPipe + CV processing logic ---

    cap.release()

    payload = {"status": "success", **data}
    if audio_transcript:
        payload["audio_transcript"] = audio_transcript
    if audio_received:
        payload["audio_received"] = True
    cleanup_audio()
    return payload


@app.get("/audio")
async def audio():
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

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="You are a LeetCode interview assistant, who gives constructive feedback on the user's answer and provides three scores from 0 to 10 for communication, clarity, and accuracy based on the following response: " + result["text"] + "Return your response as a JSON object using only the categories \"general\", \"communication\", \"clarity\", and \"accuracy\". Do not give feedback for each score, just one general one.",
    )
    print(response.text)
    # Convert (parse) JSON string → Python dictionary
    match = re.search(r'\{.*?\}', response.text, re.DOTALL)
    if match:
        json_str = match.group(0)
        data = json.loads(json_str)
        print("Parsed scores:", data)
        return data
    else:
        print("No valid JSON found in response.")

    # Now you can use it like a normal Python dict
    return 0
