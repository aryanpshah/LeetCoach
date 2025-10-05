import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { PROBLEMS } from "../lib/problems";

const BACKEND_URL_VIDEO = "http://127.0.0.1:8000/video";

const buildFallbackReport = (problemId) => ({
  session_id: `demo-${problemId}-${Date.now()}`,
  problem_id: problemId,
  status: "mock",
  meta: {
    duration_sec: 600,
    took_sec: 540
  },
  scores: {
    accuracy: 82,
    communication: 86,
    confidence: 74,
    clarity: 79,
    engagement: 80
  },
  feedback: {
    accuracy: "Solid explanation of the core logic. Mention more edge cases next time.",
    communication: "Structure is strong. Continue narrating data structures as you use them.",
    confidence: "Delivery is calm and clear. Keep eye contact up throughout.",
    clarity: "Good asymptotic reasoning; call out trade-offs explicitly.",
    engagement: "Nice pacing and focus. Maintain that energy during your summary wrap-up."
  }
});

export default function Test() {
  const { id } = useParams();
  const navigate = useNavigate();
  const problem = useMemo(() => PROBLEMS[id], [id]);

  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const recordingStreamRef = useRef(null);
  const chunksRef = useRef([]);
  const audioRecorderRef = useRef(null);
  const audioStreamRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);
  const fileInputRef = useRef(null);

  const [recording, setRecording] = useState(false);
  const [videoBlob, setVideoBlob] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);
  const [permOK, setPermOK] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [hasStarted, setHasStarted] = useState(false);
  const [showTimesUp, setShowTimesUp] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [hintVisible, setHintVisible] = useState(false);

  const timeLimitSec = useMemo(() => (problem?.timeLimitMin ?? 10) * 60, [problem?.timeLimitMin]);
  const [remaining, setRemaining] = useState(timeLimitSec);

  const stopRecording = useCallback(() => {
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
    } else if (recordingStreamRef.current) {
      recordingStreamRef.current.getTracks().forEach((track) => track.stop());
      recordingStreamRef.current = null;
    }

    const audioRecorder = audioRecorderRef.current;
    if (audioRecorder && audioRecorder.state !== "inactive") {
      audioRecorder.stop();
    } else if (audioStreamRef.current) {
      audioStreamRef.current.getTracks().forEach((track) => track.stop());
      audioStreamRef.current = null;
    }

    setRecording(false);
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const handleTimesUp = useCallback(() => {
    stopRecording();
    setShowTimesUp(true);
  }, [stopRecording]);

  const startTimer = useCallback(() => {
    if (timerRef.current) return;
    timerRef.current = setInterval(() => {
      setRemaining((prev) => {
        if (prev <= 1) {
          clearInterval(timerRef.current);
          timerRef.current = null;
          handleTimesUp();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  }, [handleTimesUp]);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        if (!mounted) return;
        setPermOK(true);
        mediaStreamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error(err);
        setErrorMsg("Camera/Microphone permission denied. Please allow access to continue.");
        setPermOK(false);
      }
    })();
    return () => {
      mounted = false;
      stopRecording();
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (audioStreamRef.current) {
        audioStreamRef.current.getTracks().forEach((track) => track.stop());
        audioStreamRef.current = null;
      }
    };
  }, [stopRecording]);

  useEffect(() => {
    setRemaining(timeLimitSec);
    setHasStarted(false);
    setHintVisible(false);
    setShowTimesUp(false);
    setShowUpload(false);
    setVideoBlob(null);
    setAudioBlob(null);
    setErrorMsg("");
    setUploadError("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    stopRecording();
  }, [timeLimitSec, stopRecording]);

  const fmt = (seconds) => {
    const minutes = Math.floor(seconds / 60).toString().padStart(2, "0");
    const secs = (seconds % 60).toString().padStart(2, "0");
    return `${minutes}:${secs}`;
  };

  const selectMimeType = () => {
    if (typeof MediaRecorder === "undefined") return null;
    const candidates = ["video/webm;codecs=vp8,opus", "video/webm;codecs=vp9,opus", "video/webm"];
    return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || null;
  };

  const selectAudioMimeType = () => {
    if (typeof MediaRecorder === "undefined") return null;
    const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus"];
    return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || null;
  };



  const handleStart = async () => {
    setErrorMsg("");
    setUploadError("");
    if (!permOK || !mediaStreamRef.current) {
      setErrorMsg("Cannot start. Camera/Mic not available.");
      return;
    }
    try {
      setHasStarted(true);
      setShowTimesUp(false);
      setShowUpload(false);
      setHintVisible(false);
      setVideoBlob(null);
      setAudioBlob(null);
      setRemaining(timeLimitSec);

      const previewStream = mediaStreamRef.current;
      const recordingStream = new MediaStream(
        previewStream.getTracks().map((track) => track.clone())
      );
      recordingStreamRef.current = recordingStream;

      const audioTracks = previewStream.getAudioTracks();
      let audioRecorder = null;
      let audioMimeType = null;
      let resolvedAudioType = "audio/webm";
      audioChunksRef.current = [];

      if (audioTracks.length > 0) {
        audioMimeType = selectAudioMimeType();
        const audioStream = new MediaStream(audioTracks.map((track) => track.clone()));
        audioStreamRef.current = audioStream;
        try {
          audioRecorder = audioMimeType
            ? new MediaRecorder(audioStream, { mimeType: audioMimeType })
            : new MediaRecorder(audioStream);
          audioRecorderRef.current = audioRecorder;
        } catch (err) {
          console.error(err);
          setErrorMsg("Audio recording could not start. Please check microphone permissions.");
          audioStream.getTracks().forEach((track) => track.stop());
          audioStreamRef.current = null;
          audioRecorderRef.current = null;
        }
      } else {
        audioStreamRef.current = null;
        audioRecorderRef.current = null;
      }

      if (audioRecorder) {
        resolvedAudioType =
          (audioRecorder.mimeType && audioRecorder.mimeType.length > 0
            ? audioRecorder.mimeType
            : audioMimeType) || "audio/webm";
        audioRecorder.ondataavailable = (event) => {
          if (event.data && event.data.size > 0) {
            audioChunksRef.current.push(event.data);
          }
        };
        audioRecorder.onstop = () => {
          if (audioStreamRef.current) {
            audioStreamRef.current.getTracks().forEach((track) => track.stop());
            audioStreamRef.current = null;
          }
          if (audioChunksRef.current.length > 0) {
            setAudioBlob(new Blob(audioChunksRef.current, { type: resolvedAudioType }));
          } else {
            setAudioBlob(null);
          }
          audioRecorderRef.current = null;
        };
        audioRecorder.onerror = (event) => {
          console.error(event.error || event);
          setErrorMsg("Audio recording error occurred. Please try again.");
        };
      } else {
        setAudioBlob(null);
      }

      const mimeType = selectMimeType();
      chunksRef.current = [];
      const recorder = mimeType
        ? new MediaRecorder(recordingStream, { mimeType })
        : new MediaRecorder(recordingStream);
      mediaRecorderRef.current = recorder;

      recorder.onstart = () => {
        setRecording(true);
        startTimer();
      };

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "video/webm" });
        setVideoBlob(blob);
        if (recordingStreamRef.current) {
          recordingStreamRef.current.getTracks().forEach((track) => track.stop());
          recordingStreamRef.current = null;
        }
      };

      recorder.onerror = (event) => {
        console.error(event.error || event);
        setErrorMsg("Recording error occurred. Please try again.");
        setRecording(false);
      };

      if (videoRef.current) {
        if (videoRef.current.srcObject !== previewStream) {
          videoRef.current.srcObject = previewStream;
        }
        const playPromise = videoRef.current.play?.();
        if (playPromise && typeof playPromise.then === "function") {
          await playPromise.catch(() => {});
        }
      }

      if (audioRecorder) {
        try {
          audioRecorder.start(1000);
        } catch (err) {
          console.error(err);
        }
      }

      recorder.start(500);
    } catch (err) {
      console.error(err);
      setErrorMsg("Failed to start recording.");
      setRecording(false);
      if (recordingStreamRef.current) {
        recordingStreamRef.current.getTracks().forEach((track) => track.stop());
        recordingStreamRef.current = null;
      }
      if (audioStreamRef.current) {
        audioStreamRef.current.getTracks().forEach((track) => track.stop());
        audioStreamRef.current = null;
      }
      audioRecorderRef.current = null;
    }
  };

  const handleStop = () => {
    stopRecording();
  };

  const handleSubmitEarly = () => {
    stopRecording();
    setShowUpload(true);
  };

  const proceedFromTimesUp = () => {
    setShowTimesUp(false);
    setShowUpload(true);
  };

  const postToBackend = async () => {
    setUploadError("");
    if (!problem) {
      setUploadError("Problem metadata missing. Refresh and try again.");
      return;
    }
    if (!videoBlob) {
      setUploadError("No recorded video found. Please try again.");
      return;
    }
    if (!audioBlob) {
      setUploadError("No recorded audio found. Please re-record with microphone enabled.");
      return;
    }
    const codeFile = fileInputRef.current?.files?.[0];
    if (!codeFile) {
      setUploadError("Please choose a code file before submitting.");
      return;
    }
    try {
      setUploading(true);
      const payload = new FormData();
      payload.append("video", videoBlob, `${id}.webm`);
      payload.append("audio", audioBlob, `${id}-audio.webm`);
      payload.append("code", codeFile, codeFile.name);
      payload.append("problem_statement", problem.statement);
      payload.append("duration_sec", String(timeLimitSec));

      const response = await fetch(BACKEND_URL_VIDEO, {
        method: "POST",
        body: payload
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      const sessionId = data.session_id || String(Date.now());
      setShowUpload(false);
      navigate(`/report/${sessionId}`, { state: { report: data } });
    } catch (err) {
      console.error(err);
      if (err instanceof TypeError) {
        const fallbackId = problem?.id || id || "unknown";
        const mockReport = buildFallbackReport(fallbackId);
        setUploadError("");
        setShowUpload(false);
        navigate(`/report/${mockReport.session_id}`, { state: { report: mockReport, isMock: true } });
      } else {
        setUploadError("Upload failed. Check your connection and try again.");
      }
    } finally {
      setUploading(false);
    }
  };

  if (!problem) {
    return <div className="container py-5 text-white">Problem not found.</div>;
  }

  return (
    <main className="container-fluid py-4">
      <div className="row g-4 test-row">
        <div className="col-12 col-lg-6 d-flex">
          <div className="card bg-dark text-light border border-secondary shadow-sm test-card flex-fill">
            <div className="card-body">
              <div className="d-flex align-items-center justify-content-between">
                <h5 className="mb-0">{problem.title}</h5>
                <span
                  className={`badge ${
                    problem.difficulty === "Easy"
                      ? "bg-success"
                      : problem.difficulty === "Medium"
                        ? "bg-warning"
                        : "bg-danger"
                  }`}
                >
                  {problem.difficulty}
                </span>
              </div>
              <div className="small text-secondary">
                {problem.tags.join(" / ")} | Time Limit: {problem.timeLimitMin} min
              </div>

              <div className="test-card__content">
                <div className={`locked-block problem-block ${hasStarted ? "" : "locked-block--blur"}`}>
                  <div className="locked-content">
                    <pre className="problem-pre">{problem.statement.trim()}</pre>
                    <hr className="border-secondary my-3" />
                    <div>
                      <div className="mb-2">
                        <strong>Example</strong>
                      </div>
                      <div className="codeish mb-2">
                        <strong>Input:</strong> {problem.io.input}
                      </div>
                      <div className="codeish mb-0">
                        <strong>Output:</strong> {problem.io.output}
                      </div>
                    </div>
                  </div>
                  {!hasStarted && (
                    <div className="locked-overlay">
                      <div>
                        Press <strong>Start</strong> to reveal the full problem.
                      </div>
                      <span>Timer will begin and the recording will start.</span>
                    </div>
                  )}
                </div>

                <div
                  className={`locked-block hint-block ${
                    hintVisible ? "hint-block--revealed" : "locked-block--blur"
                  }`}
                >
                  <div className="locked-content">
                    <strong>Hint:</strong> {problem.hint}
                  </div>
                  {!hintVisible &&
                    (hasStarted ? (
                      <button
                        type="button"
                        className="locked-overlay locked-overlay--cta"
                        onClick={() => setHintVisible(true)}
                      >
                        Click to reveal hint
                      </button>
                    ) : (
                      <div className="locked-overlay locked-overlay--info">
                        Start recording to reveal the hint
                      </div>
                    ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="col-12 col-lg-6 d-flex">
          <div className="card bg-dark text-light border border-secondary shadow-sm test-card flex-fill">
            <div className="card-body d-flex flex-column gap-3">
              <div className="d-flex justify-content-between align-items-center">
                <div className="fs-5">
                  Timer: <span className="fw-bold">{fmt(remaining)}</span>
                </div>
                <div className="small text-secondary">{recording ? "Recording..." : "Idle"}</div>
              </div>

              <div className="ratio ratio-16x9 border border-secondary rounded">
                <video ref={videoRef} className="w-100 h-100" autoPlay playsInline muted />
              </div>

              {errorMsg && <div className="alert alert-danger py-2">{errorMsg}</div>}

              <div className="d-flex gap-2">
                <button className="btn btn-success" onClick={handleStart} disabled={recording || !permOK}>
                  Start
                </button>
                <button className="btn btn-warning" onClick={handleStop} disabled={!recording}>
                  Stop
                </button>
                <button
                  className="btn btn-primary ms-auto"
                  onClick={handleSubmitEarly}
                  disabled={recording && remaining > 0}
                  title="Submit early when you are done"
                >
                  Submit
                </button>
              </div>

              <div className="form-text">
                Start begins timer and reveals the problem. Stop ends the recording. Submit opens the code upload dialog.
              </div>
            </div>
          </div>
        </div>
      </div>

      {showTimesUp && (
        <div className="modal-backdrop-simple">
          <div className="modal-card">
            <h5 className="mb-2">Time's up!</h5>
            <p className="mb-3">Your recording ended because the timer reached zero.</p>
            <div className="d-flex justify-content-end gap-2">
              <button className="btn btn-primary" onClick={proceedFromTimesUp}>
                OK
              </button>
            </div>
          </div>
        </div>
      )}

      {showUpload && (
        <div className="modal-backdrop-simple">
          <div className="modal-card">
            <h5 className="mb-2">Upload your code</h5>
            <p className="mb-3">
              Attach the file you wrote for this problem. We accept .py, .js, .cpp, .java and similar.
            </p>
            <input
              ref={fileInputRef}
              type="file"
              className="form-control mb-3"
              accept=".py,.js,.ts,.cpp,.cc,.cxx,.c,.java,.cs,.go,.rb"
            />
            {uploadError && <div className="alert alert-danger py-2">{uploadError}</div>}
            <div className="d-flex justify-content-end gap-2">
              <button className="btn btn-secondary" onClick={() => setShowUpload(false)} disabled={uploading}>
                Cancel
              </button>
              <button className="btn btn-success" onClick={postToBackend} disabled={uploading}>
                {uploading ? "Uploading..." : "Submit"}
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}









