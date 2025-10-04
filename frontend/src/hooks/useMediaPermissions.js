import { useCallback, useMemo, useState } from "react";

const DEFAULT_CONSTRAINTS = Object.freeze({ video: true, audio: true });

function describeMediaError(error) {
  if (!error) {
    return "Camera and microphone access was not granted.";
  }

  switch (error.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Camera and microphone access was denied. Please allow both permissions to continue.";
    case "NotFoundError":
      return "No usable camera or microphone was detected.";
    case "NotReadableError":
      return "Your camera or microphone is already in use by another application.";
    case "OverconstrainedError":
      return "The requested media constraints could not be satisfied by your hardware.";
    default:
      return "Unable to access camera and microphone. Please check your browser permissions.";
  }
}

export function useMediaPermissions(initialConstraints = DEFAULT_CONSTRAINTS) {
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState(null);
  const constraints = useMemo(() => ({ ...DEFAULT_CONSTRAINTS, ...initialConstraints }), [initialConstraints]);

  const requestPermissions = useCallback(async () => {
    setStatus("pending");
    setError(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      stream.getTracks().forEach((track) => track.stop());
      setStatus("granted");
      return true;
    } catch (err) {
      const friendlyMessage = describeMediaError(err);
      console.warn(friendlyMessage);
      setStatus("denied");
      setError(friendlyMessage);
      return false;
    }
  }, [constraints]);

  return {
    status,
    error,
    constraints,
    requestPermissions,
  };
}