import { useLocation, useNavigate, useParams } from "react-router-dom";
import { useMemo } from "react";
import { PROBLEMS } from "../lib/problems";

const CAT_ORDER = ["accuracy", "communication", "confidence", "efficiency", "engagement"];

const LABELS = {
  accuracy: "Accuracy",
  communication: "Communication",
  confidence: "Confidence",
  efficiency: "Efficiency",
  engagement: "Engagement"
};

const HELP = {
  accuracy: "How correct your approach and edge-case handling were.",
  communication: "Clarity and structure explaining approach, constraints, and tests.",
  confidence: "Delivery presence and steadiness in voice and posture.",
  efficiency: "Time/space complexity and quality of tradeoffs.",
  engagement: "Eye contact, pacing, and responsiveness to the prompt."
};

const KEY_PATTERNS = [
  { key: "accuracy", tokens: ["accuracy", "correct", "quality", "tests", "coverage"] },
  { key: "communication", tokens: ["communication", "clarity", "speech", "verbal", "explanation"] },
  { key: "confidence", tokens: ["confidence", "presence", "poise", "steadiness"] },
  { key: "efficiency", tokens: ["efficiency", "complexity", "runtime", "speed", "performance"] },
  { key: "engagement", tokens: ["engagement", "energy", "attentive", "professionalism", "professional", "eye contact", "posture"] }
];

const DEFAULT_FEEDBACK = {
  accuracy: "We could not evaluate code accuracy for this run. Re-upload your solution to unlock this score.",
  communication: "Speech analysis data was limited. Ensure the microphone is enabled so we can grade communication.",
  confidence: "Camera cues were inconclusive, so confidence could not be fully assessed this time.",
  efficiency: "Runtime and complexity checks are pending. Submit code for automated efficiency feedback.",
  engagement: "We need more consistent video cues to assess engagement. Keep your camera centered and well lit next time."
};

const parseScore = (value) => {
  if (value === null || value === undefined) return null;
  const num = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(num)) return null;
  const normalized = num <= 10 ? num * 10 : num;
  return Math.max(0, Math.min(100, Math.round(normalized)));
};

const canonicalKey = (key) => {
  if (!key) return null;
  let normalized = String(key).toLowerCase().trim();
  normalized = normalized.replace(/[_\s-]*(score|value|percent|pct|feedback|note|comment)$/g, "");
  normalized = normalized.replace(/\s+/g, " ");
  for (const { key: canonical, tokens } of KEY_PATTERNS) {
    if (tokens.some((token) => normalized.includes(token))) {
      return canonical;
    }
  }
  return null;
};

const extractEntries = (source) => {
  if (!source) return [];
  if (Array.isArray(source)) {
    return source.flatMap((item) => {
      if (!item) return [];
      if (typeof item === "string") {
        return [["general", item]];
      }
      if (typeof item !== "object") return [];
      const entries = [];
      const key =
        item.key ??
        item.metric ??
        item.category ??
        item.label ??
        item.name ??
        item.title ??
        item.sentiment;
      if (key !== undefined) {
        if (item.score !== undefined) entries.push([key, item.score]);
        else if (item.value !== undefined) entries.push([key, item.value]);
        else if (item.percent !== undefined) entries.push([key, item.percent]);
        if (item.feedback !== undefined) entries.push([`${key}_feedback`, item.feedback]);
        if (item.note !== undefined) entries.push([`${key}_note`, item.note]);
      }
      return entries;
    });
  }
  if (typeof source === "object") {
    return Object.entries(source);
  }
  return [];
};

const deriveFromSentiment = (summary) => {
  const derivedScores = {};
  const derivedFeedback = {};
  if (!Array.isArray(summary)) {
    return { scores: derivedScores, feedback: derivedFeedback };
  }

  const sentimentMap = summary.reduce((acc, entry) => {
    if (!entry) return acc;
    const label = (entry.sentiment ?? entry.label ?? entry.state ?? entry.category ?? "").toLowerCase();
    const score = parseScore(entry.score ?? entry.value ?? entry.percent);
    if (!label || score === null) return acc;
    acc[label] = score;
    return acc;
  }, {});

  if (sentimentMap.confident !== undefined) {
    derivedScores.confidence = sentimentMap.confident;
    derivedFeedback.confidence = `Confident posture detected in ${sentimentMap.confident}% of the recording.`;
  }

  const engagedPercent = (sentimentMap.engaged ?? 0) + (sentimentMap.attentive ?? 0);
  if (engagedPercent > 0) {
    const capped = Math.min(100, engagedPercent);
    derivedScores.engagement = parseScore(capped);
    const parts = [];
    if (sentimentMap.attentive) parts.push(`${sentimentMap.attentive}% attentive frames`);
    if (sentimentMap.engaged) parts.push(`${sentimentMap.engaged}% engaged gestures`);
    derivedFeedback.engagement = `Engagement cues tracked: ${parts.join(" + ")}.`;
  }

  if (sentimentMap.friendly !== undefined) {
    derivedScores.communication = sentimentMap.friendly;
    derivedFeedback.communication = `Friendly facial cues appeared in ${sentimentMap.friendly}% of frames.`;
  } else if (sentimentMap["excited/anxious"] !== undefined) {
    const anxious = sentimentMap["excited/anxious"];
    derivedFeedback.communication = `Video detected ${anxious}% moments of rapid or anxious delivery. Slow down to articulate each step.`;
    const adjusted = Math.max(0, 100 - anxious);
    derivedScores.communication = parseScore(adjusted);
  }

  return { scores: derivedScores, feedback: derivedFeedback };
};

const deriveAudio = (audio) => {
  const derivedScores = {};
  const derivedFeedback = {};
  extractEntries(audio).forEach(([key, value]) => {
    if (value === null || value === undefined) return;
    const lower = String(key).toLowerCase();
    if (typeof value === "string") {
      const trimmed = value.trim();
      if (!trimmed) return;
      if (lower === "general" && !derivedFeedback.communication) {
        derivedFeedback.communication = trimmed;
        return;
      }
      const canonical = canonicalKey(key);
      if (!canonical) return;
      if (!derivedFeedback[canonical]) {
        derivedFeedback[canonical] = trimmed;
      }
      return;
    }

    const canonical = canonicalKey(key);
    if (!canonical) return;
    const parsed = parseScore(value);
    if (parsed === null) return;
    derivedScores[canonical] = parsed;
  });
  return { scores: derivedScores, feedback: derivedFeedback };
};

const deriveCode = (code) => {
  const derivedScores = {};
  const derivedFeedback = {};
  extractEntries(code).forEach(([key, value]) => {
    if (value === null || value === undefined) return;
    if (typeof value === "string") {
      const trimmed = value.trim();
      if (!trimmed) return;
      const canonical = canonicalKey(key);
      if (!canonical) return;
      if (!derivedFeedback[canonical]) {
        derivedFeedback[canonical] = trimmed;
      }
      return;
    }
    const canonical = canonicalKey(key);
    if (!canonical) return;
    const parsed = parseScore(value);
    if (parsed === null) return;
    derivedScores[canonical] = parsed;
  });
  return { scores: derivedScores, feedback: derivedFeedback };
};

const normalizeReport = (rawReport, fallbackSessionId, fallbackProblemId) => {
  if (!rawReport) return null;

  const scores = {};
  const feedback = {};

  const registerEntry = (key, value) => {
    if (value === null || value === undefined || key === null || key === undefined) return;
    const canonical = canonicalKey(key);
    const numeric = canonical ? parseScore(value) : null;
    if (canonical && numeric !== null) {
      if (scores[canonical] === undefined) {
        scores[canonical] = numeric;
      } else {
        scores[canonical] = Math.round((scores[canonical] + numeric) / 2);
      }
      return;
    }

    if (typeof value === "string") {
      const trimmed = value.trim();
      if (!trimmed) return;
      const lower = String(key).toLowerCase();
      if (lower === "general" && !feedback.communication) {
        feedback.communication = trimmed;
        return;
      }
      if (canonical && !feedback[canonical]) {
        feedback[canonical] = trimmed;
      }
    }
  };

  const scoreSources = [
    rawReport.scores,
    rawReport.scorecard,
    rawReport.metrics,
    rawReport.analysis?.scores,
    rawReport.results?.scores,
    rawReport.ai_scores,
    rawReport.video_scores,
    rawReport.audio_scores,
    rawReport.code_scores
  ];

  scoreSources.forEach((source) => {
    extractEntries(source).forEach(([key, value]) => registerEntry(key, value));
  });

  CAT_ORDER.forEach((key) => {
    if (rawReport[key] !== undefined) {
      registerEntry(key, rawReport[key]);
    }
  });

  const { scores: sentimentScores, feedback: sentimentFeedback } = deriveFromSentiment(rawReport.sentiment_summary);
  Object.entries(sentimentScores).forEach(([key, value]) => registerEntry(key, value));
  Object.entries(sentimentFeedback).forEach(([key, value]) => {
    if (!feedback[key]) feedback[key] = value;
  });

  const { scores: audioScores, feedback: audioFeedback } = deriveAudio(
    rawReport.audio_analysis ?? rawReport.audio ?? rawReport.speech
  );
  Object.entries(audioScores).forEach(([key, value]) => registerEntry(key, value));
  Object.entries(audioFeedback).forEach(([key, value]) => {
    if (!feedback[key]) feedback[key] = value;
  });

  const { scores: codeScores, feedback: codeFeedback } = deriveCode(
    rawReport.code_analysis ?? rawReport.code ?? rawReport.compiler
  );
  Object.entries(codeScores).forEach(([key, value]) => registerEntry(key, value));
  Object.entries(codeFeedback).forEach(([key, value]) => {
    if (!feedback[key]) feedback[key] = value;
  });

  const feedbackSources = [
    rawReport.feedback,
    rawReport.analysis?.feedback,
    rawReport.results?.feedback,
    rawReport.text_feedback,
    rawReport.comments,
    rawReport.notes,
    rawReport.coach_notes,
    rawReport.audio_feedback,
    rawReport.video_feedback,
    rawReport.code_feedback
  ];

  feedbackSources.forEach((source) => {
    extractEntries(source).forEach(([key, value]) => registerEntry(key, value));
  });

  const normalizedScores = {};
  const normalizedFeedback = {};
  const missing = [];

  CAT_ORDER.forEach((key) => {
    normalizedScores[key] = scores[key] ?? null;
    normalizedFeedback[key] = feedback[key] ?? DEFAULT_FEEDBACK[key];
    if (normalizedScores[key] === null) {
      missing.push(key);
    }
  });

  const meta = { ...(rawReport.meta || {}) };
  const coerceNumber = (value) => {
    const num = Number(value);
    return Number.isFinite(num) ? num : undefined;
  };
  if (meta.took_sec === undefined) {
    const candidate = coerceNumber(
      rawReport.took_sec ?? rawReport.elapsed_sec ?? rawReport.elapsed ?? rawReport.duration
    );
    if (candidate !== undefined) meta.took_sec = candidate;
  }
  if (meta.duration_sec === undefined) {
    const candidate = coerceNumber(
      rawReport.duration_sec ?? rawReport.allowed_sec ?? rawReport.time_allocated
    );
    if (candidate !== undefined) meta.duration_sec = candidate;
  }

  const normalizedProblemId =
    rawReport.problem_id ?? rawReport.problemId ?? fallbackProblemId ?? null;
  const normalizedSessionId =
    rawReport.session_id ?? rawReport.sessionId ?? fallbackSessionId ?? `session-${Date.now()}`;

  return {
    ...rawReport,
    session_id: normalizedSessionId,
    problem_id: normalizedProblemId,
    meta,
    scores: normalizedScores,
    feedback: normalizedFeedback,
    missing
  };
};

export default function Report() {
  const { state } = useLocation();
  const { sessionId } = useParams();
  const navigate = useNavigate();

  const rawReport = state?.report;
  const fallbackProblemId = rawReport?.problem_id ?? rawReport?.problemId ?? state?.problemId ?? null;

  const normalized = useMemo(
    () => normalizeReport(rawReport, sessionId, fallbackProblemId),
    [rawReport, sessionId, fallbackProblemId]
  );

  if (!normalized) {
    return (
      <main className="container py-5 text-white">
        <h1 className="h4">Report unavailable</h1>
        <p className="text-secondary">
          We could not find a report in memory. Please run a new session or implement a fetch by
          <code className="ms-1">sessionId</code>: <strong>{sessionId}</strong>.
        </p>
        <button className="btn btn-primary" onClick={() => navigate("/select")}>
          Try Another Problem
        </button>
      </main>
    );
  }

  const problem = useMemo(() => PROBLEMS[normalized.problem_id ?? ""], [normalized.problem_id]);
  const tookSec = Number(normalized.meta?.took_sec ?? 0);
  const mins = Number.isFinite(tookSec) ? Math.floor(tookSec / 60) : 0;
  const secs = Number.isFinite(tookSec) ? tookSec % 60 : 0;

  const overall = useMemo(() => {
    const { sum, count } = CAT_ORDER.reduce(
      (acc, key) => {
        const value = normalized.scores[key];
        if (value === null) return acc;
        return { sum: acc.sum + value, count: acc.count + 1 };
      },
      { sum: 0, count: 0 }
    );
    if (count === 0) return null;
    return Math.round(sum / count);
  }, [normalized.scores]);

  const missingLabels = normalized.missing.map((key) => LABELS[key]);

  const downloadJSON = () => {
    const blob = new Blob([JSON.stringify(normalized, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `leetcoach-report-${normalized.session_id}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main className="container py-4 report-page">
      {normalized.status === "error" && (
        <div className="alert alert-danger d-flex align-items-center" role="alert">
          <div>
            <strong className="me-2">Analysis failed.</strong>
            {normalized.detail || "Please retry your session once the backend is available."}
          </div>
        </div>
      )}

      {missingLabels.length > 0 && (
        <div className="alert alert-warning" role="alert">
          Missing metrics: {missingLabels.join(", ")}. We will populate these once every analyzer returns data.
        </div>
      )}

      <div className="d-flex flex-wrap align-items-center justify-content-between gap-3 mb-4">
        <div>
          <h1 className="h3 text-white mb-1">Your Report</h1>
          <div className="text-secondary small">
            Problem: <span className="text-white">{problem?.title || normalized.problem_id || "Unknown"}</span>
            {"  \u2022  "}Elapsed: <span className="text-white">{mins}m {String(secs).padStart(2, "0")}s</span>
            {"  \u2022  "}Session: <span className="text-white">{normalized.session_id}</span>
          </div>
        </div>
        <div className="overall-badge d-flex align-items-center">
          <div className="text-secondary small me-2">Overall</div>
          <div className="overall-pill">{overall ?? "\u2014"}</div>
        </div>
      </div>

      <div className="row g-4">
        {CAT_ORDER.map((key) => {
          const score = normalized.scores[key];
          const displayScore = score === null ? "\u2014" : score;
          const barWidth = score === null ? "0%" : `${score}%`;

          return (
            <div className="col-12 col-md-6 col-lg-4" key={key}>
              <div className="card h-100 bg-dark text-light border border-secondary shadow-sm">
                <div className="card-body d-flex flex-column">
                  <div className="d-flex align-items-center justify-content-between mb-2">
                    <h5 className="mb-0" title={HELP[key]}>{LABELS[key]}</h5>
                    <div className="score-chip" aria-label={`${LABELS[key]} score`}>
                      {displayScore}
                    </div>
                  </div>
                  <div className="progress mb-3 bg-transparent" style={{ height: "10px" }}>
                    <div
                      className="progress-bar bg-success"
                      role="progressbar"
                      style={{ width: barWidth }}
                      aria-valuenow={score ?? 0}
                      aria-valuemin="0"
                      aria-valuemax="100"
                    />
                  </div>
                  <p className="text-secondary mb-0" style={{ minHeight: "72px" }}>
                    {normalized.feedback[key]}
                  </p>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="d-flex flex-wrap gap-2 mt-4">
        <button className="btn btn-primary" onClick={() => navigate("/select")}>
          Try Another Problem
        </button>
        {normalized.preview_url && (
          <a className="btn btn-outline-secondary" href={normalized.preview_url} target="_blank" rel="noreferrer">
            Rewatch
          </a>
        )}
      </div>
    </main>
  );
}




