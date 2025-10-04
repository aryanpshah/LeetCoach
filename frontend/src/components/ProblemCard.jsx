import { useNavigate } from "react-router-dom";
import { useMediaPermissions } from "../hooks/useMediaPermissions";

const difficultyBadge = {
  Easy: "success",
  Medium: "warning",
  Hard: "danger"
};

function ClockIcon() {
  return (
    <svg
      aria-hidden="true"
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="text-secondary"
    >
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.5" />
      <path d="M12 7.5V12l3 1.75" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function TargetIcon() {
  return (
    <svg
      aria-hidden="true"
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="text-secondary"
    >
      <circle cx="12" cy="12" r="7.5" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="12" cy="12" r="0.75" fill="currentColor" />
    </svg>
  );
}

export default function ProblemCard({ id, title, difficulty, tags, skills = [], desc, timeLimit }) {
  const badgeColor = difficultyBadge[difficulty] ?? "secondary";
  const focusArea = tags[0];
  const visibleSkills = skills.length ? skills : tags;
  const navigate = useNavigate();
  const { requestPermissions, status, error } = useMediaPermissions();

  const handleSelect = async (event) => {
    event.preventDefault();
    const granted = await requestPermissions();
    if (granted) {
      navigate(`/test/${id}`, { state: { problemId: id } });
    }
  };

  const isRequesting = status === "pending";

  return (
    <div className="col">
      <article className="card h-100 bg-dark text-light border border-secondary shadow-sm problem-card">
        <div className="card-body">
          <div className="problem-card__top">
            <div className="d-flex justify-content-between align-items-start">
              <h5 className="card-title mb-0">{title}</h5>
              <span className={`badge bg-${badgeColor}`}>{difficulty}</span>
            </div>
            <p className="small text-secondary mb-0">{tags.join(" / ")}</p>
          </div>

          <div className="problem-card__skills" aria-label="Suggested skills">
            {visibleSkills.map((skill) => (
              <span key={skill} className="problem-card__skill-chip">{skill}</span>
            ))}
          </div>

          <div className="problem-card__meta mt-auto">
            <div className="problem-card__meta-item">
              <ClockIcon />
              <div>
                <span className="meta-label">Time</span>
                <span className="text-white">{timeLimit}</span>
              </div>
            </div>
            <div className="problem-card__meta-item text-md-end">
              <TargetIcon />
              <div>
                <span className="meta-label">Focus</span>
                <span className="text-white">{focusArea}</span>
              </div>
            </div>
          </div>
        </div>

        <div className="problem-card__overlay">
          <div>
            <h6 className="text-white-50 text-uppercase small mb-2">Problem Brief</h6>
            <p className="small text-secondary mb-0">{desc}</p>
          </div>
          <div className="d-flex flex-column gap-3">
            <div className="d-flex justify-content-between align-items-center text-secondary small">
              <span className="text-white fw-semibold">Time Limit</span>
              <span>{timeLimit}</span>
            </div>
            {status === "denied" && (
              <p className="problem-card__permission-error" role="alert">{error}</p>
            )}
            <a
              href={`/test/${id}`}
              className={`btn btn-accent w-100${isRequesting ? " disabled" : ""}`}
              aria-disabled={isRequesting}
              tabIndex={isRequesting ? -1 : 0}
              aria-label={`Select problem ${title}`}
              onClick={handleSelect}
            >
              {isRequesting ? "Requesting access..." : "Select Problem"}
            </a>
          </div>
        </div>
      </article>
    </div>
  );
}