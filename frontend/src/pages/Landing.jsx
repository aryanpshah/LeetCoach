import { Link } from "react-router-dom";

export default function Landing() {
  return (
    <main className="landing d-flex flex-column justify-content-center align-items-center text-center">
      <div className="container">
        <div className="mx-auto" style={{ maxWidth: 780 }}>
          <h1 className="display-3 fw-bold hero-title mb-4">
            <span className="hero-chunk hero-chunk-orange">Leet</span>
            <span className="hero-chunk hero-chunk-blend">C</span>
            <span className="hero-chunk hero-chunk-green">oach</span>
          </h1>
          <p className="lead text-secondary mb-4">Rehearse smarter for technical interviews.</p>
          <div className="d-flex justify-content-center">
            <Link
              to="/select"
              className="btn btn-lg btn-accent"
              aria-label="Start now with problem selection"
              title="Start now with problem selection"
            >
              Start Now
            </Link>
          </div>
        </div>

        <section id="features" className="row g-4 mt-5">
          <div className="col-12 col-md-4">
            <div className="card h-100 bg-dark text-light shadow-sm border border-secondary">
              <div className="card-body">
                <span className="feature-icon mb-3" aria-hidden="true">VA</span>
                <h5 className="card-title">Video + Audio Insight</h5>
                <p className="card-text mb-0">Capture delivery cues with side-by-side webcam and voice analysis.</p>
              </div>
            </div>
          </div>
          <div className="col-12 col-md-4">
            <div className="card h-100 bg-dark text-light shadow-sm border border-secondary">
              <div className="card-body">
                <span className="feature-icon mb-3" aria-hidden="true">&lt;/&gt;</span>
                <h5 className="card-title">Code Check</h5>
                <p className="card-text mb-0">Upload code to auto-run tests and get efficiency hints.</p>
              </div>
            </div>
          </div>
          <div className="col-12 col-md-4">
            <div className="card h-100 bg-dark text-light shadow-sm border border-secondary">
              <div className="card-body">
                <span className="feature-icon mb-3" aria-hidden="true">5</span>
                <h5 className="card-title">5-Part Report</h5>
                <p className="card-text mb-0">Receive scores across accuracy, confidence, clarity, communication, professionalism.</p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}

