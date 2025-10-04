export default function Contact() {
  return (
    <section className="py-5 text-white">
      <div className="container" style={{ maxWidth: 720 }}>
        <h1 className="display-5 fw-semibold mb-3">Contact Us</h1>
        <p className="text-secondary mb-4">
          Reach out to collaborate, request features, or share feedback. We usually respond within one business day.
        </p>
        <div className="bg-dark border border-secondary rounded-3 p-4">
          <p className="mb-1"><strong>Email:</strong> <a className="link-light" href="mailto:team@leetcoach.ai">team@leetcoach.ai</a></p>
          <p className="mb-0"><strong>Twitter:</strong> <a className="link-light" href="https://twitter.com/LeetCoach" target="_blank" rel="noreferrer">@LeetCoach</a></p>
        </div>
      </div>
    </section>
  );
}
