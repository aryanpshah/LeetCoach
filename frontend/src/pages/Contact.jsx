export default function Contact() {
  return (
    <section className="py-5 text-white">
      <div className="container" style={{ maxWidth: 720 }}>
        <h1 className="display-5 fw-semibold mb-3">Contact Us</h1>
        <p className="text-secondary mb-4">
          Reach out when you want to collaborate, request a feature, or share feedback from your mock sessions.
          We usually reply within one business day.
        </p>
        <div className="bg-dark border border-secondary rounded-3 p-4 d-flex flex-column gap-3">
          <div>
            <div className="text-uppercase text-secondary fw-semibold small">Email</div>
            <a className="link-light h5 mb-0" href="mailto:team@leetcoach.ai">team@leetcoach.ai</a>
            <p className="text-secondary small mb-0">Send session recordings, product ideas, or questions anytime.</p>
          </div>
          <div>
            <div className="text-uppercase text-secondary fw-semibold small">Community</div>
            <p className="text-secondary mb-0">
              Join the LeetCoach Discord (invite coming soon) to swap strategies and get early feature updates.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
