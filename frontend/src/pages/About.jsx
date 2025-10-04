export default function About() {
  return (
    <section className="py-5 text-white">
      <div className="container" style={{ maxWidth: 760 }}>
        <h1 className="display-5 fw-semibold mb-4">About LeetCoach</h1>
        <p className="text-secondary lead">
          Preparing for coding interviews is often overwhelming. Candidates juggle hundreds of practice problems,
          endless video tutorials, and scattered notes, yet the real challenge is not just solving a problem. It is
          performing under pressure, explaining your reasoning out loud, and staying focused while the clock ticks.
          Traditional resources rarely capture that reality.
        </p>
        <p className="text-secondary">
          LeetCoach was built to close that gap. We wanted to recreate the actual interview environment: a problem
          statement that only reveals itself once you are ready, a timer that mirrors real-world constraints, and a
          camera to hold you accountable as you think through the solution. By simulating the pressure and structure of
          a real technical interview, LeetCoach helps you strengthen both problem-solving and communication skills.
        </p>
        <p className="text-secondary mb-0">
          Our platform is not just about drilling problems. It is about training the mindset required in front of an
          interviewer. Every session ends with a chance to reflect: upload your code, review your timing, and even
          revisit your recorded explanations. Over time, this process builds confidence, sharpens clarity, and reduces
          the anxiety that often overshadows knowledge. LeetCoach exists because practicing alone is not enough.
          Practicing in the right way is what sets candidates apart.
        </p>
      </div>
    </section>
  );
}
