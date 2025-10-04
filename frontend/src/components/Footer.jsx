export default function Footer() {
  return (
    <footer className="py-4 bg-dark text-secondary border-top border-secondary">
      <div className="container small text-center">
        <span>&copy; {new Date().getFullYear()} LeetCoach</span>
      </div>
    </footer>
  );
}
