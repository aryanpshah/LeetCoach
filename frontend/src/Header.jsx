import { Link, NavLink } from "react-router-dom";
import logo from "../assets/leetcoach-logo.png";

export default function Header() {
  return (
    <nav className="navbar navbar-expand navbar-dark bg-dark border-bottom border-secondary sticky-top px-3">
      <div className="container">
        <Link to="/" className="navbar-brand d-flex align-items-center" aria-label="LeetCoach home">
          <img src={logo} alt="LeetCoach" className="logo-mark" />
        </Link>
        <div className="ms-auto">
          <ul className="navbar-nav">
            <li className="nav-item"><NavLink to="/about" className="nav-link">About</NavLink></li>
            <li className="nav-item"><NavLink to="/contact" className="nav-link">Contact</NavLink></li>
          </ul>
        </div>
      </div>
    </nav>
  );
}
