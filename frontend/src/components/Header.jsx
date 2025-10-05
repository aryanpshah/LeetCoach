import { Link, NavLink } from "react-router-dom";
import logo from "../assets/leetcoach-logo.png";

export default function Header() {
  return (
    <nav className="navbar navbar-expand navbar-dark bg-dark border-bottom border-secondary sticky-top">
      <div className="container-fluid px-3">
        <div className="d-flex align-items-center justify-content-between w-100">
          <Link to="/" className="navbar-brand d-flex align-items-center me-4" aria-label="LeetCoach home">
            <img src={logo} alt="LeetCoach logo" className="header-logo" />
          </Link>
          <ul className="navbar-nav ms-auto flex-row align-items-center gap-3">
            <li className="nav-item">
              <NavLink to="/about" className="nav-link">
                About
              </NavLink>
            </li>
            <li className="nav-item">
              <NavLink to="/contact" className="nav-link">
                Contact
              </NavLink>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
}
