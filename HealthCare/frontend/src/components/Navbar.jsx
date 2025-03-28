import React from "react";
import { Link } from "react-router-dom";
import "./NavbarComponent.css";

function NavbarComponent() {
  return (
    <nav className="custom-navbar">
      <div className="container">
        <Link to="/" className="brand">
          HealthCare
        </Link>
        <ul className="nav-links">
          <li className="nav-item">
            <Link to="/">Home</Link>
          </li>
          <li className="nav-item">
            <Link to="/about">About</Link>
          </li>
          <li className="nav-item">
            <Link to="/services">Services</Link>
          </li>

          <li className="nav-item">
            <Link to="/disease">Disease</Link>
          </li>
          <li className="nav-item">
            <Link to="/doctors">Doctors</Link>
          </li>
          <li className="nav-item dropdown">
            <span className="dropdown-title">Login ▼</span>
            <ul className="dropdown-menu">
              <li>
                <Link to="/user-login">User Login</Link>
              </li>
              <li>
                <Link to="/doctor-login">Doctor Login</Link>
              </li>
            </ul>
          </li>
        </ul>
      </div>
    </nav>
  );
}

export default NavbarComponent;