import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import "./Auth.css";

const DoctorRegister = () => {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate(); // Initialize navigation hook

  const handleRegister = (e) => {
    e.preventDefault();

    // Retrieve existing users
    const users = JSON.parse(localStorage.getItem("users")) || [];

    // Normalize email for case-insensitive check
    const normalizedEmail = email.toLowerCase();
    const userExists = users.some((user) => user.email === normalizedEmail);

    if (userExists) {
      alert("User already exists! Redirecting to login.");
      navigate("/doctor-login"); // Ensure route matches App.js
      return;
    }

    // Save new user
    const newUser = { name, email: normalizedEmail, password };
    localStorage.setItem("users", JSON.stringify([...users, newUser]));

    // Ensure navigation works
    alert("Registration successful! Redirecting to login...");
    setTimeout(() => navigate("/doctor-login"), 1000); // Delayed navigation to prevent state issues
  };

  return (
    <div className="auth-container">
      <h2>Doctor Registration</h2>
      <form onSubmit={handleRegister}>
        <input
          type="text"
          placeholder="Full Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
        />
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        <button type="submit">Register</button>
      </form>
      <p>
        Already have an account? <Link to="/doctor-login">Login here</Link>
      </p>
    </div>
  );
};

export default DoctorRegister;
