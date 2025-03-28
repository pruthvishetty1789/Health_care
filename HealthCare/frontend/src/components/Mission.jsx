import React from "react";
import "./Mission.css";

const Mission = () => {
  return (
    <div className="mission-container">
      <section className="intro">
        <h1>
          🚀{" "}
          <span className="highlight">
            Our Mission: AI-Powered Early Disease Detection
          </span>
        </h1>
        <p>
          We are committed to transforming healthcare by leveraging
          <strong> cutting-edge AI</strong> for early disease detection,
          empowering medical professionals with accurate, data-driven insights.
        </p>
      </section>

      <section className="approach">
        <h2>🌟 Our Approach</h2>
        <div className="approach-cards">
          <div className="card">
            <i className="fas fa-heartbeat"></i>
            <h3>Early Detection of Critical Diseases</h3>
            <ul>
              <li>
                <strong>Cancer:</strong> Early-stage identification for timely
                treatment.
              </li>
              <li>
                <strong>Neurological Disorders:</strong> Detecting Alzheimer’s,
                Parkinson’s, and more.
              </li>
              <li>
                <strong>Cardiovascular Conditions:</strong> Early identification
                for preventive care.
              </li>
            </ul>
          </div>

          <div className="card">
            <i className="fas fa-chart-line"></i>
            <h3>Data-Driven Insights</h3>
            <p>
              Combining <strong>AI</strong> with big data analytics to provide
              healthcare professionals with accurate, real-time insights,
              enabling informed decision-making.
            </p>
          </div>

          <div className="card">
            <i className="fas fa-users"></i>
            <h3>Our Commitment</h3>
            <p>
              We strive to enhance diagnostic accuracy, reduce false positives,
              and make early disease detection accessible and affordable
              worldwide.
            </p>
          </div>
        </div>
      </section>

      <footer className="footer">
        <p>🔥 Shaping the future of healthcare with AI-powered innovation 💙</p>
      </footer>
    </div>
  );
};

export default Mission;