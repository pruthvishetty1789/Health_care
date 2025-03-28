import React from "react";
import { Link } from "react-router-dom";
import "./DiseaseInfo.css";

const DiseaseInfo = () => {
  return (
    <div className="disease-container">
      {/* Heading */}
      <h1 className="title">Disease Prediction</h1>

      {/* Information card */}
      <div className="info-card">
        <p>Get accurate predictions for diseases like:</p>
        <ul>
          <li>Diabetes</li>
          <li>Lung Cancer</li>
          <li>Stroke</li>
          <li>Heart Disease</li>
        </ul>
      </div>

      {/* Predict Button with Link to Login */}
      <Link to="/login">
        <button className="predict-btn">Predict</button>
      </Link>
    </div>
  );
};

export default DiseaseInfo;
