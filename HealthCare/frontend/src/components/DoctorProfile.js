import React from "react";
import "./DoctorProfile.css";

const DoctorProfile = () => {
  return (
    <div className="profile-container">
      {/* Profile Section */}
      <div className="profile-header">
        <div className="profile-img">
          <i className="fas fa-user-md"></i>
        </div>
        <div className="profile-info">
          <h2>Dr. Sarah Thompson</h2>
          <p>Cardiologist | 12+ Years Experience</p>
        </div>
      </div>

      {/* Appointments */}
      <div className="stats-container">
        <div className="stat-box">
          <h3>20</h3>
          <p>Appointments</p>
        </div>
        <div className="stat-box">
          <h3>98%</h3>
          <p>Success Rate</p>
        </div>
        <div className="stat-box">
          <h3>4.9⭐</h3>
          <p>Rating</p>
        </div>
      </div>

      {/* Additional Information */}
      <div className="extra-info">
        <h3>Specialization:</h3>
        <p>Cardiology, Internal Medicine</p>

        <h3>Contact:</h3>
        <p>Email: dr.sarah@example.com</p>
        <p>Phone: +91 98765 43210</p>

        <h3>Availability:</h3>
        <p>Mon-Fri: 10 AM - 4 PM</p>
      </div>
    </div>
  );
};

export default DoctorProfile;
