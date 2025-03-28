import React from "react";
import "./About.css"; // Import CSS for styling

const About = () => {
  return (
    <div className="about-container">
      <h1>About Early Health Disease Prediction</h1>
      <img
        src="https://wallpapercave.com/wp/wp2968524.jpg"
        alt="Healthcare"
        className="about-image"
      />
      <p>
        Early health disease prediction uses advanced data analysis and AI
        models to identify potential health risks before symptoms appear. By
        leveraging patient data, medical history, and lifestyle factors,
        predictive analytics help doctors provide early interventions, improving
        overall healthcare outcomes.
      </p>
      <p>
        Our platform provides accurate disease predictions using cutting-edge
        technology, ensuring proactive healthcare solutions.
      </p>
    </div>
  );
};

export default About;