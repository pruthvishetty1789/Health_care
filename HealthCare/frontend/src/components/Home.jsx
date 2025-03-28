import React from "react";
import Carousel from "./Carousel";
import About from "./About"; // Import About component
import Services from "./Services";
import Doctors from "./Doctors";
import Mission from "./Mission";
import Footer from "./Footer";
const Home = () => {
  return (
    <div>
      <Carousel />
      <About /> {/* Display About page inside Home page */}
      <Services />
      <Mission />
      <Doctors />
      <Footer />
    </div>
  );
};

export default Home;