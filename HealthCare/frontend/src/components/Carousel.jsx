import React from "react";
import Carousel from "react-bootstrap/Carousel";
import { useNavigate } from "react-router-dom";
import "bootstrap/dist/css/bootstrap.min.css";
import "./Carousel.css"; // Import CSS for styling

const slides = [
  {
    image: "https://wallpaperaccess.com/full/136949.jpg",
    title: "Find Top Doctors",
    description: "Discover the best healthcare professionals near you.",
    page: "/doctors",
  },
  {
    image:
      "https://www.intertek.com/siteassets/blogs/2024-04-16-blog-widebanner.jpg",
    title: "Book an Appointment",
    description: "Schedule a consultation with expert doctors.",
    page: "/appointment",
  },
  {
    image:
      "https://assets-global.website-files.com/5f1eeef92f5cd255b57d5931/5fbec2aa6e5748b3bd00cf9d_hc-og.jpg",

    description: "Get detailed information on health conditions.",
    page: "/disease",
  },
];

const CustomCarousel = () => {
  const navigate = useNavigate();

  return (
    <Carousel className="full-width-carousel">
      {slides.map((slide, index) => (
        <Carousel.Item key={index}>
          <img
            className="d-block w-100 carousel-image"
            src={slide.image}
            alt={`Slide ${index}`}
          />
          <Carousel.Caption className="carousel-caption">
            <h3>{slide.title}</h3>
            <p>{slide.description}</p>
            <button
              className="carousel-btn"
              onClick={() => navigate(slide.page)}
            >
              Learn More
            </button>
          </Carousel.Caption>
        </Carousel.Item>
      ))}
    </Carousel>
  );
};

export default CustomCarousel;