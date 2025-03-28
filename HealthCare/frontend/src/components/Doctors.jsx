import React from "react";
import "./Doctors.css"; // Import CSS for styling

const doctors = [
  {
    name: "Dr. John Smith",
    specialization: "Cardiologist",
    phone: "+123 456 7890",
    email: "johnsmith@example.com",
    image:
      "https://purepng.com/public/uploads/large/purepng.com-doctorsdoctorsdoctors-and-nursesa-qualified-practitioner-of-medicine-aclinicianmedical-practitioner-1421526856614w2jt9.png",
  },
  {
    name: "Dr. Sarah Johnson",
    specialization: "Neurologist",
    phone: "+123 567 8901",
    email: "sarahjohnson@example.com",
    image: "https://www.humanitas.net/content/uploads/2017/10/doctors.jpg",
  },
  {
    name: "Dr. Michael Brown",
    specialization: "Endocrinologist",
    phone: "+123 678 9012",
    email: "michaelbrown@example.com",
    image:
      "https://purepng.com/public/uploads/large/purepng.com-doctordoctorsdoctors-and-nursesclinicianmedical-practitionernotepadfemale-1421526857248uragw.png",
  },
  {
    name: "Dr. Emily Davis",
    specialization: "Hepatologist",
    phone: "+123 789 0123",
    email: "emilydavis@example.com",
    image: "https://pngimg.com/uploads/doctor/doctor_PNG16028.png",
  },
  {
    name: "Dr. William Taylor",
    specialization: "Pulmonologist",
    phone: "+123 890 1234",
    email: "williamtaylor@example.com",
    image:
      "https://irp-cdn.multiscreensite.com/d8037e1a/dms3rep/multi/Functional+Medicine+Doctor+Chicago.jpeg",
  },
  {
    name: "Dr. Olivia Martinez",
    specialization: "Neurologist",
    phone: "+123 901 2345",
    email: "oliviamartinez@example.com",
    image:
      "https://img.freepik.com/premium-photo/medical-concept-indian-beautiful-female-doctor-white-coat-with-stethoscope-waist-up-medical-student-woman-hospital-worker-looking-camera-smiling-studio-blue-background_185696-621.jpg?w=1060",
  },
];

const Doctors = () => {
  return (
    <div className="doctors-container">
      <h1>Our Specialized Doctors</h1>
      <div className="doctors-grid">
        {doctors.map((doctor, index) => (
          <div className="doctor-card" key={index}>
            <img
              src={doctor.image}
              alt={doctor.name}
              className="doctor-image"
            />
            <div className="doctor-info">
              <h3>{doctor.name}</h3>
              <p>
                <strong>Specialization:</strong> {doctor.specialization}
              </p>
              <p>
                <strong>Phone:</strong> {doctor.phone}
              </p>
              <p>
                <strong>Email:</strong> {doctor.email}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Doctors;