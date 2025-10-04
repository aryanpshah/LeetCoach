import { BrowserRouter, Routes, Route } from "react-router-dom";
import Header from "./components/Header";
import Footer from "./components/Footer";
import Landing from "./pages/Landing";
import About from "./pages/About";
import Contact from "./pages/Contact";

function Placeholder({ title, description }) {
  return (
    <div className="container py-5 text-white">
      <h1 className="display-5 fw-semibold mb-3">{title}</h1>
      {description && <p className="text-secondary mb-0">{description}</p>}
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="d-flex flex-column min-vh-100 bg-app">
        <Header />
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/about" element={<About />} />
          <Route path="/contact" element={<Contact />} />
          <Route
            path="/select"
            element={<Placeholder title="Problem Selection" description="Pick a problem to kick off your mock interview session." />}
          />
          <Route
            path="/test/:id"
            element={<Placeholder title="Test Session" description="Interactive coding session experience coming soon." />}
          />
          <Route
            path="/report/:sessionId"
            element={<Placeholder title="Report" description="Personalized coaching feedback will display here." />}
          />
        </Routes>
        <Footer />
      </div>
    </BrowserRouter>
  );
}
