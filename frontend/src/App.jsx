import { BrowserRouter, Routes, Route } from "react-router-dom";
import Header from "./components/Header";
import Footer from "./components/Footer";
import Landing from "./pages/Landing";
import About from "./pages/About";
import Contact from "./pages/Contact";
import ProblemSelect from "./pages/ProblemSelect";
import Test from "./pages/Test";
import Report from "./pages/Report";

export default function App() {
  return (
    <BrowserRouter>
      <div className="d-flex flex-column min-vh-100 bg-app">
        <Header />
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/about" element={<About />} />
          <Route path="/contact" element={<Contact />} />
          <Route path="/select" element={<ProblemSelect />} />
          <Route path="/test/:id" element={<Test />} />
          <Route path="/report/:sessionId" element={<Report />} />
        </Routes>
        <Footer />
      </div>
    </BrowserRouter>
  );
}