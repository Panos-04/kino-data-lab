import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import TrendFramesPage from "./pages/TrendFramesPage";
import SingleNumberRelationsPage from "./pages/SingleNumberRelationsPage";
import GeneralRelationsPage from "./pages/GeneralRelationsPage";
import "./App.css";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/trends" replace />} />
        <Route path="/trends" element={<TrendFramesPage />} />
        <Route
          path="/single-number-relations"
          element={<SingleNumberRelationsPage />}
        />
        <Route path="/general-relations" element={<GeneralRelationsPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;