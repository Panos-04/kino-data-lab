import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import TrendFramesPage from "./pages/TrendFramesPage";
import SingleNumberRelationsPage from "./pages/SingleNumberRelationsPage";
import GeneralRelationsPage from "./pages/GeneralRelationsPage";
import ComboTestingPage from "./pages/ComboTestingPage";
import PatternTestingPage from "./pages/PatternTestingPage";
import ShapePatternTestingPage from "./pages/ShapePatternTestingPage";
import AppTopMenu from "./components/AppTopMenu";
import ShapeMovementsPage from "./pages/ShapeMovementsPage";
import AIResultsPage from "./pages/AIResultsPage";
import "./App.css";

function App() {
  return (
    <BrowserRouter>
    <AppTopMenu />
      <Routes>
        <Route path="/" element={<Navigate to="/trends" replace />} />
        <Route path="/trends" element={<TrendFramesPage />} />
        <Route
          path="/single-number-relations"
          element={<SingleNumberRelationsPage />}
        />
        <Route path="/general-relations" element={<GeneralRelationsPage />} />
        <Route path="/combo-testing" element={<ComboTestingPage />} />
        <Route path="/pattern-testing" element={<PatternTestingPage />} />
        <Route path="/shape-pattern-testing" element={<ShapePatternTestingPage />} />
        <Route path="/shape-movements" element={<ShapeMovementsPage />} />
        <Route path="/ai-results" element={<AIResultsPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;