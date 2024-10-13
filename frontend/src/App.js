import React, { useState } from 'react';
import PredictionForm from './components/PredictionForm';
import PredictionResult from './components/PredictionResult';

function App() {
  const [result, setResult] = useState(null);

  return (
    <div>
      <h1>Face Analysis Model</h1>
      <PredictionForm onResult={setResult} />
      <PredictionResult result={result} />
    </div>
  );
}

export default App;
