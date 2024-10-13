import React from 'react';

function PredictionResult({ result }) {
  if (!result) return null;

  return (
    <div>
      <h2>Results:</h2>
      <p>Distance (Anchor-Positive): {result.rec_output[0]}</p>
      <p>Distance (Anchor-Negative): {result.rec_output[1]}</p>
      <p>Gender Prediction: {result.gender_output}</p>
      <p>Age Prediction: {result.age_output}</p>
    </div>
  );
}

export default PredictionResult;
