import React, { useState } from 'react';
import axios from 'axios';
import ImageUpload from './ImageUpload';

function PredictionForm({ onResult }) {
  const [anchorImage, setAnchorImage] = useState(null);
  const [positiveImage, setPositiveImage] = useState(null);
  const [negativeImage, setNegativeImage] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();

    const payload = {
      anchor: anchorImage,
      positive: positiveImage,
      negative: negativeImage,
    };

    try {
      const response = await axios.post('http://localhost:5000/predict', payload);
      onResult(response.data);
    } catch (error) {
      console.error('Error submitting prediction request:', error);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <ImageUpload label="Anchor Image:" onImageChange={setAnchorImage} />
      <ImageUpload label="Positive Image:" onImageChange={setPositiveImage} />
      <ImageUpload label="Negative Image:" onImageChange={setNegativeImage} />
      <button type="submit">Submit</button>
    </form>
  );
}

export default PredictionForm;
