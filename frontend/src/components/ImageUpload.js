import React from 'react';

function ImageUpload({ label, onImageChange }) {
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    const reader = new FileReader();
    reader.onloadend = () => {
      onImageChange(reader.result); // Convert image to Base64 string
    };
    reader.readAsDataURL(file);
  };

  return (
    <div>
      <label>{label}</label>
      <input type="file" accept="image/*" onChange={handleFileChange} />
    </div>
  );
}

export default ImageUpload;
