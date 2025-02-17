import React from 'react';
import './App.css';

const ImageUpload = ({ onImageUpload }) => {
  const handleImageChange = (event) => {
    const file = event.target.files[0];
    onImageUpload(file);
  };

  return (
    <div>
      <input
        type="file"
        accept="image/*"
        id="file-input"
        className="custom-file-input"
        onChange={handleImageChange}
      />
      <label htmlFor="file-input" className="custom-file-label">
        Choose File
      </label>
    </div>
  );
};

export default ImageUpload;
