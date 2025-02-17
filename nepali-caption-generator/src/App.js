
import React, { useState } from 'react';
import ImageUpload from './ImageUpload';
import './App.css';

const App = () => {
  const [image, setImage] = useState(null);
  const [caption, setCaption] = useState('');
  const [loading, setLoading] = useState(false);

  const handleImageUpload = async (file) => {
    setImage(URL.createObjectURL(file));
    setLoading(true);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict/', {
        method: 'POST',
        body: formData,
      });
      const result = await response.json();
      setCaption(result.caption);
    } catch (error) {
      console.error('Error:', error);
      setCaption('Failed to generate caption.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Image Caption Generator</h1>
      <ImageUpload onImageUpload={handleImageUpload} />
      {image && <div className="img-preview"><img src={image} alt="Uploaded" /></div>}
      {loading ? (
        <div className="loading-spinner"></div>
      ) : (
        caption && <p><strong>Caption: </strong>{caption}</p>
      )}
    </div>
  );
};

export default App;
