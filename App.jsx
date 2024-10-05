import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Canvas } from '@react-three/fiber';

function App() {
  const [image, setImage] = useState(null);
  const [segmentedImage, setSegmentedImage] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    setImage(e.target.files[0]);
    setSegmentedImage(null);
  };

  // const handleSubmit = async (e) => {
  //   e.preventDefault();
  //   if (!image) {
  //     alert("Please upload an image first.");
  //     return;
  //   }

  //   const formData = new FormData();
  //   formData.append('file', image);
  //   setLoading(true);

  //   try {
  //     const response = await axios.post('http://localhost:8000/segment', formData, {
  //       headers: { 'Content-Type': 'multipart/form-data' },
  //       responseType: 'blob',
  //     });

  //     const imageUrl = URL.createObjectURL(new Blob([response.data]));
  //     setSegmentedImage(imageUrl);
  //   } catch (error) {
  //     console.error('Error during segmentation:', error);
  //     alert('Failed to get segmented image. Please try again.');
  //   } finally {
  //     setLoading(false);
  //   }
  // };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) {
      alert("Please upload an image first.");
      return;
    }
  
    const formData = new FormData();
    formData.append('file', image);
    setLoading(true);
  
    try {
      const response = await axios.post('http://smooth-att-viz-env-2.eba-nryfavjt.us-west-2.elasticbeanstalk.com/segment', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        responseType: 'blob',
      });
  
      const imageUrl = URL.createObjectURL(new Blob([response.data]));
      setSegmentedImage(imageUrl);
    } catch (error) {
      console.error('Error during segmentation:', error);
      alert('Failed to get segmented image. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    return () => {
      if (segmentedImage) URL.revokeObjectURL(segmentedImage);
      if (image) URL.revokeObjectURL(URL.createObjectURL(image));
    };
  }, [segmentedImage, image]);

  return (
    <div style={styles.container}>
      {/* Main content section */}
      <div style={styles.mainSection}>
        <div style={styles.card}>
          <h1 style={styles.title}>Image Segmentation</h1>
          <form onSubmit={handleSubmit} style={styles.form}>
            <input
              type="file"
              accept="image/*"
              onChange={handleImageChange}
              style={styles.fileInput}
            />
            <button type="submit" style={styles.button}>
              {loading ? 'Processing...' : 'Segment Image'}
            </button>
          </form>

          {/* Display the uploaded image */}
          {image && (
            <div style={styles.imagePreview}>
              <h2 style={styles.imageText}>Uploaded Image:</h2>
              <img
                src={URL.createObjectURL(image)}
                alt="Uploaded"
                style={styles.image}
              />
            </div>
          )}

          {/* Display the segmented image returned from the backend */}
          {segmentedImage && (
            <div style={styles.imagePreview}>
              <h2 style={styles.imageText}>Segmented Image:</h2>
              <img
                src={segmentedImage}
                alt="Segmented Result"
                style={styles.image}
              />
            </div>
          )}
        </div>
      </div>

      {/* Right section for the welcome message and 3D object */}
      <div style={styles.rightSection}>
        <h1 style={styles.welcomeText}>
          Welcome to the Smooth Attention Segmentation Visualization Tool for Flood Segmentation
        </h1>

        {/* 3D Object */}
        <div style={styles.threeDContainer}>
          <Canvas>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} />
            <RotatingCube />
          </Canvas>
        </div>
      </div>
    </div>
  );
}

// 3D Cube Component with dark blue color
function RotatingCube() {
  const [rotation, setRotation] = useState([0, 0, 0]);

  useEffect(() => {
    const handle = setInterval(() => {
      setRotation((prev) => [
        prev[0] + 0.01,
        prev[1] + 0.01,
        prev[2] + 0.01,
      ]);
    }, 16); // Rotate at 60 fps

    return () => clearInterval(handle);
  }, []);

  return (
    <mesh rotation={rotation}>
      <boxGeometry args={[2, 2, 2]} />
      <meshStandardMaterial color="#00008b" /> {/* Dark blue color */}
    </mesh>
  );
}

// Styling
const styles = {
  container: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    minHeight: '100vh', // Ensure it covers the whole window height
    width: '100vw', // Ensure full width
    backgroundColor: '#f0f4f8',
    padding: '0 20px',
    margin: 0,
    boxSizing: 'border-box',
  },
  mainSection: {
    flex: 1,
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
  },
  card: {
    backgroundColor: '#fff',
    padding: '40px',
    borderRadius: '15px',
    boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
    textAlign: 'center',
    width: '100%',
    maxWidth: '500px',
  },
  title: {
    marginBottom: '20px',
    fontSize: '24px',
    color: '#333',
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: '15px',
  },
  fileInput: {
    padding: '10px',
    borderRadius: '5px',
    border: '1px solid #ddd',
    fontSize: '16px',
  },
  button: {
    padding: '10px',
    borderRadius: '5px',
    border: 'none',
    backgroundColor: '#007bff',
    color: '#fff',
    fontSize: '16px',
    cursor: 'pointer',
    transition: 'background-color 0.3s ease',
  },
  imagePreview: {
    marginTop: '20px',
    textAlign: 'left', // Align text to the left
  },
  imageText: {
    marginBottom: '10px', // Add some space below the heading
  },
  image: {
    width: '100%',
    height: 'auto',
    borderRadius: '10px',
    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
  },
  rightSection: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    textAlign: 'center',
    padding: '20px',
  },
  welcomeText: {
    fontSize: '32px',
    color: '#333',
    fontFamily: "'Dancing Script', cursive",
    fontWeight: 700,
    lineHeight: '1.5',
    marginBottom: '20px',
  },
  threeDContainer: {
    width: '300px',
    height: '300px',
  },
};

export default App;