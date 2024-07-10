import React, { useLayoutEffect, useState } from "react";
import rough from 'roughjs/bundled/rough.esm';
import getStroke from "perfect-freehand";
import axios from 'axios';

const generator = rough.generator();

const getSvgPathFromStroke = (stroke) => {
  if (!stroke.length) return "";

  const d = stroke.reduce(
    (acc, [x0, y0], i, arr) => {
      const [x1, y1] = arr[(i + 1) % arr.length];
      acc.push(x0, y0, (x0 + x1) / 2, (y0 + y1) / 2);
      return acc;
    },
    ["M", ...stroke[0], "Q"]
  );

  d.push("Z");
  return d.join(" ");
};

function createElement(points) {
  const stroke = getStroke(points, {
    size: 7,
    thinning: 1.0,
    smoothing: 1.0,
    streamline: 0.5,
  });
  const pathData = getSvgPathFromStroke(stroke);
  return {
    points, pathData
  };
}

const App = () => {
  const [elements, setElements] = useState([]);
  const [drawing, setDrawing] = useState(false);
  const [currentPoints, setCurrentPoints] = useState([]);
  const [prediction, setPrediction] = useState('');

  useLayoutEffect(() => {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const ctx2 = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx2.fillStyle = 'white';
    ctx2.fillRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = 'black';
    ctx.fillStyle = 'black';
    elements.forEach(({ pathData }) => {
      const path = new Path2D(pathData);
      ctx.stroke(path);
      ctx.fill(new Path2D(path));
    });
  }, [elements]);

  const handleMouseDown = (event) => {
    setDrawing(true);
    const { clientX, clientY } = event;
    setCurrentPoints([[clientX, clientY]]);
  };

  const handleMouseMove = (event) => {
    if (!drawing) return;
    const { clientX, clientY } = event;
    setCurrentPoints((prevPoints) => [...prevPoints, [clientX, clientY]]);
    const element = createElement([...currentPoints, [clientX, clientY]]);
    const elementCopy = [...elements];
    elementCopy[elements.length - 1] = element;
    setElements(elementCopy);
  };

  const handleMouseUp = () => {
    setDrawing(false);
    setElements((prevElements) => [
      ...prevElements,
      createElement(currentPoints)
    ]);
    setCurrentPoints([]);
  };

  const undo = () => {
    setElements((prevElements) => prevElements.slice(0, -1));
  };

  const exportImage = () => {
    const canvas = document.getElementById('canvas');
    const link = document.createElement('a');
    link.download = 'canvas.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  const Predict = async () => {
    const canvas = document.getElementById('canvas');
    canvas.toBlob(async (blob) => {
      if (blob) {
        const formData = new FormData();
        formData.append('file', blob, 'canvas.png');
        try {
          const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
          });
          setPrediction(response.data.prediction);
        } catch (error) {
          console.error('Error uploading the file', error);
        }
      }
    }, 'image/png');
  };
  return (
    <div style={{ backgroundColor: 'black', height: '100vh', display: 'flex', }}>
      <div style={{ position: 'relative' }}>
        <button onClick={undo} style={{ position: 'absolute', top: 10, left: 10 }}>
          Undo
        </button>
        <button onClick={Predict} style={{ position: 'absolute', top: 10, left: 70 }}>
          Predict
        </button>
        <button onClick={exportImage} style={{ position: 'absolute', top: 10, left: 140 }}>
          Export
        </button>
        <canvas
          id="canvas"
          width={700}
          height={600}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
        >
          Canvas
        </canvas>
        {prediction && <h2 style={{ position: 'absolute', top: 10, left: 200 }}>Prediction: {prediction}</h2>}
      </div>
    </div>
  );
};

export default App;
