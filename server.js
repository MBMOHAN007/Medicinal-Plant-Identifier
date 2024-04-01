const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const { createCanvas, loadImage } = require('canvas');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;

// Set up multer for handling file uploads
const upload = multer({ dest: 'uploads/' });

// Load the pre-trained model
const MODEL_PATH = 'path_to_your_model/model.json';
let model;

async function loadModel() {
    model = await tf.loadLayersModel(`file://${MODEL_PATH}`);
    console.log('Model loaded successfully');
}
loadModel();

// Function to preprocess the uploaded image
async function preprocessImage(imagePath) {
    const image = await loadImage(imagePath);
    const canvas = createCanvas(224, 224);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, 224, 224);

    // Convert the image data to a tensor
    const input = tf.browser.fromPixels(canvas).toFloat().div(tf.scalar(255)).expandDims();

    return input;
}

// Route for image upload and plant identification
// Route for image upload and plant identification
app.post('/identify_plant', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No file uploaded' });
        }

        const imagePath = req.file.path;

        // Log the received image path
        console.log('Received image:', imagePath);

        // Preprocess the uploaded image
        const input = await preprocessImage(imagePath);

        // Perform inference using the loaded model
        const prediction = model.predict(input);

        // Decode the prediction and return the result
        // Example: const plantName = decodePrediction(prediction);

        res.json({ result: 'predicted_plant_name' });
    } catch (error) {
        console.error('Error:', error);
        res.status(500).json({ error: 'Internal server error' });
    } finally {
        // Delete the uploaded file after processing
        fs.unlink(imagePath, err => {
            if (err) {
                console.error('Error deleting file:', err);
            }
        });
    }
});


// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${5500}`);
});
