const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusOverlay = document.getElementById('status-overlay');
const modelStatusSpan = document.getElementById('model-status');
const lastMatchSpan = document.getElementById('last-match');
const captureButton = document.getElementById('capture'); // New reference

let model;
const knownEmbeddings = [];
const labels = [];
// Threshold set strictly to 2.0 based on previous troubleshooting
const THRESHOLD = 0.8; 

// ---------------------------------------------------------------------
// 1. LANDMARK NORMALIZATION FUNCTION (Remaing the same)
// ---------------------------------------------------------------------
const normalizeLandmarks = (landmarks) => {
    // Landmarks are: [0: R-eye, 1: L-eye, 2: Nose, 3: R-ear, 4: L-ear, 5: Mouth]
    
    // 1. Calculate the distance between the eyes (Landmark 0 and 1) for scaling
    const eyeDist = Math.sqrt(
        Math.pow(landmarks[0][0] - landmarks[1][0], 2) + 
        Math.pow(landmarks[0][1] - landmarks[1][1], 2)
    );
    
    if (eyeDist < 10) return landmarks.flat(); 

    // 2. Define the center point (Nose, Landmark 2) for translation
    const noseX = landmarks[2][0];
    const noseY = landmarks[2][1];
    
    // 3. Normalize (center and scale)
    return landmarks.map(([x, y]) => [
        (x - noseX) / eyeDist, // Normalized X
        (y - noseY) / eyeDist  // Normalized Y
    ]).flat();
};
// ---------------------------------------------------------------------

// Load the BlazeFace model (Remaing the same)
async function loadModel() {
    try {
        modelStatusSpan.textContent = 'Loading BlazeFace model...';
        model = await blazeface.load();
        modelStatusSpan.textContent = 'BlazeFace Ready';
    } catch (error) {
        modelStatusSpan.textContent = 'Error loading model!';
        console.error('Error loading BlazeFace model:', error);
        throw new Error('Failed to load BlazeFace model.');
    }
}

// Set up the webcam (Remaing the same)
async function setupWebcam() {
    return new Promise((resolve, reject) => {
        modelStatusSpan.textContent = 'Awaiting camera permission...';
        const constraints = { video: true };
        
        navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
            video.srcObject = stream;
            video.onloadedmetadata = () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                
                statusOverlay.classList.add('status-hidden'); 
                modelStatusSpan.textContent = 'Webcam Active';
                resolve(video);
            };
        }).catch((err) => {
            modelStatusSpan.textContent = 'Camera Denied or Not Found!';
            console.error('Webcam setup error:', err);
            reject(err);
        });
    });
}

// Add a known face (Remaing the same)
async function addKnownFace(imagePath, label) {
    return new Promise((resolve) => {
        const img = new Image();
        img.crossOrigin = 'anonymous'; 
        img.src = imagePath;

        img.onload = async () => {
            try {
                const tensor = tf.browser.fromPixels(img);
                const predictions = await model.estimateFaces(tensor, false);
                
                if (predictions.length > 0) {
                    const normalizedEmbedding = normalizeLandmarks(predictions[0].landmarks);
                    knownEmbeddings.push(normalizedEmbedding);
                    labels.push(label);
                    console.log(`Face registered for: ${label}`);
                } else {
                    console.warn(`No face detected in image: ${imagePath}`);
                }
                tensor.dispose();
            } catch (e) {
                console.error(`Error processing face for ${label}:`, e);
            }
            resolve();
        };

        img.onerror = () => {
            console.error(`Failed to load image at path: ${imagePath}. Check filename and path.`);
            resolve();
        };
    });
}

// Compare face embeddings to find the best match (Remaing the same)
function findBestMatch(faceEmbedding) {
    let minDistance = Infinity;
    let bestMatch = 'Unknown';
    
    const normalizedFaceData = normalizeLandmarks(faceEmbedding);
    const faceTensor = tf.tensor(normalizedFaceData);

    knownEmbeddings.forEach((embedding, index) => {
        const knownTensor = tf.tensor(embedding);
        const distance = tf.norm(faceTensor.sub(knownTensor)).dataSync()[0];
        
        if (distance < minDistance) {
            minDistance = distance;
            bestMatch = labels[index];
        }
        knownTensor.dispose(); 
    });

    faceTensor.dispose(); 

    if (minDistance > THRESHOLD) {
        bestMatch = 'Unknown';
    }

    return `${bestMatch}:${minDistance}`;
}

// Detect faces and recognize them in real-time (Remaing the same)
async function detectFaces() {
    if (video.readyState < 2 || !model) { 
        requestAnimationFrame(detectFaces);
        return;
    }

    const predictions = await model.estimateFaces(video, false);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (predictions.length > 0) {
        predictions.forEach((prediction) => {
            const [startX, startY] = prediction.topLeft;
            const [endX, endY] = prediction.bottomRight;
            const faceEmbedding = prediction.landmarks;

            const matchWithDistance = findBestMatch(faceEmbedding);
            const [match, distance] = matchWithDistance.split(':');
            
            lastMatchSpan.textContent = `${match} (Dist: ${parseFloat(distance).toFixed(3)})`; 
            
            const boxColor = match === 'Unknown' ? 'red' : 'lime';

            ctx.beginPath();
            ctx.rect(startX, startY, endX - startX, endY - startY); 
            ctx.lineWidth = 3;
            ctx.strokeStyle = boxColor;
            ctx.stroke();

            ctx.font = '20px Arial';
            const text = match;
            const textMetrics = ctx.measureText(text);
            
            ctx.fillStyle = boxColor + 'aa'; 
            ctx.fillRect(startX, startY - 25, textMetrics.width + 10, 25);
            
            ctx.fillStyle = 'white';
            ctx.fillText(text, startX + 5, startY - 5);
        });
    } else {
        lastMatchSpan.textContent = 'No Face Detected';
    }

    requestAnimationFrame(detectFaces);
}

// ---------------------------------------------------------------------
// 2. NEW ENROLLMENT LOGIC IMPLEMENTATION
// ---------------------------------------------------------------------
captureButton.addEventListener('click', async () => {
    if (!model || video.readyState < 2) {
        alert('Model not loaded or video not ready. Please wait.');
        return;
    }
    
    // Temporarily draw video frame to an off-screen canvas if necessary, 
    // but for simplicity, we'll use tf.browser.fromPixels(video) directly.
    const videoTensor = tf.browser.fromPixels(video);
    
    // 1. Detect faces in the current video frame
    const predictions = await model.estimateFaces(videoTensor, false);
    videoTensor.dispose(); // Clean up tensor immediately

    if (predictions.length !== 1) {
        alert('Please ensure only ONE face is clearly visible to enroll.');
        return;
    }

    const newLabel = prompt("Enter the name for the face you are enrolling:");
    
    if (newLabel && newLabel.trim() !== '') {
        const faceEmbedding = predictions[0].landmarks;
        
        // 2. Normalize and store the embedding
        const normalizedEmbedding = normalizeLandmarks(faceEmbedding);
        knownEmbeddings.push(normalizedEmbedding);
        labels.push(newLabel.trim());
        
        // 3. Provide feedback
        alert(`Successfully enrolled face for: ${newLabel.trim()}. Total known faces: ${labels.length}`);
        console.log(`New face enrolled: ${newLabel.trim()}`);
    } else {
        alert('Enrollment cancelled or invalid name provided.');
    }
});
// ---------------------------------------------------------------------

// Initialize the webcam and model (Remaing the same)
(async function () {
    statusOverlay.classList.remove('status-hidden'); 
    
    try {
        await loadModel();
        
        // --- Add known faces (Initial load) ---
        await addKnownFace('Gaurav.jpg', 'Gaurav');
        await addKnownFace('Junaid12.jpg', 'Junaid');
        await addKnownFace('4.jpg', 'Junaid');
        await addKnownFace('5.jpg', 'Junaid');
        await addKnownFace('Junaid2.jpg', 'Junaid');
        await addKnownFace('Junaid45.jpg', 'Junaid');
        await addKnownFace('Junaid67.jpg', 'Junaid');
        // -------------------------------------------------------------------

        await setupWebcam();
        video.play();
        
        detectFaces();
    } catch (error) {
        statusOverlay.textContent = 'Initialization Failed. Check console (F12) for details.';
        console.error('Application startup failed:', error);
    }
})();