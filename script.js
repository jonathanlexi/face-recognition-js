// HTML elements
const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('canvas');
const startButton = document.getElementById('startButton');
const nameElement = document.getElementById('name');
const statusElement = document.getElementById('status');
// Load models
Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
  faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
  faceapi.nets.faceExpressionNet.loadFromUri('/models')
]).then(startVideo);

// Start video stream
async function startVideo() {
  await faceapi.nets.ssdMobilenetv1.loadFromUri('/models'); // Load the face detection model

  navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
      videoElement.srcObject = stream;
    })
    .catch(err => console.error(err));
}


// Recognize faces
videoElement.addEventListener('play', async () => {
  const labeledFaceDescriptors = await loadLabeledImages();
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);
  const labels = ['Jonathan', 'Oliver', 'Data'];
  const displaySize = { width: videoElement.width, height: videoElement.height };
  faceapi.matchDimensions(canvasElement, displaySize);

  setInterval(async () => {
    const detections = await faceapi.detectAllFaces(videoElement, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptors();

    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    canvasElement.getContext('2d').clearRect(0, 0, canvasElement.width, canvasElement.height);
    faceapi.draw.drawDetections(canvasElement, resizedDetections);
    faceapi.draw.drawFaceLandmarks(canvasElement, resizedDetections);

    const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor));
    results.forEach((result, i) => {
      const box = resizedDetections[i].detection.box;
      const { label, distance } = result;
    
      if (labels.includes(label) && distance < 0.5) {
        nameElement.innerText = label;
        statusElement.innerText = 'Diterima';
        statusElement.classList.add('status-diterima');
        statusElement.classList.remove('status-ditolak');
      } else {
        nameElement.innerText = 'Unknown';
        statusElement.innerText = 'Tidak Diterima';
        statusElement.classList.add('status-ditolak');
        statusElement.classList.remove('status-diterima');
      }
    
      const text = `${nameElement.innerText} (${Math.round(distance * 100) / 100})`;
      new faceapi.draw.DrawTextField([text], box.bottomRight).draw(canvasElement);
    });
  }, 100);
});


// Load labeled images with names
async function loadLabeledImages() {
  const labels = ['Jonathan','Oliver','Data']; // Add your name here
  return Promise.all(
    labels.map(async label => {
      const descriptions = [];
      for (let i = 1; i <= 2; i++) { // Assuming you have 3 images per person
        const img = await faceapi.fetchImage(`/labeled_images/${label}/${i}.png`);
        const detections = await faceapi.detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();
        descriptions.push(detections.descriptor);
      }
      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}

