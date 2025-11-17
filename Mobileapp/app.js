// app.js
/* Rock, Paper, Scissors — TF.js webcam classifier
   - Loads a pre-trained Layers model from ./rps_web_model/model.json
   - Grabs frames from the camera, preprocesses to 224x224 float32 / 255
   - Predicts continuously using requestAnimationFrame
   - Disposes tensors to avoid memory leaks
*/

const VIDEO    = document.getElementById('video');
const EMOJI    = document.getElementById('emoji');
const LABEL    = document.getElementById('label');
const PROB     = document.getElementById('prob');
const STATUS   = document.getElementById('status');
const BTN_SW   = document.getElementById('btnSwitch');
const BTN_PA   = document.getElementById('btnPause');
const BTN_RE   = document.getElementById('btnResume');

const MODEL_URL = './rps_web_model/model.json';
const CLASSES   = ['Rock', 'Paper', 'Scissors'];
const EMOJIS    = ['✊', '✋', '✌️'];
const INPUT_SHAPE = [224, 224]; // expected size for most MobileNet-like heads

let model = null;
let running = true;
let currentFacing = 'environment'; // try rear first on phones
let stream = null;

// Ask for camera (rear if available). Fallback to default if not.
async function setupCamera(facingMode = 'environment') {
  if (stream) {
    // Stop previous tracks if switching cameras
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode,
        width: { ideal: 640 },
        height: { ideal: 640 }
      }
    });
  } catch (e) {
    // Fallback to any camera
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  }

  VIDEO.srcObject = stream;
  await VIDEO.play();
  // Ensure the <video> has dimensions ready for fromPixels
  await new Promise(r => {
    if (VIDEO.readyState >= 2) r();
    else VIDEO.onloadeddata = () => r();
  });
}

// Load the TF.js Layers model
async function loadModel() {
  STATUS.textContent = 'Loading model…';
  model = await tf.loadLayersModel(MODEL_URL);
  // Warm-up pass (allocates WebGL textures once)
  tf.tidy(() => {
    const dummy = tf.zeros([1, INPUT_SHAPE[0], INPUT_SHAPE[1], 3]);
    model.predict(dummy);
  });
  STATUS.textContent = 'Model ready. Predicting…';
  LABEL.textContent = 'Waiting for first prediction…';
  EMOJI.textContent = '👀';
}

// Main prediction loop
function predictLoop() {
  if (!running || !model) {
    requestAnimationFrame(predictLoop);
    return;
  }

  // Use tf.tidy to automatically dispose intermediate tensors
  const { idx, prob } = tf.tidy(() => {
    // Capture frame
    const frame = tf.browser.fromPixels(VIDEO);

    // Preprocess: resize -> float -> [0,1] -> add batch
    const resized = tf.image.resizeBilinear(frame, INPUT_SHAPE, true);
    const norm = resized.toFloat().div(255.0);
    const batched = norm.expandDims(0);

    // Predict
    const logits = model.predict(batched); // shape [1, 3]
    const probs = logits.softmax();        // in case model outputs raw scores

    const arg = probs.argMax(-1);          // [1]
    const idx = arg.dataSync()[0];
    const p   = probs.dataSync()[idx];

    return { idx, prob: p };
  });

  // UI update
  EMOJI.textContent = EMOJIS[idx];
  LABEL.textContent = CLASSES[idx];
  PROB.textContent  = `Confidence: ${(prob * 100).toFixed(1)}%`;

  // Next frame
  requestAnimationFrame(predictLoop);
}

// Buttons: switch / pause / resume
BTN_SW.addEventListener('click', async () => {
  currentFacing = currentFacing === 'environment' ? 'user' : 'environment';
  STATUS.textContent = `Switching camera (${currentFacing})…`;
  BTN_SW.disabled = true;
  try {
    await setupCamera(currentFacing);
    STATUS.textContent = 'Camera ready.';
  } catch (e) {
    console.error(e);
    STATUS.textContent = 'Failed to switch camera.';
  } finally {
    BTN_SW.disabled = false;
  }
});

BTN_PA.addEventListener('click', () => {
  running = false;
  BTN_PA.disabled = true;
  BTN_RE.disabled = false;
  STATUS.textContent = 'Paused.';
});

BTN_RE.addEventListener('click', () => {
  running = true;
  BTN_RE.disabled = true;
  BTN_PA.disabled = false;
  STATUS.textContent = 'Predicting…';
});

// Boot sequence
(async function run() {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      STATUS.textContent = 'Camera API not supported in this browser.';
      return;
    }
    STATUS.textContent = 'Requesting camera…';
    await setupCamera(currentFacing);
    await loadModel();
    BTN_PA.disabled = false;
    predictLoop();
  } catch (err) {
    console.error(err);
    STATUS.textContent = 'Error: could not start camera or load model.';
    LABEL.textContent = 'Check site permissions & model path.';
    EMOJI.textContent = '❌';
  }
})();

// Clean up on page unload (stop camera tracks)
window.addEventListener('beforeunload', () => {
  if (stream) stream.getTracks().forEach(t => t.stop());
});
