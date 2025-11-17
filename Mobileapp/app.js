// ---------- Config ----------
const MODEL_PATH = 'rps_web_model/model.json'; // relative to index.html
const CLASS_NAMES = ['Rock', 'Paper', 'Scissors'];
const CLASS_EMOJI = ['✊', '✋', '✌️'];
const INPUT_SIZE = 224;

// ---------- UI refs ----------
const videoEl   = document.getElementById('cam');
const statusBox = document.getElementById('status');
const stateIcon = document.getElementById('stateIcon');
const stateMain = document.getElementById('stateMain');
const stateSub  = document.getElementById('stateSub');
const btnSwitch = document.getElementById('switchBtn');
const btnPause  = document.getElementById('pauseBtn');
const btnResume = document.getElementById('resumeBtn');

// ---------- Globals ----------
let model = null;
let stream = null;
let usingFront = true;
let running = false;

// ---------- Helpers ----------
function setStatus(icon, main, sub='') {
  stateIcon.textContent = icon;
  stateMain.textContent = main;
  stateSub.textContent  = sub;
}

async function setupCamera() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  const constraints = {
    audio: false,
    video: {
      facingMode: usingFront ? 'user' : { exact: 'environment' },
      width: { ideal: 640 }, height: { ideal: 640 }
    }
  };
  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (e) {
    // fallback without exact environment requirement
    if (!usingFront) {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
    } else {
      throw e;
    }
  }
  videoEl.srcObject = stream;
  await videoEl.play();
  return true;
}

async function verifyModelUrl(url) {
  try {
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status} ${res.statusText}`);
    }
    // small sanity check: ensure weightsManifest mentions a .bin
    const j = await res.clone().json();
    const ok = Array.isArray(j.weightsManifest) &&
               j.weightsManifest.length &&
               j.weightsManifest[0].paths &&
               j.weightsManifest[0].paths.length &&
               /\.bin$/i.test(j.weightsManifest[0].paths[0]);
    if (!ok) throw new Error('model.json looks invalid (no .bin path).');
    return true;
  } catch (err) {
    console.error('Model URL check failed:', err);
    setStatus('❌', 'Model file not reachable', `Path: ${url}\nDetails: ${err.message}`);
    throw err;
  }
}

async function loadModel(url) {
  await verifyModelUrl(url);
  setStatus('⏳', 'Loading model…', 'Fetching model.json and weight shard…');
  model = await tf.loadLayersModel(url);
  // warmup
  tf.tidy(() => {
    const warm = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]);
    model.predict(warm);
  });
  setStatus('✅', 'Model loaded', 'Show ✊, ✋, or ✌️ to start predictions.');
}

function preprocessFrame() {
  return tf.tidy(() => {
    const frame = tf.browser.fromPixels(videoEl);
    const resized = tf.image.resizeBilinear(frame, [INPUT_SIZE, INPUT_SIZE], true);
    const norm = resized.toFloat().div(255);
    return norm.expandDims(0); // [1,H,W,3]
  });
}

function showPrediction(idx, prob) {
  const emoji = CLASS_EMOJI[idx] || '❓';
  const name  = CLASS_NAMES[idx] || 'Unknown';
  setStatus(emoji, `${name}`, `Confidence: ${(prob*100).toFixed(1)}%`);
}

function loop() {
  if (!running || !model) return;
  try {
    const batched = preprocessFrame();
    const logits = model.predict(batched);
    const probs = logits.softmax ? logits.softmax() : logits; // if model ends with logits
    const data = probs.dataSync(); // small vector of length 3
    let bestI = 0, bestP = data[0];
    for (let i = 1; i < data.length; i++) if (data[i] > bestP) { bestP = data[i]; bestI = i; }
    showPrediction(bestI, bestP);
    tf.dispose([batched, logits, probs]);
  } catch (e) {
    console.error('Predict error:', e);
    setStatus('⚠️', 'Prediction error', e.message);
    running = false;
    return;
  }
  requestAnimationFrame(loop);
}

// ---------- Controls ----------
btnSwitch.addEventListener('click', async () => {
  usingFront = !usingFront;
  setStatus('📷', 'Switching camera…', usingFront ? 'Using front camera' : 'Using back camera');
  await setupCamera();
});
btnPause.addEventListener('click', () => {
  running = false;
  btnPause.disabled = true;
  btnResume.disabled = false;
  setStatus('⏸️', 'Paused', 'Click Resume to continue.');
});
btnResume.addEventListener('click', () => {
  if (!model) return;
  running = true;
  btnPause.disabled = false;
  btnResume.disabled = true;
  loop();
});

// ---------- Boot ----------
(async function run() {
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus('❌', 'Camera API not available', 'Use a modern browser (Chrome, Edge, Safari).');
      return;
    }
    await setupCamera();
  } catch (e) {
    console.error('Camera error:', e);
    setStatus('❌', 'Could not start camera', e.message);
    return;
  }

  try {
    await loadModel(MODEL_PATH);
  } catch (e) {
    // loadModel already set UI + console, just stop here
    return;
  }

  running = true;
  btnPause.disabled = false;
  btnResume.disabled = true;
  loop();
})();
