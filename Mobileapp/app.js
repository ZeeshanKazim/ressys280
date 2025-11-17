/* app.js — robust loader + camera + real-time prediction */

const VIDEO   = document.getElementById('video');
const EMOJI   = document.getElementById('pred-emoji');
const LABEL   = document.getElementById('pred-label');
const STATUS  = document.getElementById('status');

const CLASSES = ['Rock', 'Paper', 'Scissors'];
const EMOJIS  = ['✊', '✋', '✌️'];

// Absolute Pages URL first, then fallbacks:
const MODEL_CANDIDATES = [
  '/ressys280/Mobileapp/rps_web_model/model.json',
  './rps_web_model/model.json',
  'rps_web_model/model.json'
];

let model = null;
let predicting = false;

/* ------- Camera ------- */
async function setupCamera() {
  STATUS.textContent = 'Requesting camera…';
  try {
    const constraints = {
      audio: false,
      video: {
        facingMode: { ideal: 'environment' }, // rear if available
        width: { ideal: 640 },
        height: { ideal: 640 }
      }
    };

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch {
      // fallback to default camera
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    }

    VIDEO.srcObject = stream;
    await VIDEO.play();
    STATUS.textContent = 'Camera ready.';
  } catch (err) {
    console.error('Camera error:', err);
    STATUS.textContent = 'Camera error. Allow permissions and reload.';
    LABEL.textContent  = err?.message || 'Unable to start camera.';
    EMOJI.textContent  = '❌';
    throw err;
  }
}

/* ------- Model ------- */
async function loadModel() {
  STATUS.textContent = 'Loading model…';

  // Try candidates in order; keep the one that loads
  for (const url of MODEL_CANDIDATES) {
    try {
      // quick probe so we can show a clearer error if 404
      const probe = await fetch(url, { method: 'GET' });
      if (!probe.ok) throw new Error(`HTTP ${probe.status} for ${url}`);

      model = await tf.loadLayersModel(url);
      console.log('Loaded model from:', url);

      // sanity check: RPS should output 3 classes
      const out = model.outputs?.[0];
      const outSize = Array.isArray(out?.shape) ? out.shape[out.shape.length - 1] : null;
      if (outSize !== 3) {
        throw new Error(`Wrong model: output size ${outSize}, expected 3.`);
      }

      // warmup
      tf.tidy(() => model.predict(tf.zeros([1, 224, 224, 3])));
      STATUS.textContent = 'Model ready. Predicting…';
      return;
    } catch (e) {
      console.warn('Load failed for', url, e);
      // try next candidate
    }
  }

  const msg = 'Model load failed. Check folder path and file names.';
  STATUS.textContent = msg;
  LABEL.textContent  = msg + ' See console for details.';
  EMOJI.textContent  = '❌';
  throw new Error(msg);
}

/* ------- Preprocess frame to 224x224 float ------- */
function preprocess() {
  return tf.tidy(() => {
    const frame = tf.browser.fromPixels(VIDEO);         // [h,w,3] uint8
    const resized = tf.image.resizeBilinear(frame, [224, 224]);
    const float   = resized.toFloat().div(255.0);       // [0,1]
    const batched = float.expandDims(0);                // [1,224,224,3]
    return batched;
  });
}

/* ------- Prediction loop ------- */
async function predictLoop() {
  predicting = true;
  const step = () => {
    if (!predicting || !model) return;
    try {
      const input  = preprocess();
      const logits = model.predict(input);
      const probs  = Array.from(logits.dataSync());     // [3]
      tf.dispose([input, logits]);

      // argmax
      let bestIdx = 0, best = -Infinity;
      for (let i = 0; i < probs.length; i++) if (probs[i] > best) { best = probs[i]; bestIdx = i; }

      EMOJI.textContent = EMOJIS[bestIdx] || '🤖';
      LABEL.textContent = `${CLASSES[bestIdx]} (${best.toFixed(2)})`;
      STATUS.textContent = 'Predicting…';
    } catch (err) {
      console.error('Predict error:', err);
      STATUS.textContent = 'Prediction error.';
      LABEL.textContent  = err?.message || 'Unknown error';
      EMOJI.textContent  = '⚠️';
    }
    requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/* ------- Controls (optional buttons if you have them) ------- */
window.pausePred = () => { predicting = false; STATUS.textContent = 'Paused.'; };
window.resumePred = () => { if (!predicting) predictLoop(); };

/* ------- Boot ------- */
(async function run() {
  try {
    await setupCamera();
    await loadModel();
    await predictLoop();
  } catch (e) {
    console.error('Startup error:', e);
    STATUS.textContent = 'Error: could not start camera or load model.';
    EMOJI.textContent  = '❌';
  }
})();
