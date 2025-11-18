// ---------- Config ----------
const VIDEO = document.getElementById('cam');
const STATUS = document.getElementById('statusText');
const PRED = document.getElementById('predText');
const BTN_SWITCH = document.getElementById('btnSwitch');
const BTN_PAUSE = document.getElementById('btnPause');
const BTN_RESUME = document.getElementById('btnResume');
const MODEL_REL = 'rps_web_model/model.json';

// Classes and emojis (edit to match your model)
const CLASSES = ['Rock', 'Paper', 'Scissors'];
const EMOJI   = ['✊', '✋', '✌️'];

let model = null;
let stream = null;
let usingFront = true;
let running = true;

// Cache buster so GitHub Pages never serves stale 404/old bin
const BUST = `?v=${Date.now()}`;

// ---------- Helpers ----------
const setStatus = (msg, cls) => {
  STATUS.textContent = msg;
  STATUS.parentElement.classList.toggle('ok', cls === 'ok');
  STATUS.parentElement.classList.toggle('bad', cls === 'bad');
};

const setPred = (text) => { PRED.textContent = text; };

// ---------- Camera ----------
async function setupCamera() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
  }
  const constraints = {
    audio: false,
    video: {
      facingMode: usingFront ? 'user' : 'environment',
      width: { ideal: 640 }, height: { ideal: 640 }
    }
  };
  stream = await navigator.mediaDevices.getUserMedia(constraints);
  VIDEO.srcObject = stream;
  await VIDEO.play();
}

// ---------- Model loader with cache-busting ----------
async function loadModel() {
  setStatus('Fetching model.json and weight shard…');
  // Use browserHTTPRequest so we can add a weightUrlConverter and no-store
  const url = new URL(`./${MODEL_REL}${BUST}`, window.location.href).toString();
  const ioHandler = tf.io.browserHTTPRequest(url, {
    credentials: 'same-origin',
    cache: 'no-store',
    // Called for each weight file path inside model.json
    weightUrlConverter: (w) => `${w}${BUST}`
  });
  model = await tf.loadLayersModel(ioHandler);
  setStatus('Model loaded ✓', 'ok');

  // Sanity-check: if this isn't an image model, warn loudly
  const ishape = model.inputs?.[0]?.shape; // e.g. [null, 224, 224, 3]
  if (!(ishape && ishape.length === 4 && ishape[3] === 3)) {
    setStatus(
      `Loaded a non-image model (expects ${JSON.stringify(ishape)}). ` +
      `This app needs an image CNN (e.g., [null,224,224,3]).`,
      'bad'
    );
  }
}

// ---------- Prediction loop ----------
function preprocessFromVideo(size) {
  // Returns a 4D float32 tensor [1, size, size, 3]
  return tf.tidy(() => {
    let t = tf.browser.fromPixels(VIDEO);
    t = tf.image.resizeBilinear(t, [size, size], true);
    return t.expandDims(0).toFloat().div(255.0);
  });
}

async function loop() {
  if (!running || !model) { requestAnimationFrame(loop); return; }

  try {
    // Try to infer expected input size from model
    const ishape = model.inputs?.[0]?.shape;
    const size = Number(ishape?.[1]) || 224;

    const input = preprocessFromVideo(size);
    const probs = model.predict(input);
    const data = (probs.arraySync && probs.arraySync()[0]) || (await probs.data());
    tf.dispose([input, probs]);

    if (!data || !data.length) {
      setPred('—');
    } else {
      const argmax = data.indexOf(Math.max(...data));
      setPred(`${EMOJI[argmax] || ''} ${CLASSES[argmax] || ''} (${(data[argmax]*100).toFixed(0)}%)`);
    }
  } catch (err) {
    console.error(err);
    setStatus(`Prediction error: ${err.message}`, 'bad');
    setPred('—');
    // Keep the loop alive so you can fix without reload
  }

  requestAnimationFrame(loop);
}

// ---------- UI ----------
BTN_SWITCH.addEventListener('click', async () => {
  usingFront = !usingFront;
  try {
    await setupCamera();
  } catch (e) {
    console.error(e);
    setStatus('Camera error: ' + e.message, 'bad');
  }
});
BTN_PAUSE.addEventListener('click', () => { running = false; setStatus('Paused'); });
BTN_RESUME.addEventListener('click', () => { running = true; setStatus('Model loaded ✓', 'ok'); });

// ---------- Boot ----------
(async function run() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus('This browser does not support camera access.', 'bad');
    return;
  }
  try {
    await setupCamera();
  } catch (e) {
    console.error(e);
    setStatus('Camera permission denied. Allow camera and refresh.', 'bad');
    return;
  }
  try {
    await loadModel();
  } catch (e) {
    console.error(e);
    setStatus('Failed to load model. Check file path & names.', 'bad');
    return;
  }
  setStatus('Predicting…', 'ok');
  loop();
})();
