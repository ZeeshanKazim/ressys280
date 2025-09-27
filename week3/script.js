/* script.js — Matrix Factorization with TensorFlow.js
   ŷ = dot(userVec, itemVec) + userBias + itemBias  (linear)
*/

let model; // trained tf.Model

let userIdToIndex = new Map();
let itemIdToIndex = new Map();
let indexToUserId = [];
let indexToItemId = [];

let trainUserTensor, trainItemTensor, trainRatingTensor;

// Map userId -> Map(itemId -> rating) for fast ground-truth lookup & filtering
const ratingsByUser = new Map();

const $ = (id) => document.getElementById(id);

/* ---------- Build mappings & UI ---------- */
function buildIndexMaps() {
  const userIds = Array.from(new Set(ratings.map(r => r.userId))).sort((a,b)=>a-b);
  indexToUserId = userIds;
  userIdToIndex = new Map(userIds.map((u, idx) => [u, idx]));

  const itemIds = Array.from(new Set(ratings.map(r => r.itemId))).sort((a,b)=>a-b);
  indexToItemId = itemIds;
  itemIdToIndex = new Map(itemIds.map((i, idx) => [i, idx]));
}

function buildRatingsByUser() {
  ratingsByUser.clear();
  for (const r of ratings) {
    if (!ratingsByUser.has(r.userId)) ratingsByUser.set(r.userId, new Map());
    ratingsByUser.get(r.userId).set(r.itemId, r.rating);
  }
}

function populateDropdowns() {
  const userSel = $('user-select');
  const byTitle = [...movies].sort((a,b)=> a.title.localeCompare(b.title));
  const movieSel = $('movie-select');

  // Users
  for (const u of indexToUserId) {
    const opt = document.createElement('option');
    opt.value = String(u);
    opt.textContent = `User ${u}`;
    userSel.appendChild(opt);
  }

  // Movies (initially: all)
  for (const m of byTitle) {
    const opt = document.createElement('option');
    opt.value = String(m.id);
    opt.textContent = m.title;
    movieSel.appendChild(opt);
  }
}

/** Repopulate movies after user/toggle change. If rated-only is on,
 *  show only titles this user rated, adding ★actual in labels. */
function repopulateMoviesForUser() {
  const userIdRaw = parseInt($('user-select').value, 10);
  const ratedOnly = $('rated-only').checked;
  const movieSel  = $('movie-select');

  // clear all but the placeholder
  while (movieSel.options.length > 1) movieSel.remove(1);

  let list = movies;

  if (!Number.isNaN(userIdRaw) && ratedOnly) {
    const m = ratingsByUser.get(userIdRaw);
    if (m) {
      list = movies
        .filter(x => m.has(x.id))
        .map(x => ({ ...x, __rating: m.get(x.id) }));
    } else {
      list = [];
    }
  }

  const byTitle = [...list].sort((a,b)=> a.title.localeCompare(b.title));
  if (byTitle.length === 0) {
    const opt = document.createElement('option');
    opt.disabled = true;
    opt.textContent = '— no rated movies for this user —';
    movieSel.appendChild(opt);
    return;
  }

  for (const m of byTitle) {
    const opt = document.createElement('option');
    opt.value = String(m.id);
    opt.textContent = (m.__rating != null) ? `${m.title} (★${m.__rating})` : m.title;
    movieSel.appendChild(opt);
  }
}

function buildTrainingTensors() {
  const N = ratings.length;
  const uIdx = new Int32Array(N);
  const iIdx = new Int32Array(N);
  const y    = new Float32Array(N);

  for (let k = 0; k < N; k++) {
    const r = ratings[k];
    uIdx[k] = userIdToIndex.get(r.userId);
    iIdx[k] = itemIdToIndex.get(r.itemId);
    y[k]    = r.rating;
  }

  trainUserTensor   = tf.tensor2d(uIdx, [N, 1], 'int32');
  trainItemTensor   = tf.tensor2d(iIdx, [N, 1], 'int32');
  trainRatingTensor = tf.tensor2d(y,    [N, 1], 'float32');

  return { numUsers: indexToUserId.length, numItems: indexToItemId.length };
}

/* ---------- Define MF model ---------- */
function createModel(numUsers, numMovies, latentDim = 16) {
  const userInput  = tf.input({ shape: [1], dtype: 'int32', name: 'userInput' });
  const movieInput = tf.input({ shape: [1], dtype: 'int32', name: 'movieInput' });

  const userEmbeddingLayer = tf.layers.embedding({
    inputDim: numUsers, outputDim: latentDim, embeddingsInitializer: 'glorotUniform', name: 'userEmbedding'
  });
  const movieEmbeddingLayer = tf.layers.embedding({
    inputDim: numMovies, outputDim: latentDim, embeddingsInitializer: 'glorotUniform', name: 'movieEmbedding'
  });

  const userBiasLayer = tf.layers.embedding({
    inputDim: numUsers, outputDim: 1, embeddingsInitializer: 'zeros', name: 'userBias'
  });
  const itemBiasLayer = tf.layers.embedding({
    inputDim: numMovies, outputDim: 1, embeddingsInitializer: 'zeros', name: 'itemBias'
  });

  const userVec  = tf.layers.flatten().apply(userEmbeddingLayer.apply(userInput));
  const itemVec  = tf.layers.flatten().apply(movieEmbeddingLayer.apply(movieInput));
  const userBias = tf.layers.flatten().apply(userBiasLayer.apply(userInput));
  const itemBias = tf.layers.flatten().apply(itemBiasLayer.apply(movieInput));

  const dot = tf.layers.dot({ axes: 1, name: 'dotUserItem' }).apply([userVec, itemVec]);
  const addDotUser = tf.layers.add().apply([dot, userBias]);
  const pred = tf.layers.add({ name: 'rating' }).apply([addDotUser, itemBias]); // linear

  return tf.model({ inputs: [userInput, movieInput], outputs: pred, name: 'mfRecommender' });
}

/* ---------- Train (with persistent metrics line) ---------- */
async function trainModel() {
  const res = $('result');
  const metricsEl = $('metrics');
  res.textContent = 'Building model…';

  buildIndexMaps();
  const { numUsers: U, numItems: I } = buildTrainingTensors();

  model = createModel(U, I, 16);
  model.compile({ optimizer: tf.train.adam(0.001), loss: 'meanSquaredError' });

  const epochs = 8;
  const batchSize = 256;
  res.textContent = `Training model… (epochs: ${epochs}, batchSize: ${batchSize})`;

  const history = await model.fit(
    [trainUserTensor, trainItemTensor],
    trainRatingTensor,
    {
      epochs,
      batchSize,
      shuffle: true,
      validationSplit: 0.1,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          res.textContent =
            `Training… epoch ${epoch + 1}/${epochs} — loss: ${logs.loss.toFixed(4)} — ` +
            `val_loss: ${logs.val_loss?.toFixed(4)}`;
          await tf.nextFrame();
        }
      }
    }
  );

  // Compute persistent metrics
  const lastValMSE = history.history.val_loss?.at(-1);
  const valRMSE = lastValMSE ? Math.sqrt(lastValMSE) : null;

  const meanTensor = trainRatingTensor.mean();
  const mean = meanTensor.dataSync()[0];
  meanTensor.dispose();

  const mseTensor = tf.tidy(() => trainRatingTensor.sub(tf.scalar(mean)).square().mean());
  const baselineRMSE = Math.sqrt(mseTensor.dataSync()[0]);
  mseTensor.dispose();

  metricsEl.textContent =
    `Val RMSE: ${valRMSE ? valRMSE.toFixed(4) : 'n/a'}  •  Baseline RMSE: ${baselineRMSE.toFixed(4)}`;
  res.textContent = 'Model trained. Select a user and a movie, then click “Predict Rating”.';
}

/* ---------- Predict (shows • actual: <rating> when available) ---------- */
async function predictRating() {
  const res = $('result');
  if (!model) { res.textContent = 'Model not ready yet. Wait for training to finish.'; return; }

  const userIdRaw  = parseInt($('user-select').value, 10);
  const itemIdRaw  = parseInt($('movie-select').value, 10);
  if (Number.isNaN(userIdRaw) || Number.isNaN(itemIdRaw)) {
    res.textContent = 'Please choose both a user and a movie.'; return;
  }

  const uIdx = userIdToIndex.get(userIdRaw);
  const iIdx = itemIdToIndex.get(itemIdRaw);
  if (uIdx == null) { res.textContent = `User ${userIdRaw} has no ratings (cannot embed).`; return; }
  if (iIdx == null) {
    const t = movies.find(m => m.id === itemIdRaw)?.title || 'Movie';
    res.textContent = `“${t}” has no ratings in u.data (no embedding).`; return;
  }

  const uT = tf.tensor2d([uIdx], [1,1], 'int32');
  const iT = tf.tensor2d([iIdx], [1,1], 'int32');
  const yHat = tf.tidy(() => model.predict([uT, iT]));
  const predVal = (await yHat.data())[0];
  uT.dispose(); iT.dispose(); yHat.dispose();

  const title = movies.find(m => m.id === itemIdRaw)?.title || `Movie ${itemIdRaw}`;
  const clamped = Math.max(1, Math.min(5, predVal));

  const actual = ratingsByUser.get(userIdRaw)?.get(itemIdRaw);
  const actualTxt = (actual != null) ? ` • actual: ${actual}` : ' • (no ground-truth rating for this pair)';

  res.textContent =
    `Predicted rating for User ${userIdRaw} on “${title}”: ${clamped.toFixed(2)} ` +
    `(raw: ${predVal.toFixed(3)})${actualTxt}`;
}

/* ---------- Init ---------- */
window.onload = async () => {
  const res = $('result');
  try {
    await loadData();
    res.textContent = `Data loaded: ${numUsers} users, ${numMovies} movies, ${ratings.length} ratings. Preparing…`;

    buildRatingsByUser();
    buildIndexMaps();
    populateDropdowns();

    // Wire up the filter so you can see ★actual options
    $('user-select').addEventListener('change', repopulateMoviesForUser);
    $('rated-only').addEventListener('change', repopulateMoviesForUser);

    await trainModel();
  } catch (e) {
    console.error(e);
    // loadData already sets a friendly error
  }
};
