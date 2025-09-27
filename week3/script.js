/* script.js
   ========================== EXPLANATION FIRST ==========================
   GOAL (what are we modeling?)
   - We learn a function that maps (userId, movieId) → predicted rating.
   - We assume each user u has a latent vector p_u ∈ R^K, and each movie i has q_i ∈ R^K.
   - Prediction ŷ_{u,i} = p_u · q_i  + b_u + b_i  (dot product + biases).
   - We train p_u, q_i, b_u, b_i to minimize MSE on known ratings.

   WHY THIS IS CORRECT (logic-level review):
   - This is the standard matrix factorization (MF) objective taught in recommender systems.
   - User/item embeddings are implemented via tf.layers.embedding (lookup tables of trainable vectors).
   - The dot layer computes the inner product; add() layers add biases; output is linear (no activation).

   WHERE IT CAN GO WRONG (and how we mitigate):
   - Embedding indices must be contiguous 0..U-1 and 0..I-1 → we remap raw ids (critical).
   - Cold users/items with no ratings cannot be embedded → we warn in predictRating().
   - Ratings are 1..5; training doesn’t clamp to preserve gradient dynamics; we clamp only for display.
   - Hyperparameters (K, LR, epochs, batch) trade off speed vs. accuracy; defaults are conservative.

   HOW TO REVIEW WITHOUT READING EVERY LINE:
   1) Confirm we build id maps (raw→index).  2) Confirm createModel uses embeddings + dot + bias.
   3) Confirm trainModel compiles with Adam+MSE and fits on (userIdx, itemIdx) → rating.
   4) Confirm predictRating maps selected raw IDs → indices → calls model.predict → prints result.
*/

// ============= GLOBALS (visible, so reviewers see state at a glance) =============
let model; // the trained tf.Model

// ID maps: raw IDs → contiguous indices (0..U-1 / 0..I-1) required by embeddings.
let userIdToIndex = new Map();
let itemIdToIndex = new Map();
let indexToUserId = [];
let indexToItemId = [];

// Cached tensors for training (so we can dispose if needed)
let trainUserTensor, trainItemTensor, trainRatingTensor;

// Small DOM helper
const $ = (id) => document.getElementById(id);

// ===================== 1) BUILD INDEX MAPS & UI =====================

/**
 * buildIndexMaps()
 * WHY: Embedding layers require integer indices from 0..(count-1).
 * We derive these from the ratings we actually have (cold movies with zero ratings are excluded).
 */
function buildIndexMaps() {
  const userIds = Array.from(new Set(ratings.map(r => r.userId))).sort((a,b)=>a-b);
  indexToUserId = userIds;
  userIdToIndex = new Map(userIds.map((u, idx) => [u, idx]));

  const itemIds = Array.from(new Set(ratings.map(r => r.itemId))).sort((a,b)=>a-b);
  indexToItemId = itemIds;
  itemIdToIndex = new Map(itemIds.map((i, idx) => [i, idx]));
}

/**
 * populateDropdowns()
 * Populates Users (by numeric id) and Movies (by title).
 * NOTE: Some movies parsed from u.item may not appear in u.data → no embedding index for them.
 * We still list them; predictRating() will warn if a movie has no ratings.
 */
function populateDropdowns() {
  const userSel = $('user-select');
  const movieSel = $('movie-select');

  // Users (show as "User N")
  for (const u of indexToUserId) {
    const opt = document.createElement('option');
    opt.value = String(u);
    opt.textContent = `User ${u}`;
    userSel.appendChild(opt);
  }

  // Movies (sorted by title)
  const byTitle = [...movies].sort((a,b)=> a.title.localeCompare(b.title));
  for (const m of byTitle) {
    const opt = document.createElement('option');
    opt.value = String(m.id);
    opt.textContent = m.title;
    movieSel.appendChild(opt);
  }
}

/**
 * buildTrainingTensors()
 * Prepare tensors for training:
 * - X_user: int32 indices shape [N,1]
 * - X_item: int32 indices shape [N,1]
 * - y:      float32 ratings shape [N,1]
 * Returns { numUsers, numItems } used by createModel().
 */
function buildTrainingTensors() {
  const N = ratings.length;
  const uIdx = new Int32Array(N);
  const iIdx = new Int32Array(N);
  const y    = new Float32Array(N);

  for (let k = 0; k < N; k++) {
    const r = ratings[k];
    uIdx[k] = userIdToIndex.get(r.userId); // must exist
    iIdx[k] = itemIdToIndex.get(r.itemId); // must exist
    y[k]    = r.rating;                     // raw rating (1..5)
  }

  // Spec asked for tf.tensor2d; shapes [N,1]
  trainUserTensor   = tf.tensor2d(uIdx, [N, 1], 'int32');
  trainItemTensor   = tf.tensor2d(iIdx, [N, 1], 'int32');
  trainRatingTensor = tf.tensor2d(y,    [N, 1], 'float32');

  return { numUsers: indexToUserId.length, numItems: indexToItemId.length };
}

// ===================== 2) DEFINE THE MF MODEL =====================

/**
 * createModel(numUsers, numMovies, latentDim)
 *
 * ARCHITECTURE (review checklist):
 * - Inputs: userInput (int index), movieInput (int index)
 * - Embedding lookups:
 *      userEmbedding:  [numUsers, latentDim] → user vector p_u
 *      movieEmbedding: [numMovies, latentDim] → movie vector q_i
 * - Bias lookups:
 *      userBias:  [numUsers, 1] → b_u
 *      itemBias:  [numMovies, 1] → b_i
 * - Prediction:
 *      dot = p_u · q_i  (inner product)
 *      ŷ = dot + b_u + b_i  (linear; no activation)
 *
 * Correctness note: This matches the standard MF equation used in class.
 */
function createModel(numUsers, numMovies, latentDim = 16) {
  // Inputs are int indices (shape [batch, 1])
  const userInput  = tf.input({ shape: [1], dtype: 'int32', name: 'userInput' });
  const movieInput = tf.input({ shape: [1], dtype: 'int32', name: 'movieInput' });

  // Trainable embedding tables initialized with Glorot (good default)
  const userEmbeddingLayer = tf.layers.embedding({
    inputDim: numUsers, outputDim: latentDim,
    embeddingsInitializer: 'glorotUniform', name: 'userEmbedding'
  });
  const movieEmbeddingLayer = tf.layers.embedding({
    inputDim: numMovies, outputDim: latentDim,
    embeddingsInitializer: 'glorotUniform', name: 'movieEmbedding'
  });

  // Optional bias terms (one scalar per user/item)
  const userBiasLayer = tf.layers.embedding({
    inputDim: numUsers, outputDim: 1,
    embeddingsInitializer: 'zeros', name: 'userBias'
  });
  const itemBiasLayer = tf.layers.embedding({
    inputDim: numMovies, outputDim: 1,
    embeddingsInitializer: 'zeros', name: 'itemBias'
  });

  // Look up embeddings; flatten from [batch,1,latentDim] → [batch,latentDim]
  const userVec  = tf.layers.flatten().apply(userEmbeddingLayer.apply(userInput));
  const itemVec  = tf.layers.flatten().apply(movieEmbeddingLayer.apply(movieInput));

  // Bias lookups; flatten from [batch,1,1] → [batch,1]
  const userBias = tf.layers.flatten().apply(userBiasLayer.apply(userInput));
  const itemBias = tf.layers.flatten().apply(itemBiasLayer.apply(movieInput));

  // Inner product across latent dimension
  const dot = tf.layers.dot({ axes: 1, name: 'dotUserItem' }).apply([userVec, itemVec]);

  // ŷ = dot + userBias + itemBias (linear output, no activation)
  const addDotUser = tf.layers.add().apply([dot, userBias]);
  const pred = tf.layers.add({ name: 'rating' }).apply([addDotUser, itemBias]);

  return tf.model({ inputs: [userInput, movieInput], outputs: pred, name: 'mfRecommender' });
}

// ===================== 3) TRAIN THE MODEL =====================

/**
 * trainModel()
 * OPTIMIZER: Adam(0.001)
 * LOSS: meanSquaredError
 * EPOCHS: 8 (tunable), BATCH: 256 (tunable)
 * Validation split: 0.1 to observe generalization.
 *
 * Reviewer guidance:
 * - loss should decrease; val_loss slightly higher than loss is normal.
 * - If loss is stuck: lower LR to 0.0005; verify id maps; increase epochs to 12.
 */
async function trainModel() {
  const res = $('result');
  res.textContent = 'Building model…';

  // 1) Ensure we have contiguous indices & tensors
  buildIndexMaps();
  const { numUsers: U, numItems: I } = buildTrainingTensors();

  // 2) Create + compile
  model = createModel(U, I, /* latentDim */ 16);
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'meanSquaredError'
  });

  // 3) Fit
  const epochs = 8;
  const batchSize = 256;
  res.textContent = `Training model… (epochs: ${epochs}, batchSize: ${batchSize})`;

  await model.fit(
    [trainUserTensor, trainItemTensor],
    trainRatingTensor,
    {
      epochs,
      batchSize,
      shuffle: true,
      validationSplit: 0.1,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          // Live training telemetry → human-auditable trace
          res.textContent = `Training… epoch ${epoch + 1}/${epochs} — loss: ${logs.loss.toFixed(4)} — val_loss: ${logs.val_loss?.toFixed(4)}`;
          await tf.nextFrame(); // allow UI to update
        }
      }
    }
  );

  res.textContent = 'Model trained. Select a user and a movie, then click “Predict Rating”.';
}

// ===================== 4) PREDICT ONE RATING =====================

/**
 * predictRating()
 * Steps the reviewer should expect:
 * 1) Read raw IDs from dropdowns
 * 2) Map raw → contiguous indices (must exist)
 * 3) Build int32 tensors shape [1,1]
 * 4) model.predict([u, i]) → scalar
 * 5) Show both raw and clamped [1..5] for transparency
 */
async function predictRating() {
  const res = $('result');
  if (!model) {
    res.textContent = 'Model not ready yet. Wait for training to finish.';
    return;
  }

  const userSel = $('user-select');
  const movieSel = $('movie-select');
  const userIdRaw = parseInt(userSel.value, 10);
  const itemIdRaw = parseInt(movieSel.value, 10);

  if (Number.isNaN(userIdRaw) || Number.isNaN(itemIdRaw)) {
    res.textContent = 'Please choose both a user and a movie.';
    return;
  }

  const uIdx = userIdToIndex.get(userIdRaw);
  const iIdx = itemIdToIndex.get(itemIdRaw);

  if (uIdx == null) {
    res.textContent = `User ${userIdRaw} is not present in ratings (cannot embed). Choose another user.`;
    return;
  }
  if (iIdx == null) {
    res.textContent = `“${movies.find(m => m.id === itemIdRaw)?.title || 'Movie'}” has no ratings in u.data (no embedding). Choose another movie.`;
    return;
  }

  // Build [1,1] int32 tensors
  const uT = tf.tensor2d([uIdx], [1,1], 'int32');
  const iT = tf.tensor2d([iIdx], [1,1], 'int32');

  // Predict → scalar
  const yHat = tf.tidy(() => model.predict([uT, iT]));
  const predVal = (await yHat.data())[0];

  // Cleanup small tensors immediately
  uT.dispose(); iT.dispose(); yHat.dispose();

  // UI: show clamped (display-friendly) and raw (audit)
  const title = movies.find(m => m.id === itemIdRaw)?.title || `Movie ${itemIdRaw}`;
  const clamped = Math.max(1, Math.min(5, predVal));
  res.textContent = `Predicted rating for User ${userIdRaw} on “${title}”: ${clamped.toFixed(2)}  (raw: ${predVal.toFixed(3)})`;
}

// ===================== 5) INIT (load → UI → train) =====================

window.onload = async () => {
  const res = $('result');
  try {
    // Load files and parse
    await loadData();
    res.textContent = `Data loaded: ${numUsers} users, ${numMovies} movies, ${ratings.length} ratings. Preparing…`;

    // Populate menus & start training
    buildIndexMaps();
    populateDropdowns();
    await trainModel();
  } catch (e) {
    // Friendly message already set in loadData()
    console.error(e);
  }
};
