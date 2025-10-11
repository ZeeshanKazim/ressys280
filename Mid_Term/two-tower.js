/* two-tower.js
   Minimal Two‑Tower retrieval in TensorFlow.js with an optional deep (MLP) item tower.
   Baseline loss = in‑batch sampled softmax (users × positive items in the same batch).
   Deep tower = item_id embedding + MLP(tag_features) → item representation.
*/

class TwoTowerModel {
  constructor(numUsers, numItems, embDim = 32, opts = {}) {
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.embDim = embDim;
    this.lr = opts.learningRate ?? 1e-3;

    // Make names unique to avoid "Variable already registered" when retraining.
    this.uid = `tt_${Date.now()}_${Math.random().toString(36).slice(2)}`;

    this.userEmbedding = tf.variable(
      tf.randomNormal([numUsers, embDim], 0, 0.05),
      true,
      `uEmb_${this.uid}`
    );
    this.itemEmbedding = tf.variable(
      tf.randomNormal([numItems, embDim], 0, 0.05),
      true,
      `iEmb_${this.uid}`
    );

    this.optimizer = tf.train.adam(this.lr);
  }

  async compile() {
    // nothing else to do for baseline
    return;
  }

  userForward(uIdx) {
    return tf.gather(this.userEmbedding, uIdx); // [B,D]
  }

  itemForward(iIdx) {
    return tf.gather(this.itemEmbedding, iIdx); // [B,D]
  }

  // In‑batch softmax: logits = U @ I^T, labels = diag
  async trainStep(uIdx, iIdx) {
    const lossVal = await this.optimizer.minimize(() => {
      const U = this.userForward(uIdx);              // [B,D]
      const I = this.itemForward(iIdx);              // [B,D]
      const logits = tf.matMul(U, I, false, true);   // [B,B]

      const b = logits.shape[0];
      const labels = tf.eye(b);
      // softmaxCrossEntropy expects logits; returns [B]
      const per = tf.losses.softmaxCrossEntropy(labels, logits);
      const loss = per.mean();

      // tiny L2
      const reg = tf.add(
        tf.mul(1e-6, tf.sum(tf.square(U))),
        tf.mul(1e-6, tf.sum(tf.square(I)))
      );
      return tf.add(loss, reg);
    }, true).data();

    return (await lossVal)[0];
  }

  // Score one user against ALL items
  scoreUserAgainstAll(uIdx) {
    return tf.tidy(() => {
      const U = this.userForward(uIdx);             // [1,D] or [B,D]
      const allI = this.itemEmbedding;              // [N,D]
      return tf.matMul(U, allI, false, true).squeeze(); // [N] (for 1 user)
    });
  }

  // Score one user against a candidate list of indices
  scoreItems(uIdx, itemIndicesArray) {
    return tf.tidy(() => {
      const U = this.userForward(uIdx); // [1,D]
      const I = tf.gather(this.itemEmbedding, tf.tensor1d(itemIndicesArray, 'int32')); // [K,D]
      return tf.matMul(U, I, false, true).squeeze(); // [K]
    });
  }

  getItemEmbMatrix() {
    return this.itemEmbedding.read(); // Tensor2D [numItems, D]
  }

  dispose() {
    this.userEmbedding?.dispose();
    this.itemEmbedding?.dispose();
    this.optimizer?.dispose?.();
  }
}

/** Deep two‑tower:
 * item tower = learnable item_id embedding + MLP(tag one‑hot / multi‑hot)
 * user tower = ID embedding (like baseline)
 */
class DeepTwoTowerModel {
  constructor(numUsers, numItems, embDim = 32, tagDim = 200, opts = {}) {
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.embDim = embDim;
    this.tagDim = tagDim;
    this.lr = opts.learningRate ?? 1e-3;

    this.uid = `dt_${Date.now()}_${Math.random().toString(36).slice(2)}`;

    // Towers
    this.userEmbedding = tf.variable(
      tf.randomNormal([numUsers, embDim], 0, 0.05),
      true,
      `duEmb_${this.uid}`
    );
    this.itemEmbedding = tf.variable(
      tf.randomNormal([numItems, embDim], 0, 0.05),
      true,
      `diEmb_${this.uid}`
    );

    // MLP weights (tags → emb)
    const h = Math.max(embDim * 2, 32);
    this.W1 = tf.variable(tf.randomNormal([tagDim, h], 0, 0.05), true, `W1_${this.uid}`);
    this.b1 = tf.variable(tf.zeros([h]), true, `b1_${this.uid}`);
    this.W2 = tf.variable(tf.randomNormal([h, embDim], 0, 0.05), true, `W2_${this.uid}`);
    this.b2 = tf.variable(tf.zeros([embDim]), true, `b2_${this.uid}`);

    this.optimizer = tf.train.adam(this.lr);

    // Provided at compile time
    this.itemTagMat = null; // Tensor2D [numItems, tagDim]
  }

  async compile(itemTagMat) {
    // Keep a dedicated, immutable copy (used during training + inference)
    this.itemTagMat?.dispose();
    this.itemTagMat = itemTagMat.clone();
  }

  userForward(uIdx) {
    return tf.gather(this.userEmbedding, uIdx); // [B,D]
  }

  itemForward(iIdx) {
    // id emb + mlp(tags)
    const idPart = tf.gather(this.itemEmbedding, iIdx); // [B,D]
    const tags = tf.gather(this.itemTagMat, iIdx);      // [B,K]
    const h1 = tf.relu(tf.add(tf.matMul(tags, this.W1), this.b1)); // [B,H]
    const mlp = tf.add(tf.matMul(h1, this.W2), this.b2); // [B,D]
    return tf.add(idPart, mlp); // [B,D]
  }

  async trainStep(uIdx, iIdx) {
    const lossVal = await this.optimizer.minimize(() => {
      const U = this.userForward(uIdx);              // [B,D]
      const I = this.itemForward(iIdx);              // [B,D]
      const logits = tf.matMul(U, I, false, true);   // [B,B]
      const b = logits.shape[0];
      const labels = tf.eye(b);

      const per = tf.losses.softmaxCrossEntropy(labels, logits);
      const loss = per.mean();

      // light L2
      const reg = tf.addN([
        tf.mul(1e-6, tf.sum(tf.square(U))),
        tf.mul(1e-6, tf.sum(tf.square(I))),
        tf.mul(1e-6, tf.sum(tf.square(this.W1))),
        tf.mul(1e-6, tf.sum(tf.square(this.W2)))
      ]);
      return tf.add(loss, reg);
    }, true).data();

    return (await lossVal)[0];
  }

  // Build full item matrix (id emb + mlp(tags))
  getFrozenItemEmb() {
    return tf.tidy(() => {
      const h1 = tf.relu(tf.add(tf.matMul(this.itemTagMat, this.W1), this.b1)); // [N,H]
      const mlp = tf.add(tf.matMul(h1, this.W2), this.b2);                      // [N,D]
      return tf.add(this.itemEmbedding, mlp);                                   // [N,D]
    });
  }

  scoreUserAgainstAll(uIdx) {
    return tf.tidy(() => {
      const U = this.userForward(uIdx); // [1,D]
      const I = this.getFrozenItemEmb(); // [N,D]
      const s = tf.matMul(U, I, false, true).squeeze(); // [N]
      I.dispose();
      return s;
    });
  }

  scoreItems(uIdx, itemIndicesArray) {
    return tf.tidy(() => {
      const U = this.userForward(uIdx); // [1,D]
      const I = tf.gather(this.getFrozenItemEmb(), tf.tensor1d(itemIndicesArray, 'int32')); // [K,D]
      const s = tf.matMul(U, I, false, true).squeeze(); // [K]
      I.dispose();
      return s;
    });
  }

  dispose() {
    this.userEmbedding?.dispose();
    this.itemEmbedding?.dispose();
    this.W1?.dispose(); this.b1?.dispose(); this.W2?.dispose(); this.b2?.dispose();
    this.itemTagMat?.dispose();
    this.optimizer?.dispose?.();
  }
}

// Expose to window
window.TwoTowerModel = TwoTowerModel;
window.DeepTwoTowerModel = DeepTwoTowerModel;
