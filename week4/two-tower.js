/* two-tower.js — Two-Tower Retrieval (baseline & “deep”)
   Fixes: no fixed variable names + proper dispose() to avoid
   “Variable with name ... was already registered”.
   Works with the app.js you’re using.
*/

class TwoTowerModel {
  /**
   * @param {number} numUsers
   * @param {number} numItems
   * @param {number} embDim
   * @param {{
   *   deep: boolean,
   *   hiddenDim: number,
   *   lossType: 'softmax'|'bpr',
   *   learningRate: number,
   *   itemFeatures?: { data: Float32Array, dim: number }  // optional 18-genre flags
   * }} opts
   */
  constructor(numUsers, numItems, embDim, opts) {
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.embDim   = embDim;

    this.deep         = !!opts.deep;
    this.hiddenDim    = Math.max(0, opts.hiddenDim|0);
    this.lossType     = (opts.lossType || 'softmax');
    this.learningRate = +opts.learningRate || 0.003;

    // Optional item features (e.g., 18-dim genres)
    this.itemFeatDim = 0;
    this.itemFeatT   = null;
    if (opts.itemFeatures && opts.itemFeatures.data && opts.itemFeatures.dim > 0) {
      this.itemFeatDim = opts.itemFeatures.dim|0;
      // shape [numItems, featDim]
      this.itemFeatT = tf.tensor2d(opts.itemFeatures.data, [numItems, this.itemFeatDim], 'float32');
    }

    // --- Embedding tables (no explicit names => no collisions) ---
    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05, 'float32'));
    this.itemEmbedding = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05, 'float32'));

    // --- Optional MLP heads (for “deep”) ---
    // User MLP: [embDim] -> [hiddenDim] -> [embDim]
    if (this.deep && this.hiddenDim > 0) {
      this.uW1 = tf.variable(tf.randomNormal([embDim, this.hiddenDim], 0, 0.05, 'float32'));
      this.ub1 = tf.variable(tf.zeros([this.hiddenDim]));
      this.uW2 = tf.variable(tf.randomNormal([this.hiddenDim, embDim], 0, 0.05, 'float32'));
      this.ub2 = tf.variable(tf.zeros([embDim]));

      // Item tower can use optional genre features: concat([embDim] , [featDim]) -> hidden -> embDim
      const itemInDim = embDim + this.itemFeatDim;
      this.iW1 = tf.variable(tf.randomNormal([itemInDim, this.hiddenDim], 0, 0.05, 'float32'));
      this.ib1 = tf.variable(tf.zeros([this.hiddenDim]));
      this.iW2 = tf.variable(tf.randomNormal([this.hiddenDim, embDim], 0, 0.05, 'float32'));
      this.ib2 = tf.variable(tf.zeros([embDim]));
    } else {
      this.uW1 = this.ub1 = this.uW2 = this.ub2 = null;
      this.iW1 = this.ib1 = this.iW2 = this.ib2 = null;
    }

    this.opt = tf.train.adam(this.learningRate);
  }

  /* ---------- housekeeping ---------- */
  dispose() {
    const arr = [
      this.userEmbedding, this.itemEmbedding,
      this.uW1, this.ub1, this.uW2, this.ub2,
      this.iW1, this.ib1, this.iW2, this.ib2,
      this.itemFeatT
    ].filter(Boolean);
    tf.dispose(arr);
  }

  /* ---------- gather ---------- */
  userForward(idx1d) {
    // idx1d: int32 [B]
    return tf.tidy(() => {
      const u = tf.gather(this.userEmbedding, idx1d); // [B, embDim]
      if (!(this.deep && this.hiddenDim > 0)) return u; // baseline
      const h1 = tf.relu(tf.add(tf.matMul(u, this.uW1), this.ub1));   // [B, hidden]
      const o  = tf.add(tf.matMul(h1, this.uW2), this.ub2);           // [B, embDim]
      return o;
    });
  }

  itemForward(idx1d) {
    // idx1d: int32 [B]
    return tf.tidy(() => {
      const e = tf.gather(this.itemEmbedding, idx1d); // [B, embDim]
      if (!(this.deep && this.hiddenDim > 0)) return e; // baseline

      let x = e;
      if (this.itemFeatT) {
        const f = tf.gather(this.itemFeatT, idx1d);   // [B, featDim]
        x = tf.concat([e, f], 1);                     // [B, embDim+featDim]
      }
      const h1 = tf.relu(tf.add(tf.matMul(x, this.iW1), this.ib1));   // [B, hidden]
      const o  = tf.add(tf.matMul(h1, this.iW2), this.ib2);           // [B, embDim]
      return o;
    });
  }

  /* ---------- scoring ---------- */
  score(uEmb, iEmb) {
    // uEmb/iEmb: [B, embDim] -> returns [B] (diag of dot products)
    return tf.tidy(() => {
      const dots = tf.sum(tf.mul(uEmb, iEmb), -1); // [B]
      return dots;
    });
  }

  /* ---------- training step ---------- */
  async trainStep(uIdx, posIdx) {
    const lossVal = await this.opt.minimize(() => {
      return tf.tidy(() => {
        const uEmb = this.userForward(uIdx);   // [B, D]
        const pEmb = this.itemForward(posIdx); // [B, D]

        if (this.lossType === 'bpr') {
          // BPR: sample negatives of same shape
          const B = uIdx.shape[0];
          const negIdx = tf.randomUniform([B], 0, this.numItems, 'int32');
          const nEmb   = this.itemForward(negIdx);

          const sp = this.score(uEmb, pEmb); // [B]
          const sn = this.score(uEmb, nEmb); // [B]
          const diff = tf.sub(sp, sn);       // [B]
          const loss = tf.neg(tf.mean(tf.logSigmoid(diff))); // -mean(log σ(sp-sn)))
          return loss;
        }

        // In-batch softmax: logits = U @ P^T, labels = identity
        const logits = tf.matMul(uEmb, pEmb, false, true); // [B, B]
        const labels = tf.eye(logits.shape[0]);            // [B, B]
        const loss   = tf.losses.softmaxCrossEntropy(labels, logits).mean();
        return loss;
      });
    }, /* returnCost= */ true);

    const v = (await lossVal.data())[0];
    lossVal.dispose();
    return v;
  }

  /* ---------- inference ---------- */
  getUserEmbedding(uIdxScalar) {
    return tf.tidy(() => {
      const u = tf.tensor1d([uIdxScalar], 'int32');
      const e = this.userForward(u);     // [1, D]
      const out = e.squeeze();           // [D]
      const arr = out.dataSync();        // Float32Array
      u.dispose(); e.dispose(); out.dispose();
      return arr;
    });
  }

  /** scores for all items for one user (returns Float32Array of length numItems) */
  getScoresForAllItems(uIdxScalar) {
    return tf.tidy(() => {
      const u = tf.tensor1d([uIdxScalar], 'int32');    // [1]
      const ue = this.userForward(u).squeeze();        // [D]

      // Compute item tower outputs in batches to keep memory small
      const B = 1024;
      const scores = new Float32Array(this.numItems);
      for (let i = 0; i < this.numItems; i += B) {
        const end = Math.min(i + B, this.numItems);
        const idx = tf.tensor1d([...Array(end - i)].map((_,k)=>i+k), 'int32'); // [b]
        const ie  = this.itemForward(idx);                                     // [b, D]
        const s   = tf.matMul(ie, ue.reshape([this.embDim, 1])).squeeze();     // [b]
        const chunk = s.dataSync(); scores.set(chunk, i);
        tf.dispose([idx, ie, s]);
      }

      tf.dispose([u, ue]);
      return scores;
    });
  }

  /** All item tower outputs as a flat Float32Array [numItems * embDim] */
  getAllItemEmbeddings() {
    return tf.tidy(() => {
      const out = new Float32Array(this.numItems * this.embDim);
      const B = 1024;
      for (let i = 0; i < this.numItems; i += B) {
        const end = Math.min(i + B, this.numItems);
        const idx = tf.tensor1d([...Array(end - i)].map((_,k)=>i+k), 'int32');
        const emb = this.itemForward(idx); // [b, D]
        const arr = emb.dataSync();
        out.set(arr, i * this.embDim);
        tf.dispose([idx, emb]);
      }
      return out;
    });
  }
}

window.TwoTowerModel = TwoTowerModel;
