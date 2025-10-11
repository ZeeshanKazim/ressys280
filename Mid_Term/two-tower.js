/* two-tower.js
   Minimal Two‑Tower retriever in TensorFlow.js

   Baseline:
     - userEmbedding:  [numUsers, embDim] (trainable)
     - itemEmbedding:  [numItems, embDim] (trainable)
     - Loss: in‑batch sampled softmax (InfoNCE style)

   Deep:
     - user ID embedding: [numUsers, embDim] (trainable)
     - item tower: base ID embedding + MLP(tags) → embDim
       (at least one hidden layer; here 1 hidden layer ReLU)
*/

class TwoTowerModel {
  constructor(numUsers, numItems, embDim = 32, opts = {}) {
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.embDim = embDim;
    this.lr = opts.learningRate || 1e-3;

    // Variables (no fixed names to avoid "already registered" collisions)
    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05));
    this.itemEmbedding = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05));

    this.optimizer = tf.train.adam(this.lr);
  }

  async compile() { /* symmetry with Deep model */ }

  // Lookups
  userForward(userIdxTensor) {             // [B]
    return tf.gather(this.userEmbedding, userIdxTensor); // [B, D]
  }
  itemForward(itemIdxTensor) {             // [B]
    return tf.gather(this.itemEmbedding, itemIdxTensor); // [B, D]
  }

  // In‑batch sampled softmax:
  // logits = U @ I^T  (B x B), labels = eye(B)
  trainStep(userIdxTensor, itemIdxTensor) {
    return tf.tidy(() => {
      const B = userIdxTensor.shape[0];
      const oneHot = tf.oneHot(tf.range(0, B, 1, 'int32'), B); // [B,B]

      let lossVal;
      this.optimizer.minimize(() => {
        const uEmb = this.userForward(userIdxTensor); // [B,D]
        const iEmb = this.itemForward(itemIdxTensor); // [B,D]
        const logits = tf.matMul(uEmb, iEmb, false, true); // [B,B]
        const lossVec = tf.losses.softmaxCrossEntropy(oneHot, logits);
        const loss = tf.mean(lossVec);
        lossVal = loss.dataSync()[0];
        return loss;
      }, /* returnCost */ false, [
        this.userEmbedding, this.itemEmbedding
      ]);

      return lossVal;
    });
  }

  // Score user u against all items: returns [numItems]
  scoreUserAgainstAll(userIdxTensor) {
    return tf.tidy(() => {
      const u = this.userForward(userIdxTensor);              // [1,D]
      const allI = this.itemEmbedding;                        // [I,D]
      const logits = tf.matMul(u, allI, false, true);         // [1, I]
      return logits.squeeze();                                 // [I]
    });
  }

  // Score user u for a set of item indices (Array<int>)
  scoreItems(userIdxTensor, itemIdxArray) {
    return tf.tidy(() => {
      const u = this.userForward(userIdxTensor);              // [1,D]
      const iIdx = tf.tensor1d(itemIdxArray, 'int32');        // [K]
      const i = tf.gather(this.itemEmbedding, iIdx);          // [K,D]
      const logits = tf.matMul(i, u, false, true).squeeze();  // [K]
      iIdx.dispose();
      return logits;
    });
  }

  // For projection
  readItemEmbedding() { return this.itemEmbedding; }

  dispose() {
    this.userEmbedding?.dispose();
    this.itemEmbedding?.dispose();
    // Optimizer has no dispose in tfjs 4.x, will GC.
  }
}

class DeepTwoTowerModel {
  constructor(numUsers, numItems, embDim = 32, tagDim = 200, opts = {}) {
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.embDim = embDim;
    this.tagDim = tagDim;
    this.lr = opts.learningRate || 1e-3;

    // User ID table
    this.userIdEmb = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05));
    // Base item ID table (lets the model capture idiosyncrasies)
    this.itemIdEmb = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05));

    // MLP for item tags: tagDim -> hidden -> embDim
    const hidden = Math.max(32, Math.min(256, Math.round(4 * Math.sqrt(embDim * tagDim))));
    this.W1 = tf.variable(tf.randomNormal([tagDim, hidden], 0, 0.05));
    this.b1 = tf.variable(tf.zeros([hidden]));
    this.W2 = tf.variable(tf.randomNormal([hidden, embDim], 0, 0.05));
    this.b2 = tf.variable(tf.zeros([embDim]));

    this.optimizer = tf.train.adam(this.lr);

    this.itemTagMat = null;     // set in compile()
  }

  async compile(itemTagMatrix) {
    // Expect a dense 0/1 matrix [numItems, tagDim] (float32)
    this.itemTagMat = itemTagMatrix;
  }

  // MLP(tags) for a batch: tags [B,tagDim] -> [B,embDim]
  tagsToEmbedding(tagsBatch) {
    const h1 = tf.relu(tf.add(tf.matMul(tagsBatch, this.W1), this.b1)); // [B,H]
    return tf.add(tf.matMul(h1, this.W2), this.b2);                      // [B,D]
  }

  userForward(userIdxTensor) {
    return tf.gather(this.userIdEmb, userIdxTensor); // [B,D]
  }

  // Batch of item indices -> fused embedding (ID + tagsMLP)
  itemForward(itemIdxTensor) {
    const idPart = tf.gather(this.itemIdEmb, itemIdxTensor); // [B,D]
    const tags = tf.gather(this.itemTagMat, itemIdxTensor);  // [B,K]
    const tagPart = this.tagsToEmbedding(tags);              // [B,D]
    return tf.add(idPart, tagPart);                          // [B,D]
  }

  // In‑batch softmax
  trainStep(userIdxTensor, itemIdxTensor) {
    return tf.tidy(() => {
      const B = userIdxTensor.shape[0];
      const oneHot = tf.oneHot(tf.range(0, B, 1, 'int32'), B);

      let lossVal;
      this.optimizer.minimize(() => {
        const uEmb = this.userForward(userIdxTensor); // [B,D]
        const iEmb = this.itemForward(itemIdxTensor); // [B,D]
        const logits = tf.matMul(uEmb, iEmb, false, true); // [B,B]
        const lossVec = tf.losses.softmaxCrossEntropy(oneHot, logits);
        const loss = tf.mean(lossVec);
        lossVal = loss.dataSync()[0];
        return loss;
      }, false, [
        this.userIdEmb, this.itemIdEmb, this.W1, this.b1, this.W2, this.b2
      ]);

      return lossVal;
    });
  }

  // Materialize full item table (for fast scoring & projection)
  getFrozenItemEmb() {
    return tf.tidy(() => {
      const tagsPart = this.tagsToEmbedding(this.itemTagMat); // [I,D]
      return tf.add(this.itemIdEmb, tagsPart);                // [I,D]
    });
  }

  scoreUserAgainstAll(userIdxTensor) {
    return tf.tidy(() => {
      const u = this.userForward(userIdxTensor);      // [1,D]
      const all = this.getFrozenItemEmb();            // [I,D]
      const logits = tf.matMul(u, all, false, true);  // [1,I]
      all.dispose();
      return logits.squeeze();                         // [I]
    });
  }

  scoreItems(userIdxTensor, itemIdxArray) {
    return tf.tidy(() => {
      const u = this.userForward(userIdxTensor);               // [1,D]
      const idx = tf.tensor1d(itemIdxArray, 'int32');          // [K]
      const i = this.itemForward(idx);                         // [K,D]
      const logits = tf.matMul(i, u, false, true).squeeze();   // [K]
      idx.dispose();
      return logits;
    });
  }

  dispose() {
    this.userIdEmb?.dispose();
    this.itemIdEmb?.dispose();
    this.W1?.dispose(); this.b1?.dispose(); this.W2?.dispose(); this.b2?.dispose();
    // itemTagMat is owned by app.js; do not dispose here.
  }
}

// Expose to window
window.TwoTowerModel = TwoTowerModel;
window.DeepTwoTowerModel = DeepTwoTowerModel;
