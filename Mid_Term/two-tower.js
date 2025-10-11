/* two-tower.js
   Minimal Two-Tower implementations (baseline & deep) + training helpers.
   Uses in-batch sampled-softmax (logits = U @ I^T, labels = diagonal).
*/

const TwoTower = (() => {
  "use strict";

  class TwoTowerModel {
    constructor(numUsers, numItems, embDim=32){
      this.numUsers = numUsers;
      this.numItems = numItems;
      this.embDim = embDim;

      // Embedding tables
      this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05), true, "userEmbedding");
      this.itemEmbedding = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05), true, "itemEmbedding");
    }
    userForward(idx1D){ return tf.gather(this.userEmbedding, idx1D); }
    itemForward(idx1D){ return tf.gather(this.itemEmbedding, idx1D); }
    score(uEmb, iEmb){ return tf.sum(tf.mul(uEmb, iEmb), -1); } // dot
    dispose(){
      this.userEmbedding.dispose();
      this.itemEmbedding.dispose();
    }
  }

  // Deep tower: same user embedding; item tower = (ID emb + tag MLP)
  class TwoTowerDeep extends TwoTowerModel {
    constructor(numUsers, numItems, embDim, tagMatrix, tagDim){
      super(numUsers, numItems, embDim);
      // tagMatrix: Float32Array (numItems x tagDim)
      this.tagDim = tagDim|0;
      this.tagTensor = tf.tensor2d(tagMatrix, [numItems, this.tagDim], "float32");
      // simple MLP: tagDim -> 128 -> embDim
      this.w1 = tf.variable(tf.randomNormal([this.tagDim, 128], 0, 0.05), true, "w1");
      this.b1 = tf.variable(tf.zeros([128]), true, "b1");
      this.w2 = tf.variable(tf.randomNormal([128, embDim], 0, 0.05), true, "w2");
      this.b2 = tf.variable(tf.zeros([embDim]), true, "b2");
    }
    itemForward(idx1D){
      const idEmb = super.itemForward(idx1D); // [B, D]
      const tags = tf.gather(this.tagTensor, idx1D); // [B, K]
      const mlp = tf.relu(tags.matMul(this.w1).add(this.b1)).matMul(this.w2).add(this.b2); // [B,D]
      return idEmb.add(mlp.mul(0.5)); // blend
    }
    dispose(){
      super.dispose();
      this.tagTensor.dispose(); this.w1.dispose(); this.b1.dispose(); this.w2.dispose(); this.b2.dispose();
    }
  }

  async function trainInBatchSoftmax(model, batches, cfg){
    const epochs = cfg.epochs|0;
    const bs     = cfg.batchSize|0;
    const lr     = cfg.lr || 0.001;
    const opt = tf.train.adam(lr);

    // prepack integers into tensors
    const U = batches.map(b => b.u);
    const I = batches.map(b => b.i);

    for (let ep=0; ep<epochs; ep++){
      // shuffle indices
      const order = tf.util.createShuffledIndices(batches.length);
      for (let s=0; s<order.length; s+=bs){
        const idx = order.slice(s, s+bs);
        const u = tf.tensor1d(idx.map(j => U[j]), "int32");
        const i = tf.tensor1d(idx.map(j => I[j]), "int32");

        const lossVal = opt.minimize(() => {
          const uEmb = model.userForward(u); // [B,D]
          const iEmb = model.itemForward(i); // [B,D]
          const logits = uEmb.matMul(iEmb.transpose()); // [B,B]
          const labels = tf.tensor1d([...Array(idx.length).keys()], "int32"); // 0..B-1
          const loss = tf.losses.softmaxCrossEntropy(tf.oneHot(labels, idx.length), logits);
          return loss;
        }, true);

        const l = (await lossVal.data())[0];
        lossVal.dispose(); u.dispose(); i.dispose();
        if (cfg.onBatchEnd) cfg.onBatchEnd(s/bs + ep*(Math.ceil(order.length/bs)), l);
        await tf.nextFrame();
      }
    }
  }

  async function scoreAllItems(model, userIdx){
    const u = tf.tensor1d([userIdx], "int32");
    const uEmb = model.userForward(u).squeeze(); // [D]
    const scores = tf.matMul(model.itemEmbedding, uEmb.expandDims(1)).squeeze(); // [numItems]
    const arr = Array.from(await scores.data());
    u.dispose(); uEmb.dispose(); scores.dispose();
    return arr;
  }

  // Fast PCA-ish projection using SVD (tfjs) on item embeddings
  async function projectItems2D(model){
    const E = model.itemEmbedding; // [N,D]
    const { u, s, v } = tf.linalg.svd(E, true); // u:[N,N] (thin), v:[D,D]
    const two = v.slice([0,0],[v.shape[0],2]); // [D,2]
    const proj = E.matMul(two); // [N,2]
    const data = await proj.array(); // [[x,y], ...]
    two.dispose(); proj.dispose(); u.dispose(); s.dispose(); v.dispose();
    return data;
  }

  return { TwoTowerModel, TwoTowerDeep, trainInBatchSoftmax, scoreAllItems, projectItems2D };
})();

// Re-export classes to global for app.js to use
const TwoTowerModel = TwoTower.TwoTowerModel;
const TwoTowerDeep  = TwoTower.TwoTowerDeep;
