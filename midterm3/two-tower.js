/* two-tower.js — content-aware Two-Tower with BPR loss (no named vars → no collisions) */
class TwoTowerModel {
  constructor(numUsers, numItems, embDim, tagFeatDim=0, opts={}) {
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.embDim = embDim;
    this.tagFeatDim = tagFeatDim|0;
    this.lr = opts.learningRate ?? 1e-3;

    // ID embeddings
    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05));
    this.itemEmbedding = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05));

    // Optional item tag projection (one-layer MLP to same emb space)
    if (this.tagFeatDim > 0) {
      this.Wt = tf.variable(tf.randomNormal([this.tagFeatDim, embDim], 0, 0.05));
      this.bt = tf.variable(tf.zeros([embDim]));
    }

    this.optimizer = tf.train.adam(this.lr);
  }

  dispose(){
    Object.values(this).forEach(v=>v?.dispose?.());
  }

  compile(){ /* no-op kept for symmetry */ return Promise.resolve(); }

  /** itemTagMat: [numItems, tagFeatDim] or null */
  setItemTagMatrix(itemTagMat){
    this.itemTagMat = itemTagMat || null;
  }

  /** Return [B, D] user emb and [B, D] item emb; mix in tags if provided */
  _forward(uIdx, iIdx){
    return tf.tidy(()=>{
      const ue = tf.gather(this.userEmbedding, uIdx); // [B,D]
      let ie = tf.gather(this.itemEmbedding, iIdx);   // [B,D]
      if (this.itemTagMat && this.Wt){
        const tag = tf.gather(this.itemTagMat, iIdx); // [B,K]
        const tagProj = tag.matMul(this.Wt).add(this.bt).tanh(); // [B,D]
        ie = ie.add(tagProj).mul(0.5);
      }
      return {ue, ie};
    });
  }

  /** One BPR step with 1 negative per positive */
  async trainStep(uIdx, iIdx){
    const batch = uIdx.shape[0];
    const lossVal = await this.optimizer.minimize(()=>{
      const nIdx = tf.randomUniform([batch], 0, this.numItems, 'int32');
      const {ue, ie} = this._forward(uIdx, iIdx);
      const ineg = tf.gather(this.itemEmbedding, nIdx);
      const sPos = tf.sum(ue.mul(ie), -1);   // [B]
      const sNeg = tf.sum(ue.mul(ineg), -1); // [B]
      const loss = tf.neg(tf.mean(tf.logSigmoid(sPos.sub(sNeg)))); // BPR
      return loss;
    }, true);

    return lossVal.dataSync()[0];
  }

  /** scores for all items for a single user */
  async scoreUserAgainstAll(uIdx1){
    return tf.tidy(()=>{
      const ue = tf.gather(this.userEmbedding, uIdx1); // [1,D]
      const scores = this.itemEmbedding.matMul(ue.transpose()).squeeze(); // [N]
      return scores;
    });
  }

  /** score particular items for a single user */
  async scoreItems(uIdx1, itemIdxArr){
    return tf.tidy(()=>{
      const ue = tf.gather(this.userEmbedding, uIdx1); // [1,D]
      const I = tf.tensor1d(itemIdxArr, 'int32');
      const ie = tf.gather(this.itemEmbedding, I);
      const s = ie.matMul(ue.transpose()).squeeze(); // [len]
      I.dispose();
      return s;
    });
  }
}
