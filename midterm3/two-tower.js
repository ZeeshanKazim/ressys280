/* two-tower.js — minimal content-aware Two-Tower with in-batch negatives */
(function(global){
  class TwoTowerModel{
    constructor(numUsers, numItems, embDim=32, opts={}){
      this.numUsers = numUsers;
      this.numItems = numItems;
      this.embDim = embDim;
      this.opt = tf.train.adam(opts.learningRate ?? 1e-3);
      // Embeddings as Variables so we can .read() or .dataSync() safely
      this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05), true, 'userEmbedding');
      this.itemEmbedding = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05), true, 'itemEmbedding');
    }
    async compile(){ /* no-op for symmetry */ }

    userForward(uIdx){ return tf.gather(this.userEmbedding, uIdx); }
    itemForward(iIdx){ return tf.gather(this.itemEmbedding, iIdx); }

    // In-batch negatives: logits = U * I^T, labels are diagonal
    async trainStep(uIdx, iIdx){
      const loss = this.opt.minimize(()=>{
        const U = this.userForward(uIdx);     // [B,D]
        const I = this.itemForward(iIdx);     // [B,D]
        const logits = tf.matMul(U, I, false, true);  // [B,B]
        const labels = tf.tensor1d([...Array(uIdx.shape[0]).keys()], 'int32'); // 0..B-1
        const ce = tf.losses.softmaxCrossEntropy(tf.oneHot(labels, uIdx.shape[0]), logits);
        return ce;
      }, true);
      const v = (await loss.data())[0];
      loss.dispose();
      return v;
    }

    scoreUserAgainstAll(uIdx){
      const U = this.userForward(uIdx);           // [1,D] or [B,D]
      const scores = tf.matMul(U, this.itemEmbedding, false, true); // [B,N]
      return scores.squeeze(); // [N] if B==1
    }

    scoreItems(uIdx, itemIdxArr){
      const U = this.userForward(uIdx);                  // [1,D]
      const I = this.itemForward(tf.tensor1d(itemIdxArr,'int32')); // [K,D]
      return tf.matMul(U, I, false, true).squeeze();     // [K]
    }

    getItemEmbeddingTensor(){ return this.itemEmbedding.clone(); }
    dispose(){
      this.userEmbedding.dispose();
      this.itemEmbedding.dispose();
    }
  }

  global.TwoTowerModel = TwoTowerModel;
})(window);
