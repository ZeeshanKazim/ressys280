/* two-tower.js — Minimal Two-Tower Retrieval in TensorFlow.js
   Modes:
   - "baseline": user/item ID embeddings only
   - "deep":     user ID embedding + item ID embedding + item tag multi-hot -> MLP
   Loss: in-batch softmax (sampled softmax with in-batch negatives)
*/

class TwoTowerModel {
  constructor({mode='baseline', numUsers, numItems, embDim=32, lr=1e-3, itemFeatDim=0}) {
    this.mode = mode;
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.embDim = embDim;
    this.itemFeatDim = itemFeatDim|0;
    this.optimizer = tf.train.adam(lr);

    // ID embedding tables
    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05), true, 'userEmbedding_'+Math.random());
    this.itemEmbedding = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05), true, 'itemEmbedding_'+Math.random());

    // Deep extras (for items)
    if (this.mode === 'deep' && this.itemFeatDim > 0){
      // Project multi-hot tags -> dense vector, then combine with ID embedding
      this.itemFeatW1 = tf.variable(tf.randomNormal([this.itemFeatDim, Math.max(embDim, 32)], 0, 0.05), true, 'itemFeatW1_'+Math.random());
      this.itemFeatB1 = tf.variable(tf.zeros([Math.max(embDim, 32)]), true, 'itemFeatB1_'+Math.random());
      this.itemFeatW2 = tf.variable(tf.randomNormal([Math.max(embDim, 32), embDim], 0, 0.05), true, 'itemFeatW2_'+Math.random());
      this.itemFeatB2 = tf.variable(tf.zeros([embDim]), true, 'itemFeatB2_'+Math.random());
      this._itemFeatMatrix = null; // set via setItemFeatureMatrix(Float32Array len=numItems*featDim)
    }
  }

  dispose(){
    Object.values(this).forEach(v=> { if (v && typeof v.dispose==='function') v.dispose(); });
  }

  setItemFeatureMatrix(floatArray){
    if (this.mode!=='deep' || this.itemFeatDim===0) return;
    if (this._itemFeatMatrix) this._itemFeatMatrix.dispose();
    this._itemFeatMatrix = tf.tensor2d(floatArray, [this.numItems, this.itemFeatDim], 'float32');
  }

  /* ========== Towers ========== */
  userForward(userIdxTensor){ // [B]
    return tf.gather(this.userEmbedding, userIdxTensor); // [B, D]
  }

  itemForward(itemIdxTensor){ // [B]
    const idEmb = tf.gather(this.itemEmbedding, itemIdxTensor); // [B, D]
    if (this.mode==='deep' && this._itemFeatMatrix){
      const feats = tf.gather(this._itemFeatMatrix, itemIdxTensor); // [B, F]
      const h1 = tf.relu(tf.add(tf.matMul(feats, this.itemFeatW1), this.itemFeatB1));
      const proj = tf.add(tf.matMul(h1, this.itemFeatW2), this.itemFeatB2); // [B, D]
      return tf.add(idEmb, proj).div(tf.scalar(2.0)); // simple fusion
    }
    return idEmb;
  }

  /* Score = dot(u, v) */
  score(uEmb, iEmb){ // [B,D] x [B,D]
    // We will compute logits = U @ I^T, so return full matrix
    return tf.matMul(uEmb, iEmb, false, true); // [B,B]
  }

  /* ========== Training ========== */
  async trainStep(uIdxArr, iIdxArr){
    const uIdx = tf.tensor1d(uIdxArr, 'int32');
    const iIdx = tf.tensor1d(iIdxArr, 'int32');

    const lossVal = await this.optimizer.minimize(()=>{
      const U = this.userForward(uIdx);  // [B,D]
      const I = this.itemForward(iIdx);  // [B,D]
      const logits = this.score(U, I);   // [B,B]

      // labels are diagonal -> one-hot for each row
      const B = logits.shape[0];
      const labels = tf.oneHot(tf.tensor1d([...Array(B).keys()], 'int32'), B); // [B,B]
      // softmax cross entropy over columns
      const loss = tf.mean(tf.losses.softmaxCrossEntropy(labels, logits));
      return loss;
    }, true);

    uIdx.dispose(); iIdx.dispose();
    const v = lossVal.dataSync()[0];
    lossVal.dispose();
    return v;
  }

  /* ========== Inference utilities ========== */
  getAllItemEmbeddings(){ // returns tensor [numItems, D]
    const idEmb = this.itemEmbedding;
    if (this.mode!=='deep' || !this._itemFeatMatrix) return idEmb;
    const feats = this._itemFeatMatrix;
    const h1 = tf.relu(tf.add(tf.matMul(feats, this.itemFeatW1), this.itemFeatB1));
    const proj = tf.add(tf.matMul(h1, this.itemFeatW2), this.itemFeatB2);
    return tf.add(idEmb, proj).div(tf.scalar(2.0));
  }

  async scoreUserAgainstItems(uIdx, itemIdxList){
    const u = tf.tensor1d([uIdx], 'int32');
    const U = this.userForward(u); // [1,D]
    const itemsIdx = tf.tensor1d(itemIdxList, 'int32');
    const I = this.itemForward(itemsIdx); // [K,D]
    const scoresT = tf.matMul(U, I, false, true); // [1,K]
    const scores = (await scoresT.data()).slice(); // JS array
    u.dispose(); U.dispose(); itemsIdx.dispose(); I.dispose(); scoresT.dispose();
    return scores;
  }
}

window.TwoTowerModel = TwoTowerModel;
