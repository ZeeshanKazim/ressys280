/* retriever.js — Two-Tower (ID + tags) with in-batch softmax.
   Exposes:
     class TwoTowerRetriever
       constructor(nUsers, nItems, embDim, tagDim, opts)
       async compile(itemTagMat?)  // optional [nItems, tagDim] 0/1
       async trainStep(uIdx, iIdx) // tensors int32 shape [B]
       scoreUserAgainstAll(uIdxTensor) -> tf.Tensor1d [nItems]
       getItemEmbMatrix() -> tf.Tensor2d [nItems, embDim]
       dispose()
*/

class TwoTowerRetriever {
  constructor(nUsers, nItems, embDim, tagDim=0, opts={}) {
    this.nUsers = nUsers;
    this.nItems = nItems;
    this.embDim = embDim;
    this.tagDim = tagDim|0;
    this.lr = opts.learningRate || 1e-3;

    // ID embeddings (tf.Variable so .read() is valid)
    this.userEmb = tf.variable(tf.randomNormal([nUsers, embDim], 0, 0.05));
    this.itemIdEmb = tf.variable(tf.randomNormal([nItems, embDim], 0, 0.05));

    // Tag MLP if tagDim>0: K -> 64 -> emb
    if (this.tagDim > 0) {
      this.W1 = tf.variable(tf.randomNormal([this.tagDim, 64], 0, 0.05));
      this.b1 = tf.variable(tf.zeros([64]));
      this.W2 = tf.variable(tf.randomNormal([64, this.embDim], 0, 0.05));
      this.b2 = tf.variable(tf.zeros([this.embDim]));
    }

    this.optimizer = tf.train.adam(this.lr);
    this._itemTagMat = null;      // tf.Tensor2d one-hot/features (optional)
    this._cachedItemMatrix = null;// tf.Tensor2d for projection / full scoring
  }

  async compile(itemTagMat=null){
    if (this._cachedItemMatrix) { this._cachedItemMatrix.dispose(); this._cachedItemMatrix=null; }
    if (this._itemTagMat) { this._itemTagMat.dispose(); this._itemTagMat=null; }
    if (itemTagMat) this._itemTagMat = itemTagMat; // already a Tensor2D or null
    // nothing else required — optimizer set in ctor
  }

  // Produce final item embedding for a batch of indices [B]
  _itemEmbBatch(iIdx){
    return tf.tidy(() => {
      const idPart = tf.gather(this.itemIdEmb, iIdx); // [B, D]
      if (this.tagDim <= 0 || !this._itemTagMat) return idPart;
      const tags = tf.gather(this._itemTagMat, iIdx); // [B, K]
      const h = tf.relu(tags.matMul(this.W1).add(this.b1)); // [B, 64]
      const tagPart = h.matMul(this.W2).add(this.b2);       // [B, D]
      return idPart.add(tagPart); // fuse by addition
    });
  }

  // In-batch sampled softmax loss (BCE over B logits; label is diag)
  async trainStep(uIdx, iIdx){
    const loss = await this.optimizer.minimize(() => tf.tidy(() => {
      const U = tf.gather(this.userEmb, uIdx);        // [B, D]
      const Ipos = this._itemEmbBatch(iIdx);          // [B, D]
      const allI = tf.gather(this.itemIdEmb, iIdx);   // for negatives we’ll use batch items’ ID part
      // but use *final* item emb for negatives too for fidelity:
      const Iall = this._itemEmbBatch(iIdx);          // [B, D]
      const logits = tf.matMul(U, Iall, false, true); // [B, B]
      const labels = tf.eye(logits.shape[0]);         // one-hot diag
      const loss = tf.losses.softmaxCrossEntropy(labels, logits).mean();
      return loss;
    }), true);
    return loss.dataSync()[0];
  }

  // Precompute full item matrix for fast scoring / projection
  getItemEmbMatrix(){
    if (this._cachedItemMatrix) return this._cachedItemMatrix;
    const X = tf.tidy(() => {
      const idPart = this.itemIdEmb.read(); // tf.Tensor2d
      if (this.tagDim <= 0 || !this._itemTagMat) return idPart.clone();
      const h = tf.relu(this._itemTagMat.matMul(this.W1).add(this.b1)); // [N,K]->[N,64]
      const tagPart = h.matMul(this.W2).add(this.b2);                   // [N,emb]
      return idPart.add(tagPart);
    });
    this._cachedItemMatrix = X;
    return this._cachedItemMatrix;
  }

  scoreUserAgainstAll(uIdxTensor){
    // returns [nItems] tensor
    return tf.tidy(() => {
      const u = tf.gather(this.userEmb, uIdxTensor); // [1,D] or [B,D]
      const itemMat = this.getItemEmbMatrix();       // [N,D]
      const s = tf.matMul(u, itemMat, false, true);  // [B,N]
      return s.squeeze();                             // [N]
    });
  }

  dispose(){
    [this.userEmb, this.itemIdEmb, this.W1, this.b1, this.W2, this.b2].forEach(v=>v?.dispose?.());
    this._itemTagMat?.dispose?.();
    this._cachedItemMatrix?.dispose?.();
  }
}

window.TwoTowerRetriever = TwoTowerRetriever;
