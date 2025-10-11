/* two-tower.js – minimal Two-Tower models with TF.js
   Baseline: userId -> emb, itemId -> emb, score = dot(u,i).
   Deep: userId -> emb, item tags (multi-hot) -> MLP -> item emb, score = dot(u,f(tags)).
   Loss: in-batch sampled softmax (cross-entropy on the diagonal).
*/

class TwoTowerModel {
  constructor(numUsers, numItems, embDim=32, opts={}){
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.embDim = embDim;
    const std = 0.05;

    // Variables (no explicit names → no duplicate-name crash on re-train)
    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, std));
    this.itemEmbedding = tf.variable(tf.randomNormal([numItems, embDim], 0, std));

    this.optimizer = tf.train.adam(opts.learningRate ?? 1e-3);
  }
  async compile(){ /* no-op; optimizer is already set */ }

  // Gather embeddings from integer index tensors
  gatherUsers(uIdx){ return tf.gather(this.userEmbedding, uIdx); }    // [B,D]
  gatherItems(iIdx){ return tf.gather(this.itemEmbedding, iIdx); }    // [B,D]
  score(uEmb, iEmb){ return tf.matMul(uEmb, iEmb, false, true); }     // [B,B] logits

  trainStep(uIdx, iIdx){
    return tf.tidy(()=>{
      const lossFn = () => {
        const U = this.gatherUsers(uIdx);     // [B,D]
        const I = this.gatherItems(iIdx);     // [B,D]
        const logits = this.score(U, I);      // [B,B]
        const labels = tf.oneHot(tf.range(0, logits.shape[0], 1, 'int32'), logits.shape[1]);
        const loss = tf.losses.softmaxCrossEntropy(labels, logits).mean();
        return loss;
      };
      const l = this.optimizer.minimize(lossFn, true);
      return l.dataSync()[0];
    });
  }

  async scoreUserAgainstAll(uIdx){
    return tf.tidy(()=>{
      const u = this.gatherUsers(uIdx);           // [1,D]
      const scores = tf.matMul(u, this.itemEmbedding, false, true);  // [1,N]
      return scores.squeeze(); // [N]
    });
  }

  async scoreItems(uIdx, itemIndices){ // for metrics sampling
    return tf.tidy(()=>{
      const u = this.gatherUsers(uIdx);                // [1,D]
      const I = tf.gather(this.itemEmbedding, tf.tensor1d(itemIndices,'int32')); // [M,D]
      const s = tf.matMul(u, I, false, true); // [1,M]
      return s.squeeze(); // [M]
    });
  }

  dispose(){
    this.userEmbedding.dispose();
    this.itemEmbedding.dispose();
    this.optimizer.dispose?.();
  }
}

class DeepTwoTowerModel {
  constructor(numUsers, numItems, embDim=32, tagDim=200, opts={}){
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.embDim = embDim;
    this.tagDim = tagDim;
    this.optimizer = tf.train.adam(opts.learningRate ?? 1e-3);

    // user ID embedding
    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05));

    // item side: small MLP that maps tag multi-hot -> embDim
    // W1: [tagDim, 128] → ReLU → W2: [128, embDim]
    const hidden = Math.max(32, Math.min(128, tagDim));
    this.W1 = tf.variable(tf.randomNormal([tagDim, hidden], 0, 0.05));
    this.b1 = tf.variable(tf.zeros([hidden]));
    this.W2 = tf.variable(tf.randomNormal([hidden, embDim], 0, 0.05));
    this.b2 = tf.variable(tf.zeros([embDim]));

    // A cached itemTag matrix may be set via compile()
    this.itemTagMat = null; // [numItems, tagDim] float32
  }

  async compile(itemTagMat){
    // itemTagMat is required (but we also support zero-matrix fallback)
    this.itemTagMat = itemTagMat || tf.zeros([this.numItems, this.tagDim], 'float32');
  }

  itemForwardFromTags(tags){ // tags: [B,tagDim]
    const h = tf.relu(tags.matMul(this.W1).add(this.b1));
    const z = h.matMul(this.W2).add(this.b2);
    return z; // [B,embDim]
  }
  gatherUsers(uIdx){ return tf.gather(this.userEmbedding, uIdx); } // [B,D]

  trainStep(uIdx, iIdx){
    return tf.tidy(()=>{
      const lossFn = () => {
        const U = this.gatherUsers(uIdx);                                   // [B,D]
        const tagB = tf.gather(this.itemTagMat, iIdx);                      // [B,tagDim]
        const I = this.itemForwardFromTags(tagB);                           // [B,D]
        const logits = tf.matMul(U, I, false, true);                        // [B,B]
        const labels = tf.oneHot(tf.range(0, logits.shape[0], 1, 'int32'), logits.shape[1]);
        const loss = tf.losses.softmaxCrossEntropy(labels, logits).mean();
        return loss;
      };
      const l = this.optimizer.minimize(lossFn, true);
      return l.dataSync()[0];
    });
  }

  // Scores for all items (passes full itemTagMat through MLP once per call)
  async scoreUserAgainstAll(uIdx){
    return tf.tidy(()=>{
      const u = this.gatherUsers(uIdx);                     // [1,D]
      const I = this.itemForwardFromTags(this.itemTagMat);  // [N,D]
      const s = tf.matMul(u, I, false, true);               // [1,N]
      return s.squeeze();
    });
  }

  async scoreItems(uIdx, itemIndices){
    return tf.tidy(()=>{
      const u = this.gatherUsers(uIdx);                                       // [1,D]
      const tags = tf.gather(this.itemTagMat, tf.tensor1d(itemIndices,'int32'));// [M,K]
      const I = this.itemForwardFromTags(tags);                               // [M,D]
      const s = tf.matMul(u, I, false, true);                                 // [1,M]
      return s.squeeze();                                                     // [M]
    });
  }

  // for projection we want the current item representations
  getFrozenItemEmb(){
    return tf.tidy(()=> this.itemForwardFromTags(this.itemTagMat).clone() );
  }

  dispose(){
    this.userEmbedding.dispose();
    this.W1.dispose(); this.b1.dispose(); this.W2.dispose(); this.b2.dispose();
    this.itemTagMat?.dispose();
    this.optimizer.dispose?.();
  }
}

window.TwoTowerModel = TwoTowerModel;
window.DeepTwoTowerModel = DeepTwoTowerModel;
