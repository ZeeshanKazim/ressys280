/* two-tower.js
   Minimal Two-Tower model in TensorFlow.js
   - userEmbedding / itemEmbedding tables (tf.Variable)
   - optional single-hidden-layer MLP per tower (deep: true)
   - scoring by dot product
   - losses: in-batch softmax (default) or BPR pairwise
*/

class TwoTowerModel {
  /**
   * @param {number} numUsers
   * @param {number} numItems
   * @param {number} embDim
   * @param {{deep:boolean, hiddenDim:number, lossType:'softmax'|'bpr', learningRate:number}} opts
   */
  constructor(numUsers, numItems, embDim, opts={}){
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.embDim   = embDim;

    this.opts = Object.assign({
      deep: true,
      hiddenDim: 64,
      lossType: 'softmax', // or 'bpr'
      learningRate: 3e-3,
    }, opts);

    // --- embedding tables (do NOT reuse global variable names) ---
    const uInit = tf.randomNormal([numUsers, embDim], 0, 0.05);
    const iInit = tf.randomNormal([numItems, embDim], 0, 0.05);
    this.userEmbedding = tf.variable(uInit); // no "name" to avoid duplicate registration
    this.itemEmbedding = tf.variable(iInit);
    uInit.dispose(); iInit.dispose();

    // optional single hidden layer weights (for deep tower MLP)
    if (this.opts.deep && this.opts.hiddenDim > 0){
      // User tower: embDim -> hiddenDim -> embDim
      this.uW1 = tf.variable(tf.randomNormal([embDim, this.opts.hiddenDim], 0, 0.05));
      this.ub1 = tf.variable(tf.zeros([this.opts.hiddenDim]));
      this.uW2 = tf.variable(tf.randomNormal([this.opts.hiddenDim, embDim], 0, 0.05));
      this.ub2 = tf.variable(tf.zeros([embDim]));

      // Item tower
      this.iW1 = tf.variable(tf.randomNormal([embDim, this.opts.hiddenDim], 0, 0.05));
      this.ib1 = tf.variable(tf.zeros([this.opts.hiddenDim]));
      this.iW2 = tf.variable(tf.randomNormal([this.opts.hiddenDim, embDim], 0, 0.05));
      this.ib2 = tf.variable(tf.zeros([embDim]));
    } else {
      this.uW1=this.ub1=this.uW2=this.ub2=null;
      this.iW1=this.ib1=this.iW2=this.ib2=null;
    }

    this.optimizer = tf.train.adam(this.opts.learningRate);
  }

  // gather user embeddings -> [batch, embDim], then optional MLP
  userForward(userIdxTensor){
    return tf.tidy(()=> {
      let x = tf.gather(this.userEmbedding, userIdxTensor.flatten());
      if (this.uW1){
        x = x.matMul(this.uW1).add(this.ub1).relu();
        x = x.matMul(this.uW2).add(this.ub2);
      }
      return x; // [B, embDim]
    });
  }

  // gather item embeddings -> [batch, embDim], then optional MLP
  itemForward(itemIdxTensor){
    return tf.tidy(()=> {
      let y = tf.gather(this.itemEmbedding, itemIdxTensor.flatten());
      if (this.iW1){
        y = y.matMul(this.iW1).add(this.ib1).relu();
        y = y.matMul(this.iW2).add(this.ib2);
      }
      return y;
    });
  }

  // score by dot product between user and item vectors
  score(uEmb, iEmb){
    return tf.tidy(()=> tf.sum(tf.mul(uEmb, iEmb), -1)); // [B]
  }

  // one step of training; returns scalar loss (number)
  async trainStep(uIdx, iPosIdx){
    const uT = tf.tensor2d(uIdx, [uIdx.length,1], 'int32');
    const pT = tf.tensor2d(iPosIdx, [iPosIdx.length,1], 'int32');

    const lossFn = () => tf.tidy(()=>{
      const U = this.userForward(uT);   // [B, D]
      const I = this.itemForward(pT);   // [B, D]

      if (this.opts.lossType === 'bpr'){
        // sample negatives uniformly (in-batch could be used too)
        const negIdx = tf.randomUniform([iPosIdx.length], 0, this.numItems, 'int32').reshape([iPosIdx.length,1]);
        const INeg = this.itemForward(negIdx);
        const sPos = this.score(U, I);     // [B]
        const sNeg = this.score(U, INeg);  // [B]
        const l = tf.neg(tf.logSigmoid(tf.sub(sPos, sNeg))); // -log(σ(pos-neg))
        return tf.mean(l);
      } else {
        // In-batch sampled softmax:
        // logits = U @ I^T  (shape [B,B]); labels are diagonal
        const logits = tf.matMul(U, I, false, true); // [B,B]
        const labels = tf.eye(logits.shape[0]);      // [B,B] one-hot
        const loss = tf.losses.softmaxCrossEntropy(labels, logits);
        return loss.mean ? loss.mean() : tf.mean(loss);
      }
    });

    const lossVal = this.optimizer.minimize(lossFn, true, this.trainableVariables());
    const val = (await lossVal.data())[0];
    lossVal.dispose(); uT.dispose(); pT.dispose();
    return val;
  }

  trainableVariables(){
    const vars = [this.userEmbedding, this.itemEmbedding];
    if (this.uW1) vars.push(this.uW1, this.ub1, this.uW2, this.ub2);
    if (this.iW1) vars.push(this.iW1, this.ib1, this.iW2, this.ib2);
    return vars;
  }

  getAllItemEmbeddings(){
    // returns Float32Array length numItems*embDim of the *tower output* (after MLP if deep)
    const allIdx = tf.range(0, this.numItems, 1, 'int32').reshape([this.numItems,1]);
    const E = this.itemForward(allIdx); // [N,D]
    const flat = E.dataSync().slice(); // copy out
    E.dispose(); allIdx.dispose();
    return new Float32Array(flat);
  }

  dispose(){
    const vars = this.trainableVariables();
    vars.forEach(v => v.dispose());
  }
}

window.TwoTowerModel = TwoTowerModel;
