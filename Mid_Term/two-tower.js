/* two-tower.js — Baseline & Deep Two‑Tower for TF.js
   - Baseline: id embeddings + dot (no hidden layers)
   - Deep: user & item towers with 1 hidden layer; item tower can use tag features (hashed one-hot)
   - Loss: in-batch softmax (default) or BPR
*/

class TwoTowerModel {
  constructor(numUsers, numItems, embDim, opts={}){
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.embDim   = embDim;
    this.opts = Object.assign({
      deep:true, hiddenDim:64, lossType:'softmax', learningRate:3e-3, useTags:false, tagDim:0
    }, opts);

    // Embedding tables (no explicit names => no duplicate registration issues)
    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05));
    this.itemEmbedding = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05));

    // Tag projection (non-trainable input tensor will be set from app.js via setItemTags)
    this.itemTags = null;  // tf.Tensor2D [numItems, tagDim]
    this.tagW = null;      // trainable projection if useTags
    if(this.opts.useTags && this.opts.tagDim>0){
      this.tagW = tf.variable(tf.randomNormal([this.opts.tagDim, embDim], 0, 0.05));
    }

    // Deep MLP params (1 hidden) if deep
    if(this.opts.deep && this.opts.hiddenDim>0){
      // User tower: embDim -> hidden -> embDim
      this.uW1 = tf.variable(tf.randomNormal([embDim, this.opts.hiddenDim], 0, 0.05));
      this.ub1 = tf.variable(tf.zeros([this.opts.hiddenDim]));
      this.uW2 = tf.variable(tf.randomNormal([this.opts.hiddenDim, embDim], 0, 0.05));
      this.ub2 = tf.variable(tf.zeros([embDim]));

      // Item tower input size: embDim (+ embDim from tagW if useTags? we concatenate after projecting tags to embDim)
      const inItem = this.opts.useTags ? (embDim + embDim) : embDim;
      this.iW1 = tf.variable(tf.randomNormal([inItem, this.opts.hiddenDim], 0, 0.05));
      this.ib1 = tf.variable(tf.zeros([this.opts.hiddenDim]));
      this.iW2 = tf.variable(tf.randomNormal([this.opts.hiddenDim, embDim], 0, 0.05));
      this.ib2 = tf.variable(tf.zeros([embDim]));
    } else {
      this.uW1=this.ub1=this.uW2=this.ub2=null;
      this.iW1=this.ib1=this.iW2=this.ib2=null;
    }

    this.optimizer = tf.train.adam(this.opts.learningRate);
  }

  setItemTags(tensor2d){ // [numItems, tagDim], non-trainable
    if(this.itemTags) this.itemTags.dispose();
    this.itemTags = tensor2d;
  }

  userForward(userIdxTensor){
    return tf.tidy(()=>{
      let x = tf.gather(this.userEmbedding, userIdxTensor.flatten()); // [B,emb]
      if(this.uW1){
        x = x.matMul(this.uW1).add(this.ub1).relu();
        x = x.matMul(this.uW2).add(this.ub2);
      }
      // L2 normalize
      return tf.linalg.l2Normalize(x, -1);
    });
  }

  itemForward(itemIdxTensor){
    return tf.tidy(()=>{
      let idEmb = tf.gather(this.itemEmbedding, itemIdxTensor.flatten()); // [B,emb]
      if(this.opts.useTags && this.itemTags && this.tagW){
        const tagBatch = tf.gather(this.itemTags, itemIdxTensor.flatten()); // [B,tagDim]
        const tagProj  = tagBatch.matMul(this.tagW); // [B,emb]
        idEmb = tf.concat([idEmb, tagProj], -1);     // [B, 2*emb]
      }
      if(this.iW1){
        idEmb = idEmb.matMul(this.iW1).add(this.ib1).relu();
        idEmb = idEmb.matMul(this.iW2).add(this.ib2);
      }
      return tf.linalg.l2Normalize(idEmb, -1);
    });
  }

  score(uEmb, iEmb){
    return tf.sum(tf.mul(uEmb, iEmb), -1); // [B]
  }

  trainableVariables(){
    const vars = [this.userEmbedding, this.itemEmbedding];
    if(this.tagW) vars.push(this.tagW);
    if(this.uW1) vars.push(this.uW1, this.ub1, this.uW2, this.ub2);
    if(this.iW1) vars.push(this.iW1, this.ib1, this.iW2, this.ib2);
    return vars;
  }

  async trainStep(uIdxArr, iIdxArr){
    const uT = tf.tensor2d(uIdxArr, [uIdxArr.length,1], 'int32');
    const iT = tf.tensor2d(iIdxArr, [iIdxArr.length,1], 'int32');

    const lossFn = () => tf.tidy(()=>{
      const U = this.userForward(uT); // [B,emb]
      const I = this.itemForward(iT); // [B,emb]

      if(this.opts.lossType === 'bpr'){
        // sample negatives uniformly
        const negIdx = tf.randomUniform([iIdxArr.length], 0, this.numItems, 'int32').reshape([iIdxArr.length,1]);
        const INeg = this.itemForward(negIdx);
        const sPos = this.score(U,I);
        const sNeg = this.score(U,INeg);
        const l = tf.neg(tf.logSigmoid(tf.sub(sPos, sNeg)));
        return tf.mean(l);
      } else {
        // In-batch softmax: logits = U @ I^T, labels=eye
        const logits = tf.matMul(U, I, false, true); // [B,B]
        const labels = tf.eye(logits.shape[0]);
        const loss   = tf.losses.softmaxCrossEntropy(labels, logits);
        return loss.mean ? loss.mean() : tf.mean(loss);
      }
    });

    const l = this.optimizer.minimize(lossFn, true, this.trainableVariables());
    const val = (await l.data())[0];
    l.dispose(); uT.dispose(); iT.dispose();
    return val;
  }

  getAllItemEmbeddings(){
    // returns Float32Array of final tower output for each item
    const idx = tf.range(0, this.numItems, 1, 'int32').reshape([this.numItems,1]);
    const E = this.itemForward(idx); // [N,emb]
    const flat = E.dataSync().slice();
    E.dispose(); idx.dispose();
    return new Float32Array(flat);
  }

  dispose(){
    for(const v of this.trainableVariables()) v.dispose();
    if(this.itemTags) this.itemTags.dispose();
  }
}

window.TwoTowerModel = TwoTowerModel;
