// two-tower.js
// Baseline Two-Tower (embedding tables only) + Deep Two-Tower (MLP towers + genre features).
// Loss: in-batch sampled softmax (default) or BPR pairwise.

class TwoTowerBase {
  constructor(lossType='softmax'){ this.lossType = lossType; this.optimizer = tf.train.adam(0.003); }
  setOptimizer(opt){ this.optimizer = opt; }

  // ----- Losses -----
  _inBatchSoftmax(uEmb, iEmbPos){ // U:[B,D], I+:[B,D]
    const logits = uEmb.matMul(iEmbPos, false, true); // [B,B]
    const B = logits.shape[0];
    const labels = tf.oneHot(tf.range(0,B,1,'int32'), B);
    return tf.losses.softmaxCrossEntropy(labels, logits, {fromLogits:true});
  }
  _bpr(uEmb, iEmbPos){ // in-batch negatives by random permutation
    const B=uEmb.shape[0]; const negIdx=tf.randomUniform([B],0,B,'int32');
    const iEmbNeg=tf.gather(iEmbPos, negIdx);
    const sPos=tf.sum(uEmb.mul(iEmbPos),-1); // [B]
    const sNeg=tf.sum(uEmb.mul(iEmbNeg),-1); // [B]
    return tf.softplus(sPos.sub(sNeg).neg()).mean();
  }
}

class TwoTowerModel extends TwoTowerBase {
  /**
   * Baseline: user/item embeddings only (no hidden layers).
   */
  constructor(numUsers, numItems, embDim=32, lossType='softmax'){
    super(lossType);
    this.numUsers=numUsers; this.numItems=numItems; this.embDim=embDim;
    this.userEmbedding=tf.variable(tf.randomNormal([numUsers,embDim],0,0.05), true, 'userEmb');
    this.itemEmbedding=tf.variable(tf.randomNormal([numItems,embDim],0,0.05), true, 'itemEmb');
  }
  userForward(userIdx){ return tf.gather(this.userEmbedding, userIdx); }  // [B,D]
  itemForward(itemIdx){ return tf.gather(this.itemEmbedding, itemIdx); }  // [B,D]

  async trainStep(uIdxArr, iIdxArr){
    const uIdx=tf.tensor1d(uIdxArr,'int32'); const iIdx=tf.tensor1d(iIdxArr,'int32');
    const lossT = await this.optimizer.minimize(()=>{
      const U=this.userForward(uIdx), I=this.itemForward(iIdx);
      return (this.lossType==='bpr') ? this._bpr(U,I) : this._inBatchSoftmax(U,I);
    }, true);
    const loss=(await lossT.data())[0]; uIdx.dispose(); iIdx.dispose(); lossT.dispose(); return loss;
  }
  getUserEmbeddingForIndex(uIdx){ return tf.tidy(()=> tf.gather(this.userEmbedding, tf.tensor1d([uIdx],'int32'))); } // [1,D]
  getItemEmbedding(){ return this.itemEmbedding; } // [N,D] (variable ref)
}

class TwoTowerDeepModel extends TwoTowerBase {
  /**
   * Deep two-tower with 1 hidden layer per tower and genre features on item side.
   * user:  id-embedding -> Dense(hid) -> ReLU -> Dense(embDim)
   * item: [id-embedding, genre-projection] concat -> Dense(hid) -> ReLU -> Dense(embDim)
   */
  constructor(numUsers, numItems, embDim=32, genreDim=18, hiddenDim=64, lossType='softmax', itemGenreTensor/*[N,G]*/){
    super(lossType);
    this.numUsers=numUsers; this.numItems=numItems; this.embDim=embDim; this.hiddenDim=hiddenDim; this.genreDim=genreDim;
    this.itemGenres = itemGenreTensor || tf.zeros([numItems, genreDim]); // constant features

    // ID embeddings
    this.userEmbedding=tf.variable(tf.randomNormal([numUsers,embDim],0,0.05), true, 'userEmbDeep');
    this.itemEmbedding=tf.variable(tf.randomNormal([numItems,embDim],0,0.05), true, 'itemEmbDeep');

    // Genre linear map: G -> embDim
    this.Wg = tf.variable(tf.randomNormal([genreDim, embDim], 0, 0.05), true, 'Wg');
    this.bg = tf.variable(tf.zeros([embDim]), true, 'bg');

    // User tower weights
    this.Wu1 = tf.variable(tf.randomNormal([embDim, hiddenDim], 0, 0.05), true, 'Wu1');
    this.bu1 = tf.variable(tf.zeros([hiddenDim]), true, 'bu1');
    this.Wu2 = tf.variable(tf.randomNormal([hiddenDim, embDim], 0, 0.05), true, 'Wu2');
    this.bu2 = tf.variable(tf.zeros([embDim]), true, 'bu2');

    // Item tower weights
    this.Wi1 = tf.variable(tf.randomNormal([embDim*2, hiddenDim], 0, 0.05), true, 'Wi1'); // concat(idEmb, genreEmb)
    this.bi1 = tf.variable(tf.zeros([hiddenDim]), true, 'bi1');
    this.Wi2 = tf.variable(tf.randomNormal([hiddenDim, embDim], 0, 0.05), true, 'Wi2');
    this.bi2 = tf.variable(tf.zeros([embDim]), true, 'bi2');

    this.optimizer = tf.train.adam(0.003);
  }

  userForward(userIdx){ // [B] -> [B,embDim]
    return tf.tidy(()=>{
      const idEmb = tf.gather(this.userEmbedding, userIdx);        // [B,D]
      const h = idEmb.matMul(this.Wu1).add(this.bu1).relu();       // [B,H]
      return h.matMul(this.Wu2).add(this.bu2);                     // [B,D]
    });
  }

  itemForward(itemIdx){ // [B] -> [B,embDim]
    return tf.tidy(()=>{
      const idEmb = tf.gather(this.itemEmbedding, itemIdx);        // [B,D]
      const G = tf.gather(this.itemGenres, itemIdx);               // [B,genreDim]
      const gEmb = G.matMul(this.Wg).add(this.bg);                 // [B,D]
      const x = tf.concat([idEmb, gEmb], -1);                      // [B,2D]
      const h = x.matMul(this.Wi1).add(this.bi1).relu();           // [B,H]
      return h.matMul(this.Wi2).add(this.bi2);                     // [B,D]
    });
  }

  async trainStep(uIdxArr, iIdxArr){
    const uIdx=tf.tensor1d(uIdxArr,'int32'); const iIdx=tf.tensor1d(iIdxArr,'int32');
    const lossT = await this.optimizer.minimize(()=>{
      const U=this.userForward(uIdx), I=this.itemForward(iIdx);
      return (this.lossType==='bpr') ? this._bpr(U,I) : this._inBatchSoftmax(U,I);
    }, true);
    const loss=(await lossT.data())[0]; uIdx.dispose(); iIdx.dispose(); lossT.dispose(); return loss;
  }

  getUserEmbeddingForIndex(uIdx){ return this.userForward(tf.tensor1d([uIdx],'int32')); } // [1,D]
}

