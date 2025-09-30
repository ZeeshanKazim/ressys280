// two-tower.js — Baseline (emb tables) and Deep (MLP + genres) two-tower models.

class TwoTowerBase {
  constructor(lossType='softmax'){ this.lossType=lossType; this.optimizer=tf.train.adam(0.003); }
  setOptimizer(opt){ this.optimizer=opt; }

  _inBatchSoftmax(U, Ipos){ // U:[B,D], I+:[B,D] -> logits [B,B]
    const logits = U.matMul(Ipos, false, true);
    const B = logits.shape[0];
    const labels = tf.oneHot(tf.range(0,B,1,'int32'), B);
    return tf.losses.softmaxCrossEntropy(labels, logits, {fromLogits:true});
  }
  _bpr(U, Ipos){ // in-batch negatives: random permute positives
    const B=U.shape[0]; const negIdx=tf.randomUniform([B],0,B,'int32'); const Ineg=tf.gather(Ipos, negIdx);
    const sPos=tf.sum(U.mul(Ipos),-1), sNeg=tf.sum(U.mul(Ineg),-1);
    return tf.softplus(sPos.sub(sNeg).neg()).mean();
  }
}

class TwoTowerModel extends TwoTowerBase {
  // Baseline: only ID embeddings (no hidden).
  constructor(numUsers, numItems, embDim=32, lossType='softmax'){
    super(lossType);
    this.numUsers=numUsers; this.numItems=numItems; this.embDim=embDim;
    this.userEmbedding=tf.variable(tf.randomNormal([numUsers,embDim],0,0.05), true, 'userEmb');
    this.itemEmbedding=tf.variable(tf.randomNormal([numItems,embDim],0,0.05), true, 'itemEmb');
  }
  userForward(userIdx){ return tf.gather(this.userEmbedding, userIdx); }
  itemForward(itemIdx){ return tf.gather(this.itemEmbedding, itemIdx); }
  async trainStep(uIdxArr,iIdxArr){
    const u=tf.tensor1d(uIdxArr,'int32'), it=tf.tensor1d(iIdxArr,'int32');
    const lossT = await this.optimizer.minimize(()=> {
      const U=this.userForward(u), I=this.itemForward(it);
      return (this.lossType==='bpr')? this._bpr(U,I) : this._inBatchSoftmax(U,I);
    }, true);
    const L=(await lossT.data())[0]; u.dispose(); it.dispose(); lossT.dispose(); return L;
  }
  getUserEmbeddingForIndex(uIdx){ return tf.tidy(()=> tf.gather(this.userEmbedding, tf.tensor1d([uIdx],'int32'))); } // [1,D]
  getItemEmbedding(){ return this.itemEmbedding; }
}

class TwoTowerDeepModel extends TwoTowerBase {
  // Deep towers: 1 hidden layer per side, genres on item side.
  constructor(numUsers, numItems, embDim=32, genreDim=18, hiddenDim=64, lossType='softmax', itemGenreTensor){
    super(lossType);
    this.numUsers=numUsers; this.numItems=numItems; this.embDim=embDim; this.hiddenDim=hiddenDim; this.genreDim=genreDim;
    this.itemGenres = itemGenreTensor || tf.zeros([numItems, genreDim]);

    this.userEmbedding=tf.variable(tf.randomNormal([numUsers,embDim],0,0.05), true, 'userEmbDeep');
    this.itemEmbedding=tf.variable(tf.randomNormal([numItems,embDim],0,0.05), true, 'itemEmbDeep');

    this.Wg = tf.variable(tf.randomNormal([genreDim, embDim], 0, 0.05), true, 'Wg');
    this.bg = tf.variable(tf.zeros([embDim]), true, 'bg');

    this.Wu1=tf.variable(tf.randomNormal([embDim,hiddenDim],0,0.05), true,'Wu1'); this.bu1=tf.variable(tf.zeros([hiddenDim]), true,'bu1');
    this.Wu2=tf.variable(tf.randomNormal([hiddenDim,embDim],0,0.05), true,'Wu2'); this.bu2=tf.variable(tf.zeros([embDim]), true,'bu2');

    this.Wi1=tf.variable(tf.randomNormal([embDim*2,hiddenDim],0,0.05), true,'Wi1'); this.bi1=tf.variable(tf.zeros([hiddenDim]), true,'bi1');
    this.Wi2=tf.variable(tf.randomNormal([hiddenDim,embDim],0,0.05), true,'Wi2'); this.bi2=tf.variable(tf.zeros([embDim]), true,'bi2');
  }

  userForward(userIdx){
    return tf.tidy(()=>{
      const idEmb=tf.gather(this.userEmbedding,userIdx);      // [B,D]
      const h=idEmb.matMul(this.Wu1).add(this.bu1).relu();    // [B,H]
      return h.matMul(this.Wu2).add(this.bu2);                // [B,D]
    });
  }
  itemForward(itemIdx){
    return tf.tidy(()=>{
      const idEmb=tf.gather(this.itemEmbedding,itemIdx);      // [B,D]
      const G=tf.gather(this.itemGenres,itemIdx);             // [B,G]
      const gEmb=G.matMul(this.Wg).add(this.bg);              // [B,D]
      const x=tf.concat([idEmb,gEmb],-1);                     // [B,2D]
      const h=x.matMul(this.Wi1).add(this.bi1).relu();        // [B,H]
      return h.matMul(this.Wi2).add(this.bi2);                // [B,D]
    });
  }
  async trainStep(uIdxArr,iIdxArr){
    const u=tf.tensor1d(uIdxArr,'int32'), it=tf.tensor1d(iIdxArr,'int32');
    const lossT = await this.optimizer.minimize(()=> {
      const U=this.userForward(u), I=this.itemForward(it);
      return (this.lossType==='bpr')? this._bpr(U,I) : this._inBatchSoftmax(U,I);
    }, true);
    const L=(await lossT.data())[0]; u.dispose(); it.dispose(); lossT.dispose(); return L;
  }
  getUserEmbeddingForIndex(uIdx){ return this.userForward(tf.tensor1d([uIdx],'int32')); } // [1,D]
}
