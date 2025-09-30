// two-tower.js
// Minimal two-tower models in TF.js with in-batch softmax (default) or BPR loss.

class TwoTowerBase {
  constructor(numUsers, numItems, embDim, lossType='softmax'){
    this.numUsers=numUsers; this.numItems=numItems; this.embDim=embDim;
    this.lossType=lossType;
  }
  setOptimizer(opt){ this.opt=opt; }
  // --- softmax with in-batch negatives: mean(logsumexp(row) - diag(row)) ---
  static softmaxInBatchLoss(logits){
    // logits shape [B, B]
    const B = logits.shape[0];
    const max = logits.max(1, true);
    const stab = logits.sub(max);
    const lse = tf.log(tf.exp(stab).sum(1));                  // [B]
    // gather diagonal
    const idx = tf.tensor2d([...Array(B).keys()].map(i=>[i,i]), [B,2], 'int32');
    const diag = stab.gatherND(idx);                          // [B]
    const loss = lse.sub(diag).mean();                        // scalar
    idx.dispose(); max.dispose(); stab.dispose(); return loss;
  }
  // --- BPR loss: −mean(log σ(pos − neg)) ---
  static bprLoss(pos, neg){
    // pos, neg shape [B]
    const x = pos.sub(neg);
    const loss = tf.neg(tf.log(tf.sigmoid(x)).mean());
    return loss;
  }
}

class TwoTowerModel extends TwoTowerBase {
  constructor(numUsers, numItems, embDim=32, lossType='softmax'){
    super(numUsers,numItems,embDim,lossType);
    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05), true, 'userEmbedding');
    this.itemEmbedding = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05), true, 'itemEmbedding');
  }

  userForward(userIdxTensor){  // userIdxTensor: [B] or [B,1]
    const idx = userIdxTensor.reshape([-1]);
    return tf.gather(this.userEmbedding, idx);
  }
  itemForward(itemIdxTensor){  // itemIdxTensor: [B] or [B,1]
    const idx = itemIdxTensor.reshape([-1]);
    return tf.gather(this.itemEmbedding, idx);
  }

  score(uEmb, iEmb){           // dot product: [B,D]·[B,D]^T -> [B,B]
    return uEmb.matMul(iEmb, false, true);
  }

  async trainStep(userIdxArray, itemIdxArray){
    const uIdx = tf.tensor1d(userIdxArray, 'int32');
    const iIdx = tf.tensor1d(itemIdxArray, 'int32');
    const vars = [this.userEmbedding, this.itemEmbedding];

    const lossScalar = await this.opt.minimize(() => tf.tidy(() => {
      const U = this.userForward(uIdx);  // [B,D]
      const I = this.itemForward(iIdx);  // [B,D]
      const logits = this.score(U, I);   // [B,B]

      let loss;
      if (this.lossType === 'bpr') {
        // sample negatives from batch (shuffle iIdx)
        const shuffled = tf.gather(iIdx, tf.util.createShuffledIndices(iIdx.size));
        const In = this.itemForward(shuffled);
        const pos = U.mul(I).sum(1);      // [B]
        const neg = U.mul(In).sum(1);     // [B]
        loss = TwoTowerBase.bprLoss(pos, neg);
        In.dispose(); pos.dispose(); neg.dispose();
      } else {
        loss = TwoTowerBase.softmaxInBatchLoss(logits);
      }
      return loss;
    }), true, vars);

    uIdx.dispose(); iIdx.dispose();
    const val = (await lossScalar.data())[0];
    lossScalar.dispose();
    return val;
  }

  getUserEmbeddingForIndex(uIdx){ // returns [1,D]
    return tf.tidy(() => tf.gather(this.userEmbedding, tf.tensor1d([uIdx],'int32')));
  }
  getItemEmbedding(){             // returns [N,D] (variable tensor)
    return this.itemEmbedding;
  }
}

// Deep model: user tower = Embedding -> Dense(hid) -> Dense(embDim)
//             item tower = [Embedding, genres(18)] concat -> Dense(hid) -> Dense(embDim)
class TwoTowerDeepModel extends TwoTowerBase {
  constructor(numUsers, numItems, embDim=32, genreDim=18, hiddenDim=64, lossType='softmax', itemGenreTensor /*[numItems,18]*/){
    super(numUsers,numItems,embDim,lossType);
    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05), true, 'userEmbeddingDeep');
    this.itemEmbedding = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05), true, 'itemEmbeddingDeep');
    this.itemGenres = itemGenreTensor || tf.zeros([numItems, genreDim]);

    // tiny MLPs
    this.userDense1 = tf.layers.dense({units:hiddenDim, activation:'relu', useBias:true,
      kernelInitializer:'glorotUniform', name:'userDense1'});
    this.userOut   = tf.layers.dense({units:embDim, activation:null, useBias:true,
      kernelInitializer:'glorotUniform', name:'userOut'});

    this.itemDense1 = tf.layers.dense({units:hiddenDim, activation:'relu', useBias:true,
      kernelInitializer:'glorotUniform', name:'itemDense1'});
    this.itemOut   = tf.layers.dense({units:embDim, activation:null, useBias:true,
      kernelInitializer:'glorotUniform', name:'itemOut'});
  }

  userForward(userIdxTensor){
    const idx=userIdxTensor.reshape([-1]);
    const emb=tf.gather(this.userEmbedding, idx);             // [B,D]
    const h=this.userDense1.apply(emb);
    const z=this.userOut.apply(h);
    return z;
  }

  itemForward(itemIdxTensor){
    const idx=itemIdxTensor.reshape([-1]);
    const emb=tf.gather(this.itemEmbedding, idx);             // [B,D]
    const g=tf.gather(this.itemGenres, idx);                  // [B,18]
    const x=tf.concat([emb,g],1);                             // [B,D+18]
    const h=this.itemDense1.apply(x);
    const z=this.itemOut.apply(h);
    return z;
  }

  score(uEmb, iEmb){ return uEmb.matMul(iEmb, false, true); }

  async trainStep(userIdxArray, itemIdxArray){
    const uIdx = tf.tensor1d(userIdxArray, 'int32');
    const iIdx = tf.tensor1d(itemIdxArray, 'int32');

    // collect trainable variables (embeddings + layer weights)
    const vars = [
      this.userEmbedding, this.itemEmbedding,
      ...this.userDense1.getWeights(), ...this.userOut.getWeights(),
      ...this.itemDense1.getWeights(), ...this.itemOut.getWeights()
    ];

    const lossScalar = await this.opt.minimize(() => tf.tidy(() => {
      const U = this.userForward(uIdx);   // [B,D]
      const I = this.itemForward(iIdx);   // [B,D]
      let loss;
      if (this.lossType === 'bpr') {
        const shuffled = tf.gather(iIdx, tf.util.createShuffledIndices(iIdx.size));
        const In = this.itemForward(shuffled);
        const pos = U.mul(I).sum(1);
        const neg = U.mul(In).sum(1);
        loss = TwoTowerBase.bprLoss(pos, neg);
        In.dispose(); pos.dispose(); neg.dispose();
      } else {
        const logits = this.score(U, I);  // [B,B]
        loss = TwoTowerBase.softmaxInBatchLoss(logits);
      }
      return loss;
    }), true, vars);

    uIdx.dispose(); iIdx.dispose();
    const val = (await lossScalar.data())[0];
    lossScalar.dispose();
    return val;
  }

  getUserEmbeddingForIndex(uIdx){ return tf.tidy(()=>this.userForward(tf.tensor1d([uIdx],'int32'))); }
}
