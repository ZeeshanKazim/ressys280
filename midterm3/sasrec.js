/* sasrec.js — Minimal SASRec (single-head self-attention) with BPR loss
   - Padding id = 0; items are 1..numItems-1 internally. We pass in idx arrays already mapped.
   - We build variables without explicit names to avoid TFJS global name collisions.
*/
class SASRec {
  constructor(numItems, embDim=32, maxLen=30, opts={}){
    this.numItems = numItems;             // includes padding row
    this.embDim = embDim|0;
    this.maxLen = maxLen|0;
    this.negatives = opts.negatives ?? 5;
    this.lr = opts.learningRate ?? 1e-3;

    // Embeddings
    this.itemEmb = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05)); // row 0 = pad
    this.posEmb  = tf.variable(tf.randomNormal([maxLen, embDim], 0, 0.05));

    // Single-head projection weights
    this.Wq = tf.variable(tf.randomNormal([embDim, embDim], 0, 0.05));
    this.Wk = tf.variable(tf.randomNormal([embDim, embDim], 0, 0.05));
    this.Wv = tf.variable(tf.randomNormal([embDim, embDim], 0, 0.05));
    this.Wo = tf.variable(tf.randomNormal([embDim, embDim], 0, 0.05));

    // LayerNorm params
    this.g1 = tf.variable(tf.ones([embDim]));
    this.b1 = tf.variable(tf.zeros([embDim]));

    this.optimizer = tf.train.adam(this.lr);
  }

  dispose(){ Object.values(this).forEach(v=>v?.dispose?.()); }

  _layerNorm(x, g, b, eps=1e-5){
    const mean = tf.mean(x, -1, true);
    const varr = tf.mean(tf.square(x.sub(mean)), -1, true);
    return x.sub(mean).div(tf.sqrt(varr.add(eps))).mul(g).add(b);
  }

  /** Encode sequence tensor S: [B, L] (int32) → representation H_last: [B, D] */
  _encode(S){
    return tf.tidy(()=>{
      const B = S.shape[0], L = S.shape[1];
      const E = tf.gather(this.itemEmb, S);     // [B,L,D]
      // + positional
      const P = tf.tile(this.posEmb.slice([0,0],[L,this.embDim]).expandDims(0), [B,1,1]); // [B,L,D]
      let X = E.add(P);

      // projections
      const Q = X.matMul(this.Wq); // [B,L,D]
      const K = X.matMul(this.Wk); // [B,L,D]
      const V = X.matMul(this.Wv); // [B,L,D]

      // attention scores with causal + padding mask
      const KT = K.transpose([0,2,1]); // [B,D,L]
      let att = Q.matMul(KT).div(Math.sqrt(this.embDim)); // [B,L,L]

      // causal mask (upper triangle = -inf)
      const m = tf.buffer([L,L], 'float32');
      for(let i=0;i<L;i++){ for(let j=0;j<L;j++){ m.set(j<=i?0:-1e9,i,j); } }
      let M = tf.tensor(m.toTensor()); // [L,L]
      att = att.add(M); M.dispose();

      // padding mask: don't attend to pads (token id 0)
      const pad = S.equal(0).toFloat(); // [B,L]
      const padMask = pad.mul(-1e9).expandDims(1); // [B,1,L]
      att = att.add(padMask);

      // softmax and value mix
      const Wt = tf.softmax(att, -1);      // [B,L,L]
      let Z = Wt.matMul(V);                // [B,L,D]
      Z = Z.matMul(this.Wo);
      Z = this._layerNorm(Z, this.g1, this.b1);

      // take representation at last non-pad position for each sequence
      const notPad = S.notEqual(0).toInt();             // [B,L]
      const len = tf.sum(notPad, 1);                    // [B]
      const idx = tf.maximum(len.sub(1), tf.zerosLike(len)); // [B]
      const oneH = tf.oneHot(idx, L).expandDims(1);     // [B,1,L]
      const Hlast = oneH.matMul(Z).squeeze([1]);        // [B,D]
      return Hlast;
    });
  }

  /** Train BPR: inputs seq [B,L], pos [B] (next item id), with K negatives */
  async trainStep(seq, posIdx, samplerFn){
    const K = this.negatives;
    const lossVal = await this.optimizer.minimize(()=>{
      const H = this._encode(seq); // [B,D]
      const posE = tf.gather(this.itemEmb, posIdx); // [B,D]
      const sPos = tf.sum(H.mul(posE), -1);         // [B]

      // sample K negatives per batch element
      const negIdx = samplerFn ? samplerFn(posIdx, K) : tf.randomUniform([posIdx.shape[0], K], 1, this.numItems, 'int32');
      const negE = tf.gather(this.itemEmb, negIdx); // [B,K,D]
      const Hexp = H.expandDims(1);                 // [B,1,D]
      const sNeg = tf.sum(Hexp.mul(negE), -1);      // [B,K]

      // BPR: sum log σ(s_pos - s_neg) over K
      const diff = sPos.expandDims(1).sub(sNeg);    // [B,K]
      const loss = tf.neg(tf.mean(tf.logSigmoid(diff)));
      return loss;
    }, true);
    return lossVal.dataSync()[0];
  }

  /** Score all items for one sequence [1,L] */
  scoreAll(seq1){
    return tf.tidy(()=>{
      const H = this._encode(seq1); // [1,D]
      return this.itemEmb.matMul(H.transpose()).squeeze(); // [N]
    });
  }

  /** Score specific item indices for one sequence */
  scoreItems(seq1, itemIdxArr){
    return tf.tidy(()=>{
      const H = this._encode(seq1); // [1,D]
      const I = tf.tensor1d(itemIdxArr, 'int32');
      const E = tf.gather(this.itemEmb, I);
      const s = E.matMul(H.transpose()).squeeze();
      I.dispose(); return s;
    });
  }
}
