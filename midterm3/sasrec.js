/* sasrec.js — tiny SASRec-style next-item ranker (single-head Transformer) */
(function(global){
  function trilMask(L){
    // mask future: (i<j) -> -1e9, else 0
    const m = tf.buffer([L,L],'float32');
    for(let i=0;i<L;i++){
      for(let j=0;j<L;j++){
        m.set(j>i ? -1e9 : 0, i, j);
      }
    }
    return m.toTensor();
  }

  class SASRecModel{
    constructor(numItems, d=32, maxLen=30, opts={}){
      this.numItems = numItems;
      this.d = d;
      this.L = maxLen;
      this.opt = tf.train.adam(opts.learningRate ?? 1e-3);
      this.numNeg = opts.numNeg ?? 5;

      this.itemEmb = tf.variable(tf.randomNormal([numItems, d], 0, 0.05), true, 'sas_itemEmb');
      this.posEmb  = tf.variable(tf.randomNormal([maxLen, d], 0, 0.05), true, 'sas_posEmb');
      // projection to keep stability
      this.Wq = tf.variable(tf.randomNormal([d,d],0,0.05), true, 'Wq');
      this.Wk = tf.variable(tf.randomNormal([d,d],0,0.05), true, 'Wk');
      this.Wv = tf.variable(tf.randomNormal([d,d],0,0.05), true, 'Wv');
      this.maskMat = trilMask(maxLen); // [L,L]
    }

    embLookup(idx1d){ return tf.gather(this.itemEmb, idx1d); } // [B] -> [B,d]

    // seqIdx: int32 [B,L], 0 means PAD; lastPos: [B] index of last valid position (for pick)
    forward(seqIdx, lastPos){
      // pad ID is 0; ensure we reserved index 0 in caller mapping
      const E = tf.gather(this.itemEmb, seqIdx);       // [B,L,d]
      const P = tf.gather(this.posEmb, tf.range(0,this.L,'int32')); // [L,d]
      const X = tf.add(E, P);                          // broadcast [B,L,d]

      const B = seqIdx.shape[0], L = this.L, d = this.d;
      const X2 = tf.reshape(X, [B*L, d]);

      const Q = tf.reshape(tf.matMul(X2, this.Wq), [B,L,d]);
      const K = tf.reshape(tf.matMul(X2, this.Wk), [B,L,d]);
      const V = tf.reshape(tf.matMul(X2, this.Wv), [B,L,d]);

      // attention scores [B, L, L]
      const scores = tf.mul(tf.matMul(Q, K, false, true), 1/Math.sqrt(d));
      const masked = tf.add(scores, this.maskMat); // add -inf to future

      const attn = tf.softmax(masked, -1);         // [B,L,L]
      const H = tf.matMul(attn, V);                // [B,L,d]

      // pick the state at last valid position
      const idx = tf.stack([tf.range(0,B,'int32'), lastPos], 1); // [B,2]
      const Hlast = tf.gatherND(H, idx);           // [B,d]
      return Hlast;
    }

    // Pairwise logistic loss with sampled negatives per example
    async trainStep(seqIdx, lastPos, posIdx, negIdx){
      const loss = this.opt.minimize(()=>{
        const H = this.forward(seqIdx, lastPos);       // [B,d]
        const ePos = tf.gather(this.itemEmb, posIdx);  // [B,d]
        const sPos = tf.sum(tf.mul(H, ePos), 1);       // [B]

        const eNeg = tf.gather(this.itemEmb, negIdx);  // [B,numNeg,d]
        const sNeg = tf.sum(tf.mul(tf.expandDims(H,1), eNeg), 2); // [B,numNeg]

        const posLoss = tf.mean(tf.softplus(tf.neg(sPos)));      // -log(sigmoid(sPos))
        const negLoss = tf.mean(tf.softplus(sNeg));              // -log(sigmoid(-sNeg)) per element
        return tf.add(posLoss, negLoss);
      }, true);
      const v = (await loss.data())[0];
      loss.dispose();
      return v;
    }

    // Score all items for a single sequence (1,L)
    scoreAll(seqIdx1, lastPos1){
      const H = this.forward(seqIdx1, lastPos1);      // [1,d]
      return tf.matMul(H, this.itemEmb, false, true).squeeze(); // [N]
    }

    dispose(){
      this.itemEmb.dispose(); this.posEmb.dispose();
      this.Wq.dispose(); this.Wk.dispose(); this.Wv.dispose();
      this.maskMat.dispose();
    }
  }

  global.SASRecModel = SASRecModel;
})(window);
