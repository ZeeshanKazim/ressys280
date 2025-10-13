/* sasrec.js */
(() => {
  let _uid = 0;
  function nextScope(){ _uid++; return `sa_${Date.now()}_${_uid}`; }

  class SASRec {
    constructor(numItems, d=32, L=30, {learningRate=1e-3}={}){
      this.ni = numItems; this.d = d; this.L = L;
      this.scope = nextScope();
      this.itemEmb = tf.variable(tf.randomNormal([this.ni+1, this.d],0,0.05), true, `${this.scope}/E`); // 1..ni tokens; 0=pad
      this.posEmb  = tf.variable(tf.randomNormal([this.L, this.d],0,0.05), true, `${this.scope}/P`);
      this.wq = tf.variable(tf.randomNormal([this.d,this.d],0,0.08), true, `${this.scope}/wq`);
      this.wk = tf.variable(tf.randomNormal([this.d,this.d],0,0.08), true, `${this.scope}/wk`);
      this.wv = tf.variable(tf.randomNormal([this.d,this.d],0,0.08), true, `${this.scope}/wv`);
      this.wo = tf.variable(tf.randomNormal([this.d,this.d],0,0.08), true, `${this.scope}/wo`);
      this.ffn1 = tf.variable(tf.randomNormal([this.d, 4*this.d],0,0.08), true, `${this.scope}/ff1`);
      this.ffn2 = tf.variable(tf.randomNormal([4*this.d, this.d],0,0.08), true, `${this.scope}/ff2`);
      this.opt = tf.train.adam(learningRate);
      this.layernormEps = 1e-5;
    }
    dispose(){
      [this.itemEmb,this.posEmb,this.wq,this.wk,this.wv,this.wo,this.ffn1,this.ffn2].forEach(v=>v.dispose());
      this.opt.dispose?.();
    }
    layerNorm(x){
      const mean = x.mean([-1], true);
      const varr = x.sub(mean).square().mean([-1],true);
      return x.sub(mean).div(varr.add(this.layernormEps).sqrt());
    }
    // masked self-attention (causal, single head)
    selfAttention(X){ // X:[B,L,d]
      const B = X.shape[0], L = X.shape[1];
      const Q = tf.matMul(X, this.wq);
      const K = tf.matMul(X, this.wk);
      const V = tf.matMul(X, this.wv);
      let att = tf.matMul(Q, K, false, true).div(Math.sqrt(this.d)); // [B,L,L]
      // causal mask
      const mask = tf.buffer([L,L], 'float32');
      for(let i=0;i<L;i++) for(let j=0;j<L;j++) mask.set(j<=i?0:-1e9, i, j);
      const M = mask.toTensor();
      att = att.add(M);
      const P = tf.softmax(att);
      const H = tf.matMul(P, V);
      return tf.matMul(H, this.wo);
    }
    forward(seqIdx){ // seqIdx:[B,L] ints, targets last step (not included)
      const E = tf.gather(this.itemEmb, seqIdx);            // [B,L,d]
      const P = this.posEmb.reshape([1,this.L,this.d]);
      let X = this.layerNorm(E.add(P));
      const H = this.selfAttention(X);
      X = this.layerNorm(X.add(H));
      // simple FFN
      const Z1 = tf.relu(tf.matMul(X, this.ffn1));
      const Z2 = tf.matMul(Z1, this.ffn2);
      X = this.layerNorm(X.add(Z2));                        // [B,L,d]
      // take last timestep
      const last = X.slice([0, this.L-1, 0],[X.shape[0],1,this.d]).squeeze([1]); // [B,d]
      return last; // representation for next-item softmax
    }
    // full softmax over items (ni small => OK). labels ∈ [1..ni], 0 is pad
    trainStep(seqIdx, labels){
      return this.opt.minimize(() => {
        const H = this.forward(seqIdx);              // [B,d]
        const W = this.itemEmb.slice([1,0],[this.ni,this.d]); // [ni,d], tie weights
        const logits = tf.matMul(H, W, false, true); // [B,ni]
        const y = tf.oneHot(labels.sub(1), this.ni); // shift by 1 (labels in 1..ni)
        const loss = tf.losses.softmaxCrossEntropy(y, logits).mean();
        return loss;
      }, true).data().then(a=>a[0]);
    }
    scoreNext(seqIdx){ // returns [ni] scores
      const H = this.forward(seqIdx);              // [1,d]
      const W = this.itemEmb.slice([1,0],[this.ni,this.d]);
      return tf.matMul(H, W, false, true).squeeze();
    }
  }

  window.SASRec = SASRec;
})();
