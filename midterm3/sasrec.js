/* sasrec.js — minimal SASRec in TF.js (1-head, 2 blocks).
   Exposes:
     class SASRec
       constructor(nItemsPlus1, d, L, opts)
       async compile()
       buildSamples(userSeqs, maxRows, numNeg) -> {X, posY, negY}
       async train(samples, epochs, batch)
       scoreNext(seqIdxTensor, candIdxTensorPlus1) -> [B, C]
*/

class SASRec {
  constructor(nItemsPlus1, d=32, L=50, opts={}){
    this.nItemsPlus1 = nItemsPlus1; // include padding id 0
    this.d = d; this.L = L;
    this.lr = opts.learningRate || 1e-3;
    this.dropout = 0.1;

    // Embeddings
    this.itemEmb = tf.variable(tf.randomNormal([nItemsPlus1, d], 0, 0.05));
    this.posEmb  = tf.variable(tf.randomNormal([L, d], 0, 0.05));

    // 2 Transformer blocks (single head)
    const makeBlock = () => ({
      Wq: tf.variable(tf.randomNormal([d, d], 0, 0.05)),
      Wk: tf.variable(tf.randomNormal([d, d], 0, 0.05)),
      Wv: tf.variable(tf.randomNormal([d, d], 0, 0.05)),
      W1: tf.variable(tf.randomNormal([d, 4*d], 0, 0.05)),
      b1: tf.variable(tf.zeros([4*d])),
      W2: tf.variable(tf.randomNormal([4*d, d], 0, 0.05)),
      b2: tf.variable(tf.zeros([d])),
    });
    this.blocks = [makeBlock(), makeBlock()];
    this.optimizer = tf.train.adam(this.lr);
  }

  async compile(){ /* no-op */ }

  _drop(x){ return tf.dropout(x, this.dropout); }

  _block(X, blk){ // X [B,L,d] causal self-attn + FFN
    return tf.tidy(() => {
      const B = X.shape[0], L = X.shape[1];

      const Q = X.matMul(blk.Wq);
      const K = X.matMul(blk.Wk);
      const V = X.matMul(blk.Wv);
      // scaled dot attention with causal mask
      let scores = tf.matMul(Q, K, false, true); // [B,L,L]
      scores = scores.div(Math.sqrt(this.d));
      // causal mask: allow j<=i
      const mask = tf.buffer([L, L], 'float32');
      for (let i=0;i<L;i++){ for (let j=0;j<L;j++){ mask.set(j<=i?0:-1e9, i, j); } }
      const M = tf.tensor(mask.toTensor().dataSync(), [L,L]);
      scores = scores.add(M);
      const attn = tf.softmax(scores, -1);
      const C = tf.matMul(attn, V); // [B,L,d]
      let Y = X.add(this._drop(C)); // residual

      // FFN
      const H = tf.relu(Y.matMul(blk.W1).add(blk.b1)); // [B,L,4d]
      const Z = H.matMul(blk.W2).add(blk.b2);          // [B,L,d]
      Y = Y.add(this._drop(Z));
      return Y;
    });
  }

  // seqIdx: [B,L] (0 padded), candPlus1: [B,C] candidate item ids (offset +1)
  scoreNext(seqIdx, candPlus1){
    return tf.tidy(() => {
      const B = seqIdx.shape[0], L = seqIdx.shape[1];
      let X = tf.gather(this.itemEmb, seqIdx); // [B,L,d]
      const P = tf.expandDims(this.posEmb, 0).tile([B,1,1]);
      X = X.add(P);
      let Y = X;
      for (const blk of this.blocks) Y = this._block(Y, blk);
      // take last timestep representation
      const h = Y.gather([L-1], 1).squeeze([1]); // [B,d]
      const candE = tf.gather(this.itemEmb, candPlus1); // [B,C,d]
      const logits = tf.einsum('bd,bcd->bc', h, candE);
      return logits;
    });
  }

  buildSamples(userSeqs, maxRows=120000, numNeg=5){
    // userSeqs: Map(userId -> array of itemIdx+1, chronological, >=3)
    const rows = [];
    userSeqs.forEach(arr => {
      for (let t=1; t<arr.length; t++){
        const seq = arr.slice(Math.max(0, t-this.L), t);
        const target = arr[t];
        rows.push({seq, target});
      }
    });
    if (!rows.length) return null;
    // crop
    const R = rows.slice(0, maxRows);
    // build tensors with padding + negatives to be sampled later
    const padSeqs = R.map(r=>{
      const s = r.seq.slice(-this.L);
      const pad = Array(Math.max(0, this.L - s.length)).fill(0);
      return pad.concat(s);
    });
    const pos = R.map(r=>r.target);
    return { seqs: padSeqs, pos, numNeg };
  }

  async train(samples, epochs=3, batch=256){
    if (!samples) return {steps:0};
    const {seqs, pos, numNeg} = samples;
    const N = seqs.length;
    let steps=0;
    for (let ep=0; ep<epochs; ep++){
      for (let i=0; i<N; i+=batch){
        const S = seqs.slice(i, i+batch);
        const P = pos.slice(i, i+batch);
        const B = S.length;
        // negatives
        const neg = [];
        for (let k=0;k<B;k++){
          const arr = [];
          for (let n=0;n<numNeg;n++){
            let x = 1 + Math.floor(Math.random()*(this.nItemsPlus1-1));
            arr.push(x);
          }
          neg.push(arr);
        }

        const loss = await this.optimizer.minimize(() => tf.tidy(() => {
          const seqT = tf.tensor2d(S, [B, this.L], 'int32');
          // candidates per row: [pos, neg...]
          const C = P.map((p,idx)=> [p, ...neg[idx]]);
          const cT = tf.tensor2d(C, [B, 1+numNeg], 'int32');
          const logits = this.scoreNext(seqT, cT); // [B,1+neg]
          const labels = tf.tensor1d(new Array(B).fill(0), 'int32'); // index 0 = pos
          const loss = tf.losses.softmaxCrossEntropy(tf.oneHot(labels, 1+numNeg), logits).mean();
          return loss;
        }), true);
        steps++;
        if (this.onStep) this.onStep({ep:ep+1, step:steps, loss:loss.dataSync()[0]});
      }
    }
    return {steps};
  }

  dispose(){
    [this.itemEmb, this.posEmb].forEach(v=>v?.dispose?.());
    this.blocks?.forEach(b=>{
      [b.Wq,b.Wk,b.Wv,b.W1,b.b1,b.W2,b.b2].forEach(v=>v?.dispose?.());
    });
  }
}

window.SASRec = SASRec;
