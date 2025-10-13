/* two-tower.js */
(() => {
  let _uid = 0;
  function nextScope(){ _uid++; return `tt_${Date.now()}_${_uid}`; }

  class TwoTower {
    constructor(numUsers, numItems, embDim, {learningRate=1e-3}={}){
      this.nu=numUsers; this.ni=numItems; this.d=embDim;
      this.scope = nextScope();
      this.userEmb = tf.variable(tf.randomNormal([this.nu, this.d],0,0.05), true, `${this.scope}/U`);
      this.itemEmb = tf.variable(tf.randomNormal([this.ni, this.d],0,0.05), true, `${this.scope}/I`);
      this.opt = tf.train.adam(learningRate);
    }
    dispose(){
      this.userEmb.dispose(); this.itemEmb.dispose(); this.opt.dispose?.();
    }
    // In-batch softmax (diagonal are positives)
    trainStep(uIdx, iIdx){
      return this.opt.minimize(() => {
        const U = tf.gather(this.userEmb, uIdx);  // [B,d]
        const I = tf.gather(this.itemEmb, iIdx);  // [B,d]
        const logits = tf.matMul(U, I, false, true);         // [B,B]
        const labels = tf.eye(logits.shape[0]);              // one-hots (small B; OK)
        const loss = tf.losses.softmaxCrossEntropy(labels, logits).mean();
        return loss;
      }, true).data().then(a=>a[0]);
    }
    scoreUserAgainstAll(uIdx1){
      const U = tf.gather(this.userEmb, uIdx1);             // [1,d]
      const scores = tf.matMul(U, this.itemEmb, false, true);// [1,ni]
      return scores.squeeze();
    }
  }
  window.TwoTower = TwoTower;
})();
