// two-tower.js — Minimal Two-Tower retrieval in TF.js
// Baseline: user/item ID embeddings with in-batch softmax
// Deep: user ID embedding + item tag MLP -> embedding; same loss

function _shuffle(a){ for(let i=a.length-1;i>0;i--){ const j=(Math.random()*(i+1))|0; [a[i],a[j]]=[a[j],a[i]]; } return a; }

class TwoTowerBaseline {
  constructor(numUsers, numItems, embDim=32, lr=1e-3){
    this.numUsers=numUsers; this.numItems=numItems; this.embDim=embDim;
    this.userEmb = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05));
    this.itemEmb = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05));
    this.opt = tf.train.adam(lr);
  }
  dispose(){ this.userEmb.dispose(); this.itemEmb.dispose(); this.opt.dispose?.(); }

  _gather(table, idx){ return tf.gather(table, idx); }

  _lossBatch(uIdx, iIdx){
    return tf.tidy(()=> {
      const U = this._gather(this.userEmb, uIdx);   // [B,D]
      const I = this._gather(this.itemEmb, iIdx);   // [B,D]
      const logits = tf.matMul(U, I, false, true);  // [B,B]
      const labels = tf.range(0, logits.shape[0], 1, 'int32');
      const loss = tf.losses.softmaxCrossEntropy(tf.oneHot(labels, logits.shape[1]), logits).mean();
      return loss;
    });
  }

  async train(interactions, idxUser, idxItem, {epochs=5, batchSize=256, onStep=null}={}){
    const pairs = interactions.map(r=>({u: idxUser.get(r.user), i: idxItem.get(r.item)}))
                              .filter(p=>p.u!=null && p.i!=null);
    const totalSteps = Math.ceil(pairs.length/batchSize)*epochs;
    let step=0;
    for (let e=0;e<epochs;e++){
      _shuffle(pairs);
      for (let s=0;s<pairs.length;s+=batchSize){
        const batch = pairs.slice(s, s+batchSize);
        const uIdx = tf.tensor1d(batch.map(p=>p.u), 'int32');
        const iIdx = tf.tensor1d(batch.map(p=>p.i), 'int32');
        const loss = this.opt.minimize(()=>this._lossBatch(uIdx,iIdx), true);
        const val = (await loss.data())[0];
        uIdx.dispose(); iIdx.dispose(); loss.dispose();
        onStep && onStep(++step, totalSteps, val);
        await tf.nextFrame();
      }
    }
  }

  async recommend(userId, USERS, idxUser, idxItem, RECIPES, topK=50){
    const seen = new Set((USERS.get(userId)||[]).map(x=>x.item));
    const u = idxUser.get(userId); if (u==null) return [];
    return tf.tidy(()=>{
      const uEmb = tf.gather(this.userEmb, tf.tensor1d([u],'int32')); // [1,D]
      const scores = tf.matMul(uEmb, this.itemEmb, false, true).reshape([this.numItems]);
      const arr = scores.dataSync();
      const ranked = [];
      for (let j=0;j<this.numItems;j++){
        const itemId = window.DATASET.revItem[j];
        if (seen.has(itemId)) continue;
        ranked.push({itemId, score: arr[j], title: (RECIPES.get(itemId)?.title)||('Recipe '+itemId)});
      }
      ranked.sort((a,b)=>b.score-a.score);
      return ranked.slice(0, topK);
    });
  }

  async projectItems2D(sampleN=500){
    const N = Math.min(sampleN, this.numItems);
    const idx = tf.tensor1d([...Array(N).keys()], 'int32');
    const M = tf.gather(this.itemEmb, idx); // [N,D]
    // simple PCA: covariance via SVD on centered matrix
    const mean = M.mean(0);
    const X = M.sub(mean);
    const {u,s,v} = tf.linalg.svd(tf.matMul(X, X, true, false)); // [N,N]
    const U2 = u.slice([0,0],[N,2]); // top-2
    const proj = U2.mul(s.slice([0],[2]).sqrt().reshape([1,2])); // scale
    const xy = await proj.array();
    idx.dispose(); M.dispose(); mean.dispose(); X.dispose(); u.dispose(); s.dispose(); v.dispose(); U2.dispose(); proj.dispose();
    return xy;
  }
}

class TwoTowerDeep {
  constructor(numUsers, numItems, embDim, featDim, lr, itemFeatures /* Float32Array[] */){
    this.numUsers=numUsers; this.numItems=numItems; this.embDim=embDim; this.featDim=featDim;
    // towers
    this.userEmb = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05));
    // item MLP: featDim -> 2*emb -> emb
    this.W1 = tf.variable(tf.randomNormal([featDim, embDim*2], 0, 0.05));
    this.b1 = tf.variable(tf.zeros([embDim*2]));
    this.W2 = tf.variable(tf.randomNormal([embDim*2, embDim], 0, 0.05));
    this.b2 = tf.variable(tf.zeros([embDim]));
    this.itemFeat = tf.tensor2d(itemFeatures.map(x=>Array.from(x)), [numItems, featDim], 'float32');
    this.opt = tf.train.adam(lr);
  }
  dispose(){
    this.userEmb.dispose();
    this.W1.dispose(); this.b1.dispose(); this.W2.dispose(); this.b2.dispose();
    this.itemFeat.dispose(); this.opt.dispose?.();
  }

  _itemForward(idx){ // idx: [B]
    return tf.tidy(()=>{
      const F = tf.gather(this.itemFeat, idx);      // [B,F]
      const h1 = tf.relu(tf.add(tf.matMul(F, this.W1), this.b1)); // [B,2E]
      const z  = tf.add(tf.matMul(h1, this.W2), this.b2);         // [B,E]
      return z;
    });
  }
  _lossBatch(uIdx, iIdx){
    return tf.tidy(()=>{
      const U = tf.gather(this.userEmb, uIdx); // [B,E]
      const I = this._itemForward(iIdx);       // [B,E]
      const logits = tf.matMul(U, I, false, true); // [B,B]
      const labels = tf.range(0, logits.shape[0], 1, 'int32');
      const loss = tf.losses.softmaxCrossEntropy(tf.oneHot(labels, logits.shape[1]), logits).mean();
      return loss;
    });
  }
  async train(interactions, idxUser, idxItem, {epochs=5,batchSize=256,onStep=null}={}){
    const pairs = interactions.map(r=>({u: idxUser.get(r.user), i: idxItem.get(r.item)}))
                              .filter(p=>p.u!=null && p.i!=null);
    const totalSteps = Math.ceil(pairs.length/batchSize)*epochs; let step=0;
    for (let e=0;e<epochs;e++){
      _shuffle(pairs);
      for (let s=0;s<pairs.length;s+=batchSize){
        const batch = pairs.slice(s, s+batchSize);
        const uIdx = tf.tensor1d(batch.map(p=>p.u),'int32');
        const iIdx = tf.tensor1d(batch.map(p=>p.i),'int32');
        const loss = this.opt.minimize(()=>this._lossBatch(uIdx,iIdx), true);
        const val=(await loss.data())[0];
        uIdx.dispose(); iIdx.dispose(); loss.dispose();
        onStep && onStep(++step, totalSteps, val);
        await tf.nextFrame();
      }
    }
  }

  async recommend(userId, USERS, idxUser, idxItem, RECIPES, topK=50){
    const seen = new Set((USERS.get(userId)||[]).map(x=>x.item));
    const u = idxUser.get(userId); if (u==null) return [];
    return tf.tidy(()=>{
      const uEmb = tf.gather(this.userEmb, tf.tensor1d([u],'int32')); // [1,E]
      // compute item tower for all items in batches to limit memory
      const B = 1024;
      const scores = new Float32Array(this.numItems);
      for (let s=0;s<this.numItems;s+=B){
        const idx = tf.tensor1d([...Array(Math.min(B,this.numItems-s)).keys()].map(k=>k+s),'int32');
        const IZ = this._itemForward(idx); // [b,E]
        const sc = tf.matMul(uEmb, IZ, false, true).reshape([IZ.shape[0]]);
        sc.dataSync().forEach((v,off)=>{ scores[s+off]=v; });
        idx.dispose(); IZ.dispose(); sc.dispose();
      }
      const ranked = [];
      for (let j=0;j<this.numItems;j++){
        const itemId = window.DATASET.revItem[j];
        if (seen.has(itemId)) continue;
        ranked.push({itemId, score: scores[j], title: RECIPES.get(itemId)?.title || ('Recipe '+itemId)});
      }
      ranked.sort((a,b)=>b.score-a.score);
      return ranked.slice(0, topK);
    });
  }

  async projectItems2D(sampleN=500){
    const N = Math.min(sampleN, this.numItems);
    const idx = tf.tensor1d([...Array(N).keys()], 'int32');
    const Z = this._itemForward(idx);
    const mean = Z.mean(0);
    const X = Z.sub(mean);
    const {u,s,v} = tf.linalg.svd(tf.matMul(X, X, true, false));
    const U2 = u.slice([0,0],[N,2]);
    const proj = U2.mul(s.slice([0],[2]).sqrt().reshape([1,2]));
    const xy = await proj.array();
    idx.dispose(); Z.dispose(); mean.dispose(); X.dispose(); u.dispose(); s.dispose(); v.dispose(); U2.dispose(); proj.dispose();
    return xy;
  }
}

window.TwoTowerBaseline = TwoTowerBaseline;
window.TwoTowerDeep = TwoTowerDeep;
