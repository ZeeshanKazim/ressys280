/* two-tower.js
   Minimal retrieval models in TensorFlow.js

   1) TwoTowerModel (Baseline): ID embeddings only, trained with in-batch softmax.
   2) TwoTowerDeepModel (Deep): user ID embedding + item tower = concat(ID emb, tag BoW) -> Dense -> item emb.

   Loss: in-batch softmax (sampled by batch). For batch of users U and positive items I+:
   logits = U @ I^T ; labels are diag indices; softmax cross-entropy.

   Methods exposed:
   - constructor(...)
   - async train(uIdxArray, iIdxArray, {epochs,batchSize,onBatch})
   - getUserEmbedding(uIdx)  // returns tf.Tensor [D]
   - getItemEmbeddingMatrix() // tf.Variable [numItems, D]
   - dispose()
*/

class TwoTowerModel {
  constructor(numUsers, numItems, embDim=32, lr=1e-3){
    this.numUsers = numUsers; this.numItems=numItems; this.embDim=embDim;

    // Tables
    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05), true, 'tt_user');
    this.itemEmbedding = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05), true, 'tt_item');

    this.optimizer = tf.train.adam(lr);
  }

  // gather rows
  userForward(uIdx){ return tf.gather(this.userEmbedding, uIdx); }   // [B,D]
  itemForward(iIdx){ return tf.gather(this.itemEmbedding, iIdx); }   // [B,D]

  score(uEmb, iEmb){ return tf.matMul(uEmb, iEmb, false, true); }     // [B,B]

  _batchLoss(uIdx, iIdx){
    return tf.tidy(()=>{
      const uEmb = this.userForward(uIdx);   // [B,D]
      const iEmb = this.itemForward(iIdx);   // [B,D]
      const logits = this.score(uEmb, iEmb); // [B,B]
      const labels = tf.range(0, uIdx.size, 1, 'int32'); // diag
      const loss = tf.losses.softmaxCrossEntropy(tf.oneHot(labels, uIdx.size), logits).mean();
      return loss;
    });
  }

  async train(uArr, iArr, {epochs=5, batchSize=256, onBatch=()=>{}}={}){
    const N = uArr.length;
    const stepsPerEpoch = Math.ceil(N/batchSize);
    for(let e=0;e<epochs;e++){
      for(let s=0;s<stepsPerEpoch;s++){
        const a = s*batchSize, b = Math.min(N, a+batchSize);
        const u = tf.tensor1d(uArr.slice(a,b), 'int32');
        const i = tf.tensor1d(iArr.slice(a,b), 'int32');

        const loss = this.optimizer.minimize(()=>this._batchLoss(u,i), true);
        const val = (await loss.data())[0]; loss.dispose(); u.dispose(); i.dispose();
        onBatch?.(e*stepsPerEpoch + s, val);
        await tf.nextFrame();
      }
    }
  }

  getUserEmbedding(uIdx){ return tf.tidy(()=> tf.gather(this.userEmbedding, tf.tensor1d([uIdx],'int32')).squeeze() ); }
  getItemEmbeddingMatrix(){ return this.itemEmbedding; }

  dispose(){
    this.userEmbedding.dispose();
    this.itemEmbedding.dispose();
    this.optimizer.dispose && this.optimizer.dispose();
  }
}

/* Deep variant: item tower uses tags (bag-of-words) with one hidden layer -> item embedding.
   We concatenate item ID embedding + tag MLP output, project to embDim.
*/
class TwoTowerDeepModel {
  constructor({numUsers, numItems, embDim=32, lr=1e-3, tagVocabSize=0, itemTagMatrix=null}){
    this.numUsers=numUsers; this.numItems=numItems; this.embDim=embDim;
    this.tagVocabSize = tagVocabSize;
    this.itemTagMatrix = itemTagMatrix; // Float32Array length M*K (dense)

    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05), true, 'deep_user');
    this.itemIdEmbedding = tf.variable(tf.randomNormal([numItems, Math.max(8, Math.floor(embDim/2))], 0, 0.05), true, 'deep_item_id');

    // Item tag weights (one hidden layer + output)
    const hidden = Math.max(16, Math.floor(embDim*2));
    this.W1 = tf.variable(tf.randomNormal([tagVocabSize, hidden], 0, 0.05), true, 'deep_W1');
    this.b1 = tf.variable(tf.zeros([hidden]), true, 'deep_b1');
    this.W2 = tf.variable(tf.randomNormal([hidden, embDim], 0, 0.05), true, 'deep_W2');
    this.b2 = tf.variable(tf.zeros([embDim]), true, 'deep_b2');

    // Final projection after concat (idEmb + tagEmb)
    const totalDim = Math.max(8, Math.floor(embDim/2)) + embDim;
    this.Wp = tf.variable(tf.randomNormal([totalDim, embDim], 0, 0.05), true, 'deep_Wp');
    this.bp = tf.variable(tf.zeros([embDim]), true, 'deep_bp');

    this.optimizer = tf.train.adam(lr);

    // Pre-build constant tag matrix tensor (on demand)
    this._tagsTensor = null;
  }

  _ensureTagTensor(){
    if (this._tagsTensor) return;
    if (this.tagVocabSize===0){ this._tagsTensor = tf.zeros([this.numItems, 0]); return; }
    const M=this.numItems, K=this.tagVocabSize;
    const x = tf.tensor2d(this.itemTagMatrix, [M,K], 'float32');
    this._tagsTensor = x;
  }

  userForward(uIdx){ return tf.gather(this.userEmbedding, uIdx); } // [B,D]

  itemForward(iIdx){
    this._ensureTagTensor();
    const idEmb = tf.gather(this.itemIdEmbedding, iIdx);          // [B,Did]
    const tags  = tf.gather(this._tagsTensor, iIdx);              // [B,K]
    const h = tf.relu(tags.matMul(this.W1).add(this.b1));         // [B,H]
    const tagEmb = h.matMul(this.W2).add(this.b2);                // [B,D]
    const concat = tf.concat([idEmb, tagEmb], 1);                 // [B,Did+D]
    const proj = concat.matMul(this.Wp).add(this.bp);             // [B,D]
    return proj;
  }

  score(uEmb, iEmb){ return tf.matMul(uEmb, iEmb, false, true); }

  _batchLoss(uIdx, iIdx){
    return tf.tidy(()=>{
      const uEmb = this.userForward(uIdx);   // [B,D]
      const iEmb = this.itemForward(iIdx);   // [B,D]
      const logits = this.score(uEmb, iEmb); // [B,B]
      const labels = tf.range(0, uIdx.size, 1, 'int32');
      const loss = tf.losses.softmaxCrossEntropy(tf.oneHot(labels, uIdx.size), logits).mean();
      return loss;
    });
  }

  async train(uArr, iArr, {epochs=5, batchSize=256, onBatch=()=>{}}={}){
    const N = uArr.length;
    const stepsPerEpoch = Math.ceil(N/batchSize);
    for(let e=0;e<epochs;e++){
      for(let s=0;s<stepsPerEpoch;s++){
        const a = s*batchSize, b = Math.min(N, a+batchSize);
        const u = tf.tensor1d(uArr.slice(a,b), 'int32');
        const i = tf.tensor1d(iArr.slice(a,b), 'int32');
        const loss = this.optimizer.minimize(()=>this._batchLoss(u,i), true);
        const val = (await loss.data())[0]; loss.dispose(); u.dispose(); i.dispose();
        onBatch?.(e*stepsPerEpoch + s, val);
        await tf.nextFrame();
      }
    }
  }

  getUserEmbedding(uIdx){ return tf.tidy(()=> tf.gather(this.userEmbedding, tf.tensor1d([uIdx],'int32')).squeeze()); }
  getItemEmbeddingMatrix(){
    this._ensureTagTensor();
    // Materialize full item embeddings (for scoring/projection)
    return tf.tidy(()=>{
      const allIdx = tf.range(0, this.numItems, 1, 'int32');
      const idEmb = tf.gather(this.itemIdEmbedding, allIdx);
      const tags  = tf.gather(this._tagsTensor, allIdx);
      const h = tf.relu(tags.matMul(this.W1).add(this.b1));
      const tagEmb = h.matMul(this.W2).add(this.b2);
      const concat = tf.concat([idEmb, tagEmb], 1);
      return tf.variable(concat.matMul(this.Wp).add(this.bp), true, 'deep_items_full');
    });
  }

  dispose(){
    this.userEmbedding.dispose(); this.itemIdEmbedding.dispose();
    this.W1.dispose(); this.b1.dispose(); this.W2.dispose(); this.b2.dispose();
    this.Wp.dispose(); this.bp.dispose();
    this._tagsTensor?.dispose();
    this.optimizer.dispose && this.optimizer.dispose();
  }
}

// expose
window.TwoTowerModel = TwoTowerModel;
window.TwoTowerDeepModel = TwoTowerDeepModel;
