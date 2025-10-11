/* two-tower.js
   Minimal Two-Tower for retrieval in TF.js.

   Modes:
   - "baseline": user_id -> emb, item_id -> emb, score = dot(U, I)
   - "deep":     user_id -> emb; item -> concat(id_emb, tag_bag_emb) -> MLP -> emb
                 score = dot(U, I_deep)

   Loss: in-batch softmax (sampled softmax with negatives = other items in the batch).
*/

class TwoTowerModel {
  constructor(opts) {
    this.numUsers = opts.numUsers;
    this.numItems = opts.numItems;
    this.embDim   = opts.embDim || 32;
    this.mode     = opts.mode || 'baseline';       // 'baseline' | 'deep'
    this.maxTagId = opts.maxTagId || 0;            // only for deep
    this.tagDim   = opts.tagDim || Math.max(8, Math.min(32, Math.floor(this.embDim/2)));
    this.mlpHidden= opts.mlpHidden || [64];
    this.lr       = opts.lr || 1e-3;

    // Variables (always)
    this.userEmbedding = tf.variable(tf.randomNormal([this.numUsers, this.embDim], 0, 0.05), true, 'userEmbedding');
    this.itemEmbedding = tf.variable(tf.randomNormal([this.numItems, this.embDim], 0, 0.05), true, 'itemEmbedding');

    // Deep-only extras
    if (this.mode === 'deep') {
      this.tagEmbedding = tf.variable(tf.randomNormal([this.maxTagId + 1, this.tagDim], 0, 0.05), true, 'tagEmbedding');
      // MLP weights: input = embDim (id) + tagDim
      let prev = this.embDim + this.tagDim;
      this.mlpWeights = [];
      for (const h of this.mlpHidden) {
        const w = tf.variable(tf.randomNormal([prev, h], 0, Math.sqrt(2/prev)), true);
        const b = tf.variable(tf.zeros([h]), true);
        this.mlpWeights.push({w,b});
        prev = h;
      }
      // Output proj to embDim
      this.outW = tf.variable(tf.randomNormal([prev, this.embDim], 0, Math.sqrt(2/prev)), true);
      this.outB = tf.variable(tf.zeros([this.embDim]), true);
    }

    this.optimizer = tf.train.adam(this.lr);
  }

  // Gather rows by int32 indices -> [B, D]
  static gather(table, idx1D) { return tf.gather(table, idx1D.flatten()); }

  userForward(userIdx) {
    return TwoTowerModel.gather(this.userEmbedding, userIdx);
  }

  // For baseline: just id embedding.
  itemForwardBaseline(itemIdx) {
    return TwoTowerModel.gather(this.itemEmbedding, itemIdx);
  }

  // For deep: id emb + tag-bag emb -> MLP -> D
  // tagBag: ragged representation given as {indices: Int32Array, splits: Int32Array}
  // where splits len = B+1, spans in indices for each item in batch.
  itemForwardDeep(itemIdx, tagBag) {
    const idEmb = TwoTowerModel.gather(this.itemEmbedding, itemIdx); // [B,D]
    const B = idEmb.shape[0];

    // Bag-mean tag embedding
    const indices = tf.tensor1d(tagBag.indices, 'int32');
    const splits  = tf.tensor1d(tagBag.splits, 'int32'); // len B+1
    const tagRows = tf.gather(this.tagEmbedding, indices); // [NNZ, tagDim]

    // segmentMean by ragged splits
    const counts = tf.sub(splits.slice(1), splits.slice(0, B)); // [B]
    const segIds = tf.tidy(() => tf.concat(
      Array.from({length:B}, (_,i) => tf.fill([counts.arraySync()[i]], i)), 0
    ));
    const summed = tf.unsortedSegmentSum(tagRows, segIds, B); // [B, tagDim]
    const denom  = tf.maximum(1, counts).reshape([B,1]);
    const tagMean = tf.div(summed, denom);

    // concat & MLP (ReLU)
    let x = tf.concat([idEmb, tagMean], 1);
    for (const {w,b} of this.mlpWeights) x = tf.relu(tf.add(tf.matMul(x, w), b));
    const out = tf.add(tf.matMul(x, this.outW), this.outB);   // [B,embDim]

    indices.dispose(); splits.dispose(); tagRows.dispose(); counts.dispose(); segIds.dispose(); summed.dispose(); denom.dispose();
    return out;
  }

  itemForward(itemIdx, tagBagOpt) {
    return (this.mode === 'deep')
      ? this.itemForwardDeep(itemIdx, tagBagOpt)
      : this.itemForwardBaseline(itemIdx);
  }

  // In-batch softmax loss: logits = U · I^T, labels = diagonal
  lossInBatch(usersEmb, itemsEmb) {
    const logits = tf.matMul(usersEmb, itemsEmb, false, true); // [B,B]
    const labels = tf.tensor1d(Array.from({length:logits.shape[0]}, (_,i)=>i), 'int32');
    const ce = tf.losses.softmaxCrossEntropy(tf.oneHot(labels, logits.shape[1]), logits).mean();
    labels.dispose();
    return ce;
  }

  // Single training step on one batch
  // batch: {uIdx: Int32Array, iIdx: Int32Array, tagBag?: {indices:Int32Array,splits:Int32Array}}
  trainStep(batch) {
    return this.optimizer.minimize(() => {
      const uIdx = tf.tensor1d(batch.uIdx, 'int32');
      const iIdx = tf.tensor1d(batch.iIdx, 'int32');
      const uEmb = this.userForward(uIdx);
      const iEmb = this.itemForward(iIdx, batch.tagBag);
      const loss = this.lossInBatch(uEmb, iEmb);
      uIdx.dispose(); iIdx.dispose(); uEmb.dispose(); iEmb.dispose();
      return loss;
    }, true).dataSync()[0];
  }

  // Inference helpers
  getUserEmbedding(uIdx) {
    return tf.tidy(()=>this.userEmbedding.gather(tf.tensor1d([uIdx],'int32')).squeeze());
  }

  // Returns [numItems, embDim] tensor
  getAllItemEmbeddings(itemTagRaggedOpt) {
    if (this.mode === 'baseline') return this.itemEmbedding;
    // Deep: compute in manageable batches, return stacked (as a tensor)
    const B = 1024;
    const out = [];
    for (let start = 0; start < this.numItems; start += B) {
      const end = Math.min(this.numItems, start + B);
      const idx = tf.tensor1d([...Array(end-start).keys()].map(i=>i+start), 'int32');
      const tagBag = itemTagRaggedOpt.slice(start, end);
      const emb = this.itemForwardDeep(idx, tagBag);
      out.push(emb);
      idx.dispose();
    }
    return tf.concat(out, 0);
  }
}

if (typeof window !== 'undefined') window.TwoTowerModel = TwoTowerModel;
export { TwoTowerModel };
