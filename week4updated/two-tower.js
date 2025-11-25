// two-tower.js
// Minimal Two-Tower retrieval model in TensorFlow.js.
//
// - User tower: user_id  -> embedding
// - Item tower: item_id  -> embedding
// - Scoring: dot product in a shared latent space
// - Loss: in-batch sampled softmax (default) or BPR pairwise ranking
//
// This file is used by app.js, which:
//   * Calls new TwoTowerModel(numUsers, numItems, embDim, options)
//   * Calls model.trainStep(userIdxArray, itemIdxArray) per batch
//   * Uses getUserEmbedding() + getScoresForAllItems() for inference
//   * Uses getItemEmbeddings() for PCA visualization
//
// Requirements: TensorFlow.js loaded globally as `tf` (see index.html).

class TwoTowerModel {
    /**
     * @param {number} numUsers      Number of distinct users
     * @param {number} numItems      Number of distinct items
     * @param {number} embeddingDim  Embedding dimensionality
     * @param {Object} options       { learningRate, lossType: 'softmax' | 'bpr' }
     */
    constructor(numUsers, numItems, embeddingDim, options = {}) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embeddingDim = embeddingDim;

        const { learningRate = 0.001, lossType = 'softmax' } = options;
        this.lossType = lossType; // 'softmax' (in-batch negatives) or 'bpr'

        // ---------------------------------------------------------------------
        // Embedding tables
        // ---------------------------------------------------------------------
        // Two separate embedding matrices (towers) that project users/items
        // into the same latent space. Relevance is computed via similarity
        // (dot product).
        this.userEmbeddings = tf.variable(
            tf.randomNormal([numUsers, embeddingDim], 0, 0.05),
            true,
            'user_embeddings'
        );

        this.itemEmbeddings = tf.variable(
            tf.randomNormal([numItems, embeddingDim], 0, 0.05),
            true,
            'item_embeddings'
        );

        // Adam optimizer is stable for embedding training
        this.optimizer = tf.train.adam(learningRate);
    }

    // -------------------------------------------------------------------------
    // Forward passes (towers)
    // -------------------------------------------------------------------------

    /**
     * User tower: lookup embeddings for user indices.
     * @param {tf.Tensor1D} userIndices int32 tensor of shape [B]
     * @returns {tf.Tensor2D} shape [B, D]
     */
    userForward(userIndices) {
        return tf.gather(this.userEmbeddings, userIndices);
    }

    /**
     * Item tower: lookup embeddings for item indices.
     * @param {tf.Tensor1D} itemIndices int32 tensor of shape [B]
     * @returns {tf.Tensor2D} shape [B, D]
     */
    itemForward(itemIndices) {
        return tf.gather(this.itemEmbeddings, itemIndices);
    }

    /**
     * Dot-product scoring between user and item embeddings.
     * @param {tf.Tensor2D} userEmbeddings [B, D]
     * @param {tf.Tensor2D} itemEmbeddings [B, D]
     * @returns {tf.Tensor1D} scores [B]
     */
    score(userEmbeddings, itemEmbeddings) {
        return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), -1);
    }

    // -------------------------------------------------------------------------
    // Training
    // -------------------------------------------------------------------------

    /**
     * Single optimization step on a batch of (userIdx, itemIdx) positive pairs.
     * Depending on this.lossType, uses:
     *  - in-batch sampled softmax: positives on the diagonal, all other
     *    items in the batch are negatives;
     *  - BPR pairwise loss: positive vs sampled negative item.
     *
     * @param {number[]} userIndices  Array of user indices
     * @param {number[]} itemIndices  Array of item indices
     * @returns {number} scalar loss value
     */
    trainStep(userIndices, itemIndices) {
        if (!userIndices.length) return 0;

        const lossValue = tf.tidy(() => {
            const userTensor = tf.tensor1d(userIndices, 'int32');
            const itemTensor = tf.tensor1d(itemIndices, 'int32');

            const lossFn = () => {
                if (this.lossType === 'bpr') {
                    return this._bprLoss(userTensor, itemTensor);
                }
                // default: in-batch softmax
                return this._softmaxLoss(userTensor, itemTensor);
            };

            const { value, grads } = this.optimizer.computeGradients(lossFn);
            this.optimizer.applyGradients(grads);

            const scalar = value.dataSync()[0]; // extract JS number
            return scalar;
        });

        return lossValue;
    }

    /**
     * In-batch sampled softmax loss.
     *
     * Given batch user embeddings U and positive items I+:
     *   logits = U @ I+^T  (shape [B, B])
     *   labels = one-hot of diagonal (each user i's positive item is i)
     * Softmax cross-entropy over each row encourages the positive pair
     * to score higher than all in-batch negatives.
     */
    _softmaxLoss(userTensor, itemTensor) {
        const userEmbs = this.userForward(userTensor); // [B, D]
        const itemEmbs = this.itemForward(itemTensor); // [B, D]

        const logits = tf.matMul(userEmbs, itemEmbs, false, true); // [B, B]
        const batchSize = userTensor.shape[0];

        const labels = tf.oneHot(
            tf.range(0, batchSize, 1, 'int32'),
            batchSize
        ); // [B, B]

        const loss = tf.losses.softmaxCrossEntropy(labels, logits);
        return loss;
    }

    /**
     * Bayesian Personalized Ranking (BPR) loss.
     *
     * For each (user, positiveItem) pair:
     *   1. Sample a random negative item
     *   2. Compute s_pos = score(u, i+), s_neg = score(u, i-)
     *   3. Loss = -log(sigmoid(s_pos - s_neg))
     *
     * This optimizes pairwise ranking: positives should rank above negatives.
     */
    _bprLoss(userTensor, itemTensor) {
        const batchSize = userTensor.shape[0];
        const posIndices = itemTensor.dataSync(); // TypedArray length B

        const negIndicesArr = new Int32Array(batchSize);
        for (let i = 0; i < batchSize; i++) {
            let neg = Math.floor(Math.random() * this.numItems);
            if (neg === posIndices[i]) {
                neg = (neg + 1) % this.numItems;
            }
            negIndicesArr[i] = neg;
        }
        const negTensor = tf.tensor1d(negIndicesArr, 'int32');

        const userEmbs = this.userForward(userTensor);      // [B, D]
        const posItemEmbs = this.itemForward(itemTensor);   // [B, D]
        const negItemEmbs = this.itemForward(negTensor);    // [B, D]

        const posScores = this.score(userEmbs, posItemEmbs); // [B]
        const negScores = this.score(userEmbs, negItemEmbs); // [B]

        const diff = tf.sub(posScores, negScores);           // [B]
        const sigmoidDiff = tf.sigmoid(diff);
        const logSigmoid = tf.log(sigmoidDiff.add(1e-8));
        const loss = tf.neg(logSigmoid);                     // [B]
        return tf.mean(loss);                                // scalar
    }

    // -------------------------------------------------------------------------
    // Inference helpers
    // -------------------------------------------------------------------------

    /**
     * Get a single user embedding as a 1D tensor.
     * Used by app.js for scoring against all items.
     *
     * @param {number} userIndex
     * @returns {tf.Tensor1D} shape [D]
     */
    getUserEmbedding(userIndex) {
        return tf.tidy(() => {
            const idx = tf.tensor1d([userIndex], 'int32'); // [1]
            const emb2d = this.userForward(idx);           // [1, D]
            const emb1d = emb2d.squeeze();                 // [D]
            return emb1d;
        });
    }

    /**
     * Compute scores for a given user embedding against ALL items.
     * This is effectively itemEmbeddings @ userEmbedding.
     *
     * @param {tf.Tensor1D} userEmbedding shape [D]
     * @returns {Float32Array} scores of length numItems
     */
    getScoresForAllItems(userEmbedding) {
        const scores = tf.tidy(() => {
            const u = userEmbedding.reshape([this.embeddingDim, 1]); // [D, 1]
            const scoresTensor = this.itemEmbeddings.matMul(u).squeeze(); // [numItems]
            const arr = scoresTensor.dataSync(); // Float32Array (copied)
            return arr;
        });
        return scores;
    }

    /**
     * Access the full item embedding matrix for visualization (PCA).
     * Caller can use arraySync() on the returned tensor.
     *
     * @returns {tf.Tensor2D} shape [numItems, embeddingDim]
     */
    getItemEmbeddings() {
        return this.itemEmbeddings;
    }
}
