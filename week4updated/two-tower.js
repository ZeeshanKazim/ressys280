// two-tower.js
// Minimal Two-Tower retrieval model in TensorFlow.js.
//
// - User tower: user_id  -> embedding
// - Item tower: item_id  -> embedding
// - Scoring: dot product
// - Loss: in-batch softmax (default) or BPR pairwise

class TwoTowerModel {
    /**
     * @param {number} numUsers
     * @param {number} numItems
     * @param {number} embeddingDim
     * @param {Object} options { learningRate, lossType: 'softmax' | 'bpr' }
     */
    constructor(numUsers, numItems, embeddingDim, options = {}) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embeddingDim = embeddingDim;

        const { learningRate = 0.001, lossType = 'softmax' } = options;
        this.lossType = lossType;

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

        this.optimizer = tf.train.adam(learningRate);
    }

    // ---------------------------------------------------------------------
    // Towers
    // ---------------------------------------------------------------------

    userForward(userIndices) {
        return tf.gather(this.userEmbeddings, userIndices);
    }

    itemForward(itemIndices) {
        return tf.gather(this.itemEmbeddings, itemIndices);
    }

    score(userEmbeddings, itemEmbeddings) {
        return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), -1);
    }

    // ---------------------------------------------------------------------
    // Training step
    // ---------------------------------------------------------------------

    trainStep(userIndices, itemIndices) {
        if (!userIndices.length) return 0;

        const lossValue = tf.tidy(() => {
            const userTensor = tf.tensor1d(userIndices, 'int32');
            const itemTensor = tf.tensor1d(itemIndices, 'int32');

            const lossFn = () => {
                if (this.lossType === 'bpr') return this._bprLoss(userTensor, itemTensor);
                return this._softmaxLoss(userTensor, itemTensor);
            };

            const { value, grads } = this.optimizer.computeGradients(lossFn);
            this.optimizer.applyGradients(grads);

            const scalar = value.dataSync()[0];
            return scalar;
        });

        return lossValue;
    }

    // In-batch softmax loss
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

    // BPR loss (optional)
    _bprLoss(userTensor, itemTensor) {
        const batchSize = userTensor.shape[0];
        const posIndices = itemTensor.dataSync();

        const negIndicesArr = new Int32Array(batchSize);
        for (let i = 0; i < batchSize; i++) {
            let neg = Math.floor(Math.random() * this.numItems);
            if (neg === posIndices[i]) {
                neg = (neg + 1) % this.numItems;
            }
            negIndicesArr[i] = neg;
        }
        const negTensor = tf.tensor1d(negIndicesArr, 'int32');

        const userEmbs = this.userForward(userTensor);
        const posItemEmbs = this.itemForward(itemTensor);
        const negItemEmbs = this.itemForward(negTensor);

        const posScores = this.score(userEmbs, posItemEmbs);
        const negScores = this.score(userEmbs, negItemEmbs);

        const diff = tf.sub(posScores, negScores);
        const sigmoidDiff = tf.sigmoid(diff);
        const logSigmoid = tf.log(sigmoidDiff.add(1e-8));
        const loss = tf.neg(logSigmoid);
        return tf.mean(loss);
    }

    // ---------------------------------------------------------------------
    // Inference
    // ---------------------------------------------------------------------

    getUserEmbedding(userIndex) {
        return tf.tidy(() => {
            const idx = tf.tensor1d([userIndex], 'int32');
            const emb2d = this.userForward(idx); // [1, D]
            return emb2d.squeeze(); // [D]
        });
    }

    getScoresForAllItems(userEmbedding) {
        const scores = tf.tidy(() => {
            const u = userEmbedding.reshape([this.embeddingDim, 1]); // [D,1]
            const scoresTensor = this.itemEmbeddings.matMul(u).squeeze(); // [numItems]
            const arr = scoresTensor.dataSync();
            return arr;
        });
        return scores;
    }

    getItemEmbeddings() {
        return this.itemEmbeddings;
    }
}
