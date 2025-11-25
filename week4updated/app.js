// app.js
// MovieLens 100K Two-Tower + RAG-style demo (pure browser, TensorFlow.js)
//
// Features:
//  - Load MovieLens 100K (u.data, u.item)
//  - Build user/item indexers
//  - Train a Two-Tower model with in-batch softmax or BPR loss
//  - Live loss chart on canvas
//  - PCA projection of item embeddings
//  - Test mode: show historical top-10 vs top-10 recommended
//  - Offline eval for sampled user: Precision@5, Recall@5, NDCG@5
//  - Text query mode (RAG-style): vector-search over titles, then re-rank with Two-Tower
//
// NOTE: index.html must include:
//   - Buttons: #loadData, #train, #test
//   - Canvases: #lossChart, #embeddingChart
//   - Divs: #status, #results
//   - (Optional) elements for query mode:
//       input#queryText, input#queryUserId, button#querySearch, div#queryResults

class MovieLensApp {
    constructor() {
        // Raw parsed data
        this.interactions = []; // full interactions used for splitting
        this.items = new Map(); // itemId -> { title, year }

        // Train / eval split
        this.trainInteractions = []; // subset of interactions used for training
        this.testItemsByUser = new Map(); // userId -> [itemId,...] hold-out positives for metrics

        // ID <-> index mappings
        this.userMap = new Map(); // raw userId -> 0..numUsers-1
        this.itemMap = new Map(); // raw itemId -> 0..numItems-1
        this.reverseUserMap = new Map(); // index -> raw userId
        this.reverseItemMap = new Map(); // index -> raw itemId

        // For displaying user history
        this.userTopRated = new Map(); // userId -> [interactions sorted by rating+time]
        this.qualifiedUsers = []; // userIds with >= minHistory interactions

        // Evaluation / query helpers
        this.titleTokensByItem = new Map(); // itemId -> [token, ...] for simple text retrieval

        // Model & training state
        this.model = null;
        this.lossHistory = [];
        this.isTraining = false;

        // Embedding visualization state
        this.embeddingPoints = []; // [{x,y,title,year}, ...]

        // Hyperparameters / config
        this.config = {
            maxInteractions: 80000,
            embeddingDim: 32,
            batchSize: 512,
            epochs: 20,
            learningRate: 0.001,
            lossType: 'softmax', // 'softmax' or 'bpr'
            minRatingsForQualifiedUser: 20,
            numTestPerUser: 3,
            minRatingForPositive: 4.0, // used when selecting test positives
            metricsK: 5 // K for Precision@K, Recall@K, NDCG@K
        };

        this.initializeUI();
    }

    // ---------------------------------------------------------------------
    // UI wiring
    // ---------------------------------------------------------------------

    initializeUI() {
        const loadBtn = document.getElementById('loadData');
        const trainBtn = document.getElementById('train');
        const testBtn = document.getElementById('test');

        if (loadBtn) {
            loadBtn.addEventListener('click', () => this.loadData());
        }
        if (trainBtn) {
            trainBtn.addEventListener('click', () => this.train());
        }
        if (testBtn) {
            testBtn.addEventListener('click', () => this.test());
        }

        // Optional: RAG-style query recommend
        const queryBtn = document.getElementById('querySearch');
        if (queryBtn) {
            queryBtn.addEventListener('click', () => this.handleQueryRecommend());
        }

        const embeddingCanvas = document.getElementById('embeddingChart');
        if (embeddingCanvas) {
            embeddingCanvas.addEventListener('mousemove', (event) =>
                this.handleEmbeddingHover(event)
            );
        }

        this.updateStatus('Click "Load Data" to start.');
    }

    updateStatus(message) {
        const statusDiv = document.getElementById('status');
        if (statusDiv) {
            statusDiv.textContent = message;
        }
    }

    // ---------------------------------------------------------------------
    // Data loading
    // ---------------------------------------------------------------------

    async loadData() {
        if (this.isTraining) return;

        const trainBtn = document.getElementById('train');
        const testBtn = document.getElementById('test');
        if (trainBtn) trainBtn.disabled = true;
        if (testBtn) testBtn.disabled = true;

        this.updateStatus('Loading MovieLens 100K from ./data ...');

        // Reset state in case of re-load
        this.interactions = [];
        this.items.clear();
        this.trainInteractions = [];
        this.testItemsByUser.clear();
        this.userMap.clear();
        this.itemMap.clear();
        this.reverseUserMap.clear();
        this.reverseItemMap.clear();
        this.userTopRated.clear();
        this.qualifiedUsers = [];
        this.lossHistory = [];
        this.embeddingPoints = [];
        this.titleTokensByItem.clear();
        this.model = null;
        this.updateLossChart();

        const embeddingCanvas = document.getElementById('embeddingChart');
        if (embeddingCanvas) {
            const ctx = embeddingCanvas.getContext('2d');
            if (ctx) {
                ctx.clearRect(0, 0, embeddingCanvas.width, embeddingCanvas.height);
            }
        }
        const resultsDiv = document.getElementById('results');
        if (resultsDiv) {
            resultsDiv.innerHTML = '';
        }
        const queryResultsDiv = document.getElementById('queryResults');
        if (queryResultsDiv) {
            queryResultsDiv.innerHTML = '';
        }

        try {
            const [interResp, itemResp] = await Promise.all([
                fetch('data/u.data'),
                fetch('data/u.item')
            ]);

            if (!interResp.ok) {
                throw new Error(`Failed to load data/u.data (status ${interResp.status})`);
            }
            if (!itemResp.ok) {
                throw new Error(`Failed to load data/u.item (status ${itemResp.status})`);
            }

            // Parse interactions
            const interText = await interResp.text();
            const interLines = interText.trim().split(/\r?\n/);

            this.interactions = interLines
                .slice(0, this.config.maxInteractions)
                .map((line) => {
                    const parts = line.split('\t');
                    if (parts.length < 4) return null;
                    const userId = parseInt(parts[0], 10);
                    const itemId = parseInt(parts[1], 10);
                    const rating = parseFloat(parts[2]);
                    const ts = parseInt(parts[3], 10);
                    if (
                        Number.isNaN(userId) ||
                        Number.isNaN(itemId) ||
                        Number.isNaN(rating) ||
                        Number.isNaN(ts)
                    ) {
                        return null;
                    }
                    return { userId, itemId, rating, timestamp: ts };
                })
                .filter((x) => x !== null);

            // Parse items
            const itemText = await itemResp.text();
            const itemLines = itemText.trim().split(/\r?\n/);

            itemLines.forEach((line) => {
                const parts = line.split('|');
                if (parts.length < 2) return;
                const itemId = parseInt(parts[0], 10);
                if (Number.isNaN(itemId)) return;
                const rawTitle = parts[1];

                const yearMatch = rawTitle.match(/\((\d{4})\)\s*$/);
                const year = yearMatch ? parseInt(yearMatch[1], 10) : null;
                const cleanTitle = rawTitle.replace(/\(\d{4}\)\s*$/, '').trim();

                this.items.set(itemId, { title: cleanTitle, year: year || null });
                this.titleTokensByItem.set(itemId, this.tokenizeTitle(cleanTitle));
            });

            // Build indexers and train/test split
            this.createMappingsAndSplit();
            this.findQualifiedUsers();

            const summary =
                `Loaded ${this.interactions.length.toLocaleString()} interactions, ` +
                `${this.userMap.size} users, ${this.itemMap.size} items.\n` +
                `${this.qualifiedUsers.length} users have at least ${this.config.minRatingsForQualifiedUser} ratings.`;
            this.updateStatus(summary);

            if (trainBtn) trainBtn.disabled = false;
        } catch (err) {
            console.error(err);
            this.updateStatus(
                `Error loading data: ${err.message}\n` +
                    'If you opened index.html from disk, start a local HTTP server or use GitHub Pages.'
            );
        }
    }

    tokenizeTitle(title) {
        // Very lightweight tokenization for text retrieval over titles
        const text = title.toLowerCase();
        // Remove simple punctuation
        const cleaned = text.replace(/[^a-z0-9\s]/g, ' ');
        const tokens = cleaned
            .split(/\s+/)
            .map((t) => t.trim())
            .filter((t) => t.length > 1); // ignore single-letter tokens
        return tokens;
    }

    createMappingsAndSplit() {
        // ---- Create dense indices ----
        const userSet = new Set();
        const itemSet = new Set();
        this.interactions.forEach((i) => {
            userSet.add(i.userId);
            itemSet.add(i.itemId);
        });

        let idx = 0;
        userSet.forEach((userId) => {
            this.userMap.set(userId, idx);
            this.reverseUserMap.set(idx, userId);
            idx += 1;
        });

        idx = 0;
        itemSet.forEach((itemId) => {
            this.itemMap.set(itemId, idx);
            this.reverseItemMap.set(idx, itemId);
            idx += 1;
        });

        // ---- Per-user grouping ----
        const byUser = new Map(); // userId -> [interactions]
        this.interactions.forEach((inter) => {
            const u = inter.userId;
            if (!byUser.has(u)) byUser.set(u, []);
            byUser.get(u).push(inter);
        });

        // ---- Train / test split + top-rated ----
        this.trainInteractions = [];
        this.testItemsByUser.clear();
        this.userTopRated.clear();

        const minRating = this.config.minRatingForPositive;
        const numTestPerUser = this.config.numTestPerUser;

        byUser.forEach((list, userId) => {
            // Chronological sort for splitting
            const byTime = list.slice().sort((a, b) => a.timestamp - b.timestamp);
            const testCandidates = byTime.filter((i) => i.rating >= minRating);
            const testSlice =
                testCandidates.length > numTestPerUser
                    ? testCandidates.slice(-numTestPerUser)
                    : testCandidates.slice();

            const testIds = new Set(testSlice.map((i) => i.itemId));
            this.testItemsByUser.set(userId, Array.from(testIds));

            // Train interactions = all minus selected test ones
            byTime.forEach((i) => {
                if (!testIds.has(i.itemId)) {
                    this.trainInteractions.push(i);
                }
            });

            // For UI historical view: sort by rating desc, then recency desc
            const topSorted = list
                .slice()
                .sort((a, b) => (b.rating !== a.rating ? b.rating - a.rating : b.timestamp - a.timestamp));
            this.userTopRated.set(userId, topSorted);
        });
    }

    findQualifiedUsers() {
        this.qualifiedUsers = [];
        const minRatings = this.config.minRatingsForQualifiedUser;

        this.userTopRated.forEach((list, userId) => {
            if (list.length >= minRatings) {
                this.qualifiedUsers.push(userId);
            }
        });
    }

    // ---------------------------------------------------------------------
    // Training
    // ---------------------------------------------------------------------

    async train() {
        if (this.isTraining) return;
        if (!this.trainInteractions.length) {
            this.updateStatus('No training interactions. Load data first.');
            return;
        }

        const trainBtn = document.getElementById('train');
        const testBtn = document.getElementById('test');
        if (trainBtn) trainBtn.disabled = true;
        if (testBtn) testBtn.disabled = true;

        this.isTraining = true;
        this.lossHistory = [];
        this.updateLossChart();
        this.updateStatus('Initializing Two-Tower model & preparing training tensors...');

        this.model = new TwoTowerModel(
            this.userMap.size,
            this.itemMap.size,
            this.config.embeddingDim,
            {
                learningRate: this.config.learningRate,
                lossType: this.config.lossType
            }
        );

        const userIndices = this.trainInteractions.map((i) => this.userMap.get(i.userId));
        const itemIndices = this.trainInteractions.map((i) => this.itemMap.get(i.itemId));
        const numBatches = Math.ceil(userIndices.length / this.config.batchSize);

        this.updateStatus(
            `Training with ${userIndices.length.toLocaleString()} interactions, ` +
                `${numBatches} batches/epoch, ${this.config.epochs} epochs...`
        );

        for (let epoch = 0; epoch < this.config.epochs; epoch++) {
            let epochLoss = 0;

            for (let batch = 0; batch < numBatches; batch++) {
                const start = batch * this.config.batchSize;
                const end = Math.min(start + this.config.batchSize, userIndices.length);
                const batchUsers = userIndices.slice(start, end);
                const batchItems = itemIndices.slice(start, end);

                const loss = this.model.trainStep(batchUsers, batchItems);
                epochLoss += loss;
                this.lossHistory.push(loss);
                this.updateLossChart();

                if (batch % 10 === 0 || batch === numBatches - 1) {
                    this.updateStatus(
                        `Epoch ${epoch + 1}/${this.config.epochs} – ` +
                            `Batch ${batch + 1}/${numBatches}, Loss: ${loss.toFixed(4)}`
                    );
                }

                // yield to browser so UI remains responsive
                // eslint-disable-next-line no-await-in-loop
                await new Promise((resolve) => setTimeout(resolve, 0));
            }

            epochLoss /= numBatches;
            this.updateStatus(
                `Epoch ${epoch + 1}/${this.config.epochs} finished. Mean loss: ${epochLoss.toFixed(4)}`
            );
        }

        this.isTraining = false;
        if (trainBtn) trainBtn.disabled = false;
        if (testBtn) testBtn.disabled = false;

        this.updateStatus('Training completed ✅ – computing embedding PCA projection...');
        await this.visualizeEmbeddings();
        this.updateStatus(
            'Training completed ✅ – click "Test" to see recommendations or run a text query.'
        );
    }

    updateLossChart() {
        const canvas = document.getElementById('lossChart');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!this.lossHistory.length) {
            ctx.fillStyle = '#9ca3af';
            ctx.font = '13px Arial';
            ctx.fillText('Loss curve will appear here during training.', 16, canvas.height / 2);
            return;
        }

        const maxLoss = Math.max(...this.lossHistory);
        const minLoss = Math.min(...this.lossHistory);
        const range = maxLoss - minLoss || 1;

        const margin = 30;
        const width = canvas.width - margin * 2;
        const height = canvas.height - margin * 2;

        ctx.save();
        ctx.translate(margin, margin);

        // axes
        ctx.strokeStyle = '#e5e7eb';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(0, height);
        ctx.lineTo(width, height);
        ctx.stroke();

        // line
        ctx.beginPath();
        this.lossHistory.forEach((loss, idx) => {
            const x = (idx / Math.max(1, this.lossHistory.length - 1)) * width;
            const y = height - ((loss - minLoss) / range) * height;
            if (idx === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = '#2563eb';
        ctx.lineWidth = 2;
        ctx.stroke();

        // labels
        ctx.fillStyle = '#6b7280';
        ctx.font = '11px Arial';
        ctx.fillText(`min: ${minLoss.toFixed(4)}`, 4, 12);
        ctx.fillText(`max: ${maxLoss.toFixed(4)}`, 4, 26);
        ctx.fillText(`batches: ${this.lossHistory.length}`, width - 120, height + 18);

        ctx.restore();
    }

    // ---------------------------------------------------------------------
    // Embedding visualization (PCA on items)
    // ---------------------------------------------------------------------

    async visualizeEmbeddings() {
        if (!this.model) return;
        const canvas = document.getElementById('embeddingChart');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        this.embeddingPoints = [];

        try {
            const embTensor = this.model.getItemEmbeddings(); // [numItems, dim]
            const allEmb = embTensor.arraySync();
            const totalItems = allEmb.length;
            if (!totalItems) return;

            const sampleSize = Math.min(500, totalItems);
            const sampleIdx = [];
            for (let i = 0; i < sampleSize; i++) {
                sampleIdx.push(Math.floor((i * totalItems) / sampleSize));
            }
            const sampleEmb = sampleIdx.map((i) => allEmb[i]);

            const projected = this.computePCA(sampleEmb, 2); // [[x,y],...]

            const xs = projected.map((p) => p[0]);
            const ys = projected.map((p) => p[1]);
            const xMin = Math.min(...xs);
            const xMax = Math.max(...xs);
            const yMin = Math.min(...ys);
            const yMax = Math.max(...ys);
            const xRange = xMax - xMin || 1;
            const yRange = yMax - yMin || 1;

            const margin = 40;
            const width = canvas.width - margin * 2;
            const height = canvas.height - margin * 2;

            ctx.save();
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.fillStyle = '#111827';
            ctx.font = '14px Arial';
            ctx.fillText('Item Embeddings Projection (PCA)', 12, 20);
            ctx.fillStyle = '#6b7280';
            ctx.font = '11px Arial';
            ctx.fillText(
                `Sampled ${sampleSize} / ${totalItems} movies – hover to see titles`,
                12,
                36
            );

            ctx.translate(margin, margin);
            ctx.strokeStyle = '#e5e7eb';
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(0, height);
            ctx.lineTo(width, height);
            ctx.stroke();

            const points = [];
            ctx.fillStyle = 'rgba(37, 99, 235, 0.78)';

            for (let i = 0; i < sampleSize; i++) {
                const nx = (projected[i][0] - xMin) / xRange;
                const ny = (projected[i][1] - yMin) / yRange;
                const x = nx * width;
                const y = height - ny * height;

                const itemIndex = sampleIdx[i];
                const itemId = this.reverseItemMap.get(itemIndex);
                const meta = this.items.get(itemId) || { title: `Item ${itemId}`, year: null };

                ctx.beginPath();
                ctx.arc(x, y, 3, 0, Math.PI * 2);
                ctx.fill();

                points.push({
                    x: x + margin,
                    y: y + margin,
                    title: meta.title,
                    year: meta.year
                });
            }

            this.embeddingPoints = points;
            ctx.restore();
        } catch (err) {
            console.error(err);
            this.updateStatus(`Error during PCA visualization: ${err.message}`);
        }
    }

    computePCA(embeddings, dimensions) {
        const n = embeddings.length;
        if (!n || !embeddings[0]) return [];
        const dim = embeddings[0].length;

        const mean = new Array(dim).fill(0);
        embeddings.forEach((emb) => {
            for (let i = 0; i < dim; i++) mean[i] += emb[i];
        });
        for (let i = 0; i < dim; i++) mean[i] /= n;

        const centered = embeddings.map((emb) => emb.map((v, i) => v - mean[i]));

        const cov = Array.from({ length: dim }, () => new Array(dim).fill(0));
        centered.forEach((emb) => {
            for (let i = 0; i < dim; i++) {
                const vi = emb[i];
                for (let j = 0; j < dim; j++) {
                    cov[i][j] += vi * emb[j];
                }
            }
        });
        for (let i = 0; i < dim; i++) {
            for (let j = 0; j < dim; j++) {
                cov[i][j] /= n;
            }
        }

        const components = [];

        for (let d = 0; d < dimensions; d++) {
            let v = new Array(dim).fill(1 / Math.sqrt(dim));
            for (let iter = 0; iter < 10; iter++) {
                const newV = new Array(dim).fill(0);
                for (let i = 0; i < dim; i++) {
                    let s = 0;
                    for (let j = 0; j < dim; j++) {
                        s += cov[i][j] * v[j];
                    }
                    newV[i] = s;
                }
                let norm = 0;
                for (let i = 0; i < dim; i++) norm += newV[i] * newV[i];
                norm = Math.sqrt(norm) || 1;
                for (let i = 0; i < dim; i++) v[i] = newV[i] / norm;
            }
            components.push(v);

            // deflate covariance
            for (let i = 0; i < dim; i++) {
                for (let j = 0; j < dim; j++) {
                    cov[i][j] -= v[i] * v[j];
                }
            }
        }

        const projected = centered.map((emb) =>
            components.map((comp) => {
                let s = 0;
                for (let i = 0; i < dim; i++) s += emb[i] * comp[i];
                return s;
            })
        );

        return projected;
    }

    handleEmbeddingHover(event) {
        if (!this.embeddingPoints.length) return;
        const canvas = event.currentTarget;
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const radius = 8;
        let closest = null;
        let bestDist2 = radius * radius;

        this.embeddingPoints.forEach((p) => {
            const dx = p.x - x;
            const dy = p.y - y;
            const dist2 = dx * dx + dy * dy;
            if (dist2 <= bestDist2) {
                bestDist2 = dist2;
                closest = p;
            }
        });

        if (closest) {
            const label = closest.year ? `${closest.title} (${closest.year})` : closest.title;
            canvas.title = label;
        } else {
            canvas.title = '';
        }
    }

    // ---------------------------------------------------------------------
    // Testing: historical vs recommended + metrics
    // ---------------------------------------------------------------------

    async test() {
        if (!this.model) {
            this.updateStatus('Model not trained yet – click "Train" first.');
            return;
        }
        if (!this.qualifiedUsers.length) {
            this.updateStatus(
                'No qualified users with enough history. Check data load / split configuration.'
            );
            return;
        }

        this.updateStatus(
            'Sampling a user with ≥ ' +
                this.config.minRatingsForQualifiedUser +
                ' ratings and generating recommendations...'
        );

        try {
            const idx = Math.floor(Math.random() * this.qualifiedUsers.length);
            const userId = this.qualifiedUsers[idx];
            const userInteractions = this.userTopRated.get(userId) || [];
            const userIndex = this.userMap.get(userId);

            const userEmb = this.model.getUserEmbedding(userIndex);
            const scores = this.model.getScoresForAllItems(userEmb); // Float32Array

            // UI recommendations: exclude items user has already rated (history)
            const ratedItemIds = new Set(userInteractions.map((i) => i.itemId));
            const candidateScores = [];
            for (let itemIndex = 0; itemIndex < scores.length; itemIndex++) {
                const itemId = this.reverseItemMap.get(itemIndex);
                if (itemId == null) continue;
                if (ratedItemIds.has(itemId)) continue;
                candidateScores.push({
                    itemId,
                    itemIndex,
                    score: scores[itemIndex]
                });
            }
            candidateScores.sort((a, b) => b.score - a.score);
            const topRecommendations = candidateScores.slice(0, 10);

            // Offline eval for this user (Precision@K, Recall@K, NDCG@K)
            const metrics = this.computeRankingMetricsForUser(userId, scores);

            this.displayResults(userId, userInteractions, topRecommendations, metrics);
            this.updateStatus('Recommendations and metrics generated – scroll down to inspect.');
        } catch (err) {
            console.error(err);
            this.updateStatus(`Error during test: ${err.message}`);
        }
    }

    computeRankingMetricsForUser(userId, scoresArray) {
        const K = this.config.metricsK;
        const testItems = this.testItemsByUser.get(userId) || [];
        if (!testItems.length) {
            return {
                precision: 0,
                recall: 0,
                ndcg: 0,
                hasEval: false
            };
        }
        const relevant = new Set(testItems);

        // Sort all items by predicted score
        const indices = [];
        for (let i = 0; i < scoresArray.length; i++) indices.push(i);
        indices.sort((a, b) => scoresArray[b] - scoresArray[a]);

        const topKIndices = indices.slice(0, K);
        const topKItemIds = topKIndices.map((idx) => this.reverseItemMap.get(idx));

        // Precision & Recall
        let hits = 0;
        const relFlags = [];
        topKItemIds.forEach((itemId) => {
            const isHit = itemId != null && relevant.has(itemId);
            if (isHit) hits += 1;
            relFlags.push(isHit ? 1 : 0);
        });

        const precision = hits / K;
        const recall = hits / relevant.size;

        // NDCG@K (binary relevance)
        let dcg = 0;
        for (let i = 0; i < relFlags.length; i++) {
            if (relFlags[i] === 1) {
                const denom = Math.log2(i + 2); // position i => log2(i+2)
                dcg += 1 / denom;
            }
        }

        const idealHits = Math.min(K, relevant.size);
        let idcg = 0;
        for (let i = 0; i < idealHits; i++) {
            idcg += 1 / Math.log2(i + 2);
        }
        const ndcg = idcg > 0 ? dcg / idcg : 0;

        return {
            precision,
            recall,
            ndcg,
            hasEval: true
        };
    }

    displayResults(userId, userInteractions, recommendations, metrics) {
        const resultsDiv = document.getElementById('results');
        if (!resultsDiv) return;

        const topRated = userInteractions.slice(0, 10);

        let html = `
            <h2>Recommendations for User ${userId}</h2>
            <p style="font-size:13px; color:#4b5563;">
                Offline evaluation on this user's held-out positives (K=${this.config.metricsK}):
        `;

        if (metrics && metrics.hasEval) {
            html += `
                Precision@${this.config.metricsK}: <strong>${metrics.precision.toFixed(
                3
            )}</strong>,
                Recall@${this.config.metricsK}: <strong>${metrics.recall.toFixed(3)}</strong>,
                NDCG@${this.config.metricsK}: <strong>${metrics.ndcg.toFixed(3)}</strong>
            `;
        } else {
            html += 'Not enough held-out interactions to compute metrics.';
        }
        html += '</p>';

        html += `
            <div class="side-by-side">
                <div>
                    <h3>Top 10 Rated Movies (Historical)</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Movie</th>
                                <th>Rating</th>
                                <th>Year</th>
                            </tr>
                        </thead>
                        <tbody>
        `;

        topRated.forEach((inter, idx) => {
            const meta = this.items.get(inter.itemId) || {
                title: `Item ${inter.itemId}`,
                year: null
            };
            html += `
                <tr>
                    <td>${idx + 1}</td>
                    <td>${meta.title}</td>
                    <td>${inter.rating.toFixed(1)}</td>
                    <td>${meta.year != null ? meta.year : 'N/A'}</td>
                </tr>
            `;
        });

        html += `
                        </tbody>
                    </table>
                </div>
                <div>
                    <h3>Top 10 Recommended Movies (Two-Tower)</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Movie</th>
                                <th>Score</th>
                                <th>Year</th>
                            </tr>
                        </thead>
                        <tbody>
        `;

        recommendations.forEach((rec, idx) => {
            const meta = this.items.get(rec.itemId) || {
                title: `Item ${rec.itemId}`,
                year: null
            };
            html += `
                <tr>
                    <td>${idx + 1}</td>
                    <td>${meta.title}</td>
                    <td>${rec.score.toFixed(4)}</td>
                    <td>${meta.year != null ? meta.year : 'N/A'}</td>
                </tr>
            `;
        });

        html += `
                        </tbody>
                    </table>
                </div>
            </div>
        `;

        resultsDiv.innerHTML = html;
    }

    // ---------------------------------------------------------------------
    // Simple RAG-style query: text -> title similarity -> Two-Tower re-rank
    // ---------------------------------------------------------------------

    handleQueryRecommend() {
        if (!this.model) {
            this.updateStatus('Train the model before running text query recommendations.');
            return;
        }

        const queryInput = document.getElementById('queryText');
        const userIdInput = document.getElementById('queryUserId');
        if (!queryInput) {
            this.updateStatus(
                'Query input (#queryText) not found. Add it to index.html to use text queries.'
            );
            return;
        }

        const rawQuery = queryInput.value.trim();
        if (!rawQuery) {
            this.updateStatus('Enter a natural-language query (e.g., "dark sci-fi with AI").');
            return;
        }

        let userId = null;
        if (userIdInput && userIdInput.value.trim() !== '') {
            const parsed = parseInt(userIdInput.value.trim(), 10);
            if (!Number.isNaN(parsed) && this.userMap.has(parsed)) {
                userId = parsed;
            }
        }

        // If user id not specified or invalid, fall back to random qualified user
        if (userId == null) {
            if (!this.qualifiedUsers.length) {
                this.updateStatus(
                    'No qualified users available for query re-ranking. Check data loading.'
                );
                return;
            }
            const idx = Math.floor(Math.random() * this.qualifiedUsers.length);
            userId = this.qualifiedUsers[idx];
        }

        this.queryRecommend(rawQuery, userId);
    }

    queryRecommend(queryText, userId) {
        const userIndex = this.userMap.get(userId);
        if (userIndex == null) {
            this.updateStatus(`User ${userId} not found in mappings.`);
            return;
        }

        const queryTokens = this.tokenizeTitle(queryText); // reuse title tokenizer
        if (!queryTokens.length) {
            this.updateStatus('Query is too short – please describe the movie you want.');
            return;
        }

        // Stage 1: simple text-based retrieval over titles (vector search proxy)
        const textScores = []; // {itemId, textScore}
        const querySet = new Set(queryTokens);

        this.items.forEach((meta, itemId) => {
            const tokens = this.titleTokensByItem.get(itemId) || [];
            if (!tokens.length) return;
            const titleSet = new Set(tokens);
            let overlap = 0;
            querySet.forEach((t) => {
                if (titleSet.has(t)) overlap += 1;
            });
            if (overlap === 0) return;
            const norm = Math.sqrt(querySet.size * titleSet.size) || 1;
            const score = overlap / norm; // cosine-like
            textScores.push({ itemId, textScore: score });
        });

        if (!textScores.length) {
            this.updateStatus(
                'No items matched the query tokens. Try a simpler description or different keywords.'
            );
            return;
        }

        textScores.sort((a, b) => b.textScore - a.textScore);
        const candidateCount = Math.min(100, textScores.length);
        const candidates = textScores.slice(0, candidateCount);

        // Stage 2: Two-Tower ranking using user embedding
        const userEmb = this.model.getUserEmbedding(userIndex);
        const allScoresArr = this.model.getScoresForAllItems(userEmb);

        // Combine text and collaborative signal
        const alpha = 0.4; // weight for text similarity
        const beta = 0.6; // weight for two-tower score
        const combined = candidates.map((c) => {
            const itemIndex = this.itemMap.get(c.itemId);
            const modelScore =
                itemIndex != null ? allScoresArr[itemIndex] : 0;
            return {
                itemId: c.itemId,
                textScore: c.textScore,
                modelScore,
                combinedScore: alpha * c.textScore + beta * modelScore
            };
        });

        combined.sort((a, b) => b.combinedScore - a.combinedScore);
        const topK = combined.slice(0, 10);

        this.displayQueryResults(queryText, userId, topK);
        this.updateStatus(
            'Text query recommendations ready – top movies ranked by text + Two-Tower signals.'
        );
    }

    displayQueryResults(queryText, userId, results) {
        const container = document.getElementById('queryResults');
        if (!container) return;

        let html = `
            <h2>Query-based Recommendations</h2>
            <p style="font-size:13px; color:#4b5563;">
                Query: <strong>${this.escapeHtml(queryText)}</strong><br/>
                Re-ranked for user <strong>${userId}</strong> (Two-Tower).
            </p>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Movie</th>
                        <th>Year</th>
                        <th>Text Score</th>
                        <th>Two-Tower Score</th>
                        <th>Combined</th>
                    </tr>
                </thead>
                <tbody>
        `;

        results.forEach((r, idx) => {
            const meta = this.items.get(r.itemId) || {
                title: `Item ${r.itemId}`,
                year: null
            };
            html += `
                <tr>
                    <td>${idx + 1}</td>
                    <td>${this.escapeHtml(meta.title)}</td>
                    <td>${meta.year != null ? meta.year : 'N/A'}</td>
                    <td>${r.textScore.toFixed(3)}</td>
                    <td>${r.modelScore.toFixed(4)}</td>
                    <td>${r.combinedScore.toFixed(4)}</td>
                </tr>
            `;
        });

        html += `
                </tbody>
            </table>
        `;

        container.innerHTML = html;
    }

    escapeHtml(str) {
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
    }
}

// Initialize app
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MovieLensApp();
    window.app = app; // for debugging in console
});
