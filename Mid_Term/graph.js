/* graph.js
   Lightweight Personalized PageRank over a user–item bipartite graph.
   We run a two‑step random walk (item→user→item) within a local neighborhood.
   Returns: Map<itemId, score> for re‑ranking.
*/

function personalizedPageRankForUser(userId, user2items, item2users, opts = {}) {
  const alpha = opts.alpha ?? 0.15; // restart prob
  const iters = opts.iters ?? 20;
  const maxUsers = opts.maxUsers ?? 4000;   // keep local neighborhood bounded
  const maxItems = opts.maxItems ?? 6000;

  // Seed = user's own items
  const seedItems = new Set((user2items.get(userId) || []).map(x => x.i));
  if (!seedItems.size) return new Map();

  // Neighborhood: users who touched seedItems, and the items they touched
  const nbrUsers = new Set();
  for (const i of seedItems) {
    const uSet = item2users.get(i);
    if (!uSet) continue;
    for (const u of uSet) {
      if (nbrUsers.size < maxUsers) nbrUsers.add(u);
    }
  }
  const candItems = new Set([...seedItems]);
  for (const u of nbrUsers) {
    const arr = user2items.get(u) || [];
    for (const row of arr) {
      if (candItems.size < maxItems) candItems.add(row.i);
    }
  }

  // Degree caches
  const degItem = new Map();
  const degUser = new Map();
  for (const i of candItems) degItem.set(i, (item2users.get(i) || new Set()).size || 1);
  for (const u of nbrUsers) degUser.set(u, (user2items.get(u) || []).length || 1);

  // Scores
  let score = new Map();
  let teleport = new Map();
  for (const i of candItems) score.set(i, seedItems.has(i) ? 1 / seedItems.size : 0);
  teleport = new Map(score); // same support

  // Iterate: item→user→item with restart
  for (let t = 0; t < iters; t++) {
    // Push to users
    const userScore = new Map();
    for (const [i, s] of score.entries()) {
      const users = item2users.get(i) || new Set();
      const di = degItem.get(i) || 1;
      for (const u of users) {
        if (!nbrUsers.has(u)) continue;
        const val = (userScore.get(u) || 0) + s / di;
        userScore.set(u, val);
      }
    }
    // Normalize by each user degree
    for (const [u, v] of userScore.entries()) {
      userScore.set(u, v / (degUser.get(u) || 1));
    }

    // Pull back to items
    const next = new Map();
    for (const [u, su] of userScore.entries()) {
      const arr = user2items.get(u) || [];
      for (const row of arr) {
        if (!candItems.has(row.i)) continue;
        const di = degItem.get(row.i) || 1;
        next.set(row.i, (next.get(row.i) || 0) + su / di);
      }
    }

    // Restart
    for (const i of candItems) {
      const walk = (1 - alpha) * (next.get(i) || 0);
      const restart = alpha * (teleport.get(i) || 0);
      next.set(i, walk + restart);
    }
    score = next;
  }
  return score;
}

window.personalizedPageRankForUser = personalizedPageRankForUser;
