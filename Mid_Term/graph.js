/* graph.js
   Personalized PageRank on a bipartite user–item graph.

   We seed with the picked user and their interacted items,
   then run a few power iterations with restart α.

   Return: Map<itemId -> score> (items only)
*/

function personalizedPageRankForUser(userId, user2items, item2users, opts = {}) {
  const alpha = opts.alpha ?? 0.15;
  const iters = opts.iters ?? 20;

  // Collect node sets
  const userIds = new Set([userId]);
  const itemIds = new Set();
  const seeds = new Set();

  const hist = user2items.get(userId) || [];
  for (const { i } of hist) { itemIds.add(i); seeds.add(i); }

  // Neighborhood expansion (1 hop) to avoid empty graph
  for (const iid of itemIds) {
    const uSet = item2users.get(iid);
    if (uSet) for (const u of uSet) userIds.add(u);
  }
  for (const u of userIds) {
    const arr = user2items.get(u) || [];
    for (const { i } of arr) itemIds.add(i);
  }

  // Indexing
  const users = Array.from(userIds);
  const items = Array.from(itemIds);
  const U = users.length, I = items.length;
  const uIndex = new Map(users.map((u, k) => [u, k]));
  const iIndex = new Map(items.map((i, k) => [i, k]));

  // Transition: random walk from user -> items and item -> users (normalized)
  const outU = new Array(U).fill(0).map(() => []);
  const outI = new Array(I).fill(0).map(() => []);

  for (let ui = 0; ui < U; ui++) {
    const uId = users[ui];
    const arr = user2items.get(uId) || [];
    const deg = arr.length || 1;
    for (const { i } of arr) {
      const ii = iIndex.get(i);
      if (ii != null) outU[ui].push([ii, 1 / deg]);
    }
  }
  for (let ii = 0; ii < I; ii++) {
    const itId = items[ii];
    const set = item2users.get(itId) || new Set();
    const deg = set.size || 1;
    for (const u of set) {
      const ui = uIndex.get(u);
      if (ui != null) outI[ii].push([ui, 1 / deg]);
    }
  }

  // Initial vector: mass on seed items (or the user if no history)
  let pU = new Float32Array(U).fill(0);
  let pI = new Float32Array(I).fill(0);
  if (seeds.size) {
    const val = 1 / seeds.size;
    for (const iid of seeds) {
      const ii = iIndex.get(iid);
      if (ii != null) pI[ii] = val;
    }
  } else {
    const ui = uIndex.get(userId);
    if (ui != null) pU[ui] = 1;
  }

  for (let t = 0; t < iters; t++) {
    const nextU = new Float32Array(U).fill(0);
    const nextI = new Float32Array(I).fill(0);

    // user -> item
    for (let ui = 0; ui < U; ui++) {
      const mass = (1 - alpha) * pU[ui];
      if (!mass) continue;
      for (const [ii, w] of outU[ui]) nextI[ii] += mass * w;
    }
    // item -> user
    for (let ii = 0; ii < I; ii++) {
      const mass = (1 - alpha) * pI[ii];
      if (!mass) continue;
      for (const [ui, w] of outI[ii]) nextU[ui] += mass * w;
    }

    // restart
    for (let ui = 0; ui < U; ui++) nextU[ui] += alpha * (ui === uIndex.get(userId) ? 1 : 0);
    if (seeds.size) {
      const val = alpha / seeds.size;
      for (const iid of seeds) {
        const ii = iIndex.get(iid);
        if (ii != null) nextI[ii] += val;
      }
    }

    pU = nextU; pI = nextI;
  }

  // Return only item scores
  const out = new Map();
  for (let ii = 0; ii < I; ii++) out.set(items[ii], pI[ii]);
  return out;
}

window.personalizedPageRankForUser = personalizedPageRankForUser;
