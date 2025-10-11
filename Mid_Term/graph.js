/* graph.js
   Build a simple co-visitation item graph and run Personalized PageRank.
   Intended for light, in-browser re-rank of a candidate set.
*/

function buildCoVisGraph(interactions, itemIdToIdx, opts={}) {
  const alpha = opts.alpha ?? 0.75;        // decay for within-user pairs by distance
  const maxNeighbors = opts.maxNeighbors ?? 64;

  // neighbors[idx] = Map(neiIdx -> weight)
  const neighbors = new Map();

  // Collect per user
  const byUser = new Map();
  for (const r of interactions) {
    if (!byUser.has(r.user)) byUser.set(r.user, []);
    byUser.get(r.user).push(r.item);
  }
  // For each user, add pairwise co-views with decay
  for (const [u, items] of byUser) {
    const arr = items.slice(0, 200); // safety cap
    for (let i=0;i<arr.length;i++){
      const ai = itemIdToIdx.get(arr[i]); if (ai==null) continue;
      for (let j=i+1;j<arr.length;j++){
        const aj = itemIdToIdx.get(arr[j]); if (aj==null) continue;
        const w = Math.pow(alpha, j-i-1);
        if (!neighbors.has(ai)) neighbors.set(ai, new Map());
        if (!neighbors.has(aj)) neighbors.set(aj, new Map());
        neighbors.get(ai).set(aj, (neighbors.get(ai).get(aj)||0)+w);
        neighbors.get(aj).set(ai, (neighbors.get(aj).get(ai)||0)+w);
      }
    }
  }

  // Keep top-k neighbors
  for (const [i, map] of neighbors) {
    const top = [...map.entries()].sort((a,b)=>b[1]-a[1]).slice(0, maxNeighbors);
    neighbors.set(i, new Map(top));
  }
  return {neighbors};
}

// Personalized PageRank on neighbors graph. seeds: array of item indices.
function personalizedPageRank(graph, seeds, opts={}) {
  const n = Math.max(...graphNeighborsKeys(graph))+1;
  const d = opts.d ?? 0.85;
  const iters = opts.iters ?? 30;
  const seedProb = 1.0 / (seeds.length||1);
  const p0 = new Float32Array(n);
  for (const s of seeds) if (s<n) p0[s] = seedProb;

  const p = new Float32Array(n);
  const deg = new Float32Array(n);
  // compute degrees
  for (const [i,nei] of graph.neighbors) {
    let sum = 0; for (const [,w] of nei) sum += w;
    deg[i] = sum || 1;
  }

  // Power iteration: p = (1-d)*p0 + d*W^T * p
  const tmp = new Float32Array(n);
  for (let t=0;t<iters;t++){
    tmp.fill(0);
    for (const [i,nei] of graph.neighbors) {
      const mass = p[i]/deg[i];
      for (const [j,w] of nei) tmp[j] += w * mass;
    }
    for (let i=0;i<n;i++) p[i] = (1-d)*p0[i] + d*tmp[i];
  }
  return p;
}

function graphNeighborsKeys(g){ return [...g.neighbors.keys()]; }

if (typeof window !== 'undefined') {
  window.buildCoVisGraph = buildCoVisGraph;
  window.personalizedPageRank = personalizedPageRank;
}
export { buildCoVisGraph, personalizedPageRank };
