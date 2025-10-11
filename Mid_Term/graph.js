/* graph.js – simple co-visitation graph + Personalized PageRank */

function buildItemGraph(user2items){
  // Undirected weighted graph over items: weight = co-occurrence count across users
  const G = new Map(); // itemId -> Map(itemId -> w)
  function bump(a,b){
    if (a===b) return;
    if (!G.has(a)) G.set(a,new Map());
    const m = G.get(a); m.set(b, (m.get(b)||0)+1);
  }
  for (const [,arr] of user2items){
    const uniq = Array.from(new Set(arr.map(x=>x.i)));
    for (let i=0;i<uniq.length;i++){
      for (let j=i+1;j<uniq.length;j++){
        bump(uniq[i], uniq[j]); bump(uniq[j], uniq[i]);
      }
    }
  }
  // normalize to transition probabilities
  for (const [node, nbr] of G){
    let s=0; for (const w of nbr.values()) s+=w;
    if (s>0){ for (const k of nbr.keys()) nbr.set(k, nbr.get(k)/s); }
  }
  return G;
}

let _graphCache = null;

function personalizedPageRankForUser(u, user2items, item2users, {alpha=0.15,iters=20}={}){
  if (!_graphCache) _graphCache = buildItemGraph(user2items);
  const G = _graphCache;

  const seen = new Set((user2items.get(u)||[]).map(x=>x.i));
  if (!seen.size) return new Map();

  const nodes = Array.from(G.keys());
  const idx = new Map(nodes.map((id,i)=>[id,i]));

  const n = nodes.length;
  const p = new Float32Array(n);
  for (const id of seen){ if (idx.has(id)) p[idx.get(id)] = 1/seen.size; }

  let r = new Float32Array(n); r.set(p);

  // r_{t+1} = (1-α) * P^T r_t + α p
  for (let t=0;t<iters;t++){
    const next = new Float32Array(n);
    for (let a=0;a<n;a++){
      const mass = (1-alpha) * r[a];
      const nbr = G.get(nodes[a]); if (!nbr) continue;
      for (const [b,w] of nbr.entries()){
        const bi = idx.get(b);
        next[bi] += mass * w;
      }
    }
    for (let i=0;i<n;i++) next[i]+= alpha * p[i];
    r = next;
  }

  const out = new Map();
  for (let i=0;i<n;i++){ if (r[i]>0) out.set(nodes[i], r[i]); }
  return out;
}

window.personalizedPageRankForUser = personalizedPageRankForUser;
