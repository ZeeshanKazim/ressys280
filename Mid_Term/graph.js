/* graph.js
   Tiny co-vis graph and Personalized PageRank re-rank.
   Graph nodes = recipe IDs, edges connect items co-rated by a user.
*/
const Graph = (() => {
  "use strict";

  const adj = new Map(); // itemId -> Map(itemId -> weight)

  function addEdge(a,b,w=1){
    if (a===b) return;
    if (!adj.has(a)) adj.set(a, new Map());
    if (!adj.has(b)) adj.set(b, new Map());
    adj.get(a).set(b, (adj.get(a).get(b)||0) + w);
    adj.get(b).set(a, (adj.get(b).get(a)||0) + w);
  }

  function buildCoVis(interactions, {maxItems}={}){
    adj.clear();
    // group by user
    const map = new Map();
    for (const x of interactions){
      if (!map.has(x.u)) map.set(x.u, []);
      map.get(x.u).push(x.i);
    }
    for (const [u, items] of map){
      const uniq = Array.from(new Set(items));
      for (let i=0;i<uniq.length;i++){
        for (let j=i+1;j<uniq.length;j++){
          addEdge(uniq[i], uniq[j], 1);
        }
      }
    }
  }

  // Personalized PageRank given a user’s seen items boosts neighbors
  function rerankWithPPR(userId, rankedPairs, userToItems, alpha=0.15, iters=20){
    const seen = userToItems.get(userId) || new Map();
    if (!seen.size) return rankedPairs;

    // teleport vector over seen items
    const teleport = new Map(); for (const i of seen.keys()) teleport.set(i, 1/seen.size);

    // scores init
    const r = new Map(); // item -> score
    for (const [i,_] of teleport) r.set(i, 1/teleport.size);

    // iterate
    for (let t=0;t<iters;t++){
      const nr = new Map();
      // distribute
      for (const [i,ri] of r){
        const nbr = adj.get(i);
        if (!nbr || nbr.size===0) continue;
        const Z = [...nbr.values()].reduce((a,b)=>a+b,0);
        for (const [j,w] of nbr){
          nr.set(j, (nr.get(j)||0) + (1-alpha)*ri*(w/Z));
        }
      }
      // teleport back to seen
      for (const [i,pi] of teleport){
        nr.set(i, (nr.get(i)||0) + alpha*pi);
      }
      // normalize
      const sum = [...nr.values()].reduce((a,b)=>a+b,0) || 1;
      for (const k of nr.keys()) nr.set(k, nr.get(k)/sum);
      // swap
      r.clear(); for (const [k,v] of nr) r.set(k,v);
    }

    // re-rank pairs [itemId,score] by adding small boost λ * ppr
    const lambda = 0.15;
    const out = rankedPairs.map(([iid, sc]) => [iid, sc + lambda*(r.get(iid)||0)]);
    out.sort((a,b)=> b[1]-a[1]);
    return out;
  }

  return { buildCoVis, rerankWithPPR };
})();
