/* graph.js — very small Personalized PageRank on bipartite user–item graph.
   personalizedPageRankForUser(uId, user2items, item2users, {alpha=0.15, iters=20})
   Returns Map(itemId -> score).
*/
function personalizedPageRankForUser(uId, user2items, item2users, opts={}){
  const alpha = opts.alpha ?? 0.15;
  const iters = opts.iters ?? 20;

  // Build neighbors lazily
  const uHist = user2items.get(uId) || [];
  const items = new Set(uHist.map(x=>x.i));
  // candidate frontier: items ↔ users
  const allItems = new Set();
  for (const {i} of uHist){
    allItems.add(i);
    const neighUsers = item2users.get(i) || new Set();
    for (const v of neighUsers){
      const vs = user2items.get(v) || [];
      vs.forEach(r=>allItems.add(r.i));
    }
  }

  const pi = new Map(); // item score
  const tele = 1 / Math.max(1, allItems.size);
  // init
  allItems.forEach(i=>pi.set(i, tele));

  for (let t=0;t<iters;t++){
    const next = new Map();
    // distribute from items -> users -> items
    for (const i of allItems){
      const us = item2users.get(i) || new Set();
      const mass = (1-alpha) * (pi.get(i)||0);
      const shareU = us.size ? mass/us.size : 0;
      for (const u of us){
        const its = user2items.get(u) || [];
        const shareI = its.length ? shareU/its.length : 0;
        for (const e of its){
          next.set(e.i, (next.get(e.i)||0) + shareI);
        }
      }
    }
    // teleport
    for (const i of allItems){
      next.set(i, (next.get(i)||0) + alpha*tele);
    }
    pi.clear();
    next.forEach((v,k)=>pi.set(k,v));
  }
  return pi;
}

window.personalizedPageRankForUser = personalizedPageRankForUser;
