/* graph.js — simple Personalized PageRank over user–item bipartite graph */
function personalizedPageRankForUser(uId, user2items, item2users, opts={}){
  const alpha = opts.alpha ?? 0.15, iters = opts.iters ?? 20;
  // Build neighbor lists on the fly
  const uNeighbors = (u)=> (user2items.get(u)||[]).map(x=>x.i);
  const iNeighbors = (i)=> Array.from(item2users.get(i)||[]);
  const tele = new Map(); // teleport set = items user has touched
  for(const it of uNeighbors(uId)) tele.set(it, 1);

  // initialize item scores uniform
  const items = new Set(); for(const it of item2users.keys()) items.add(it);
  const score = new Map([...items].map(i=>[i, 0]));

  // power iterations
  for(let t=0;t<iters;t++){
    const next = new Map([...items].map(i=>[i, 0]));
    // push mass from users to items and items to users
    const uMass = new Map(); // user mass from items
    for(const i of items){
      const uit = iNeighbors(i);
      const share = (score.get(i)||0) / Math.max(1, uit.length);
      for(const u of uit) uMass.set(u, (uMass.get(u)||0) + share);
    }
    for(const [u, mass] of uMass.entries()){
      const its = uNeighbors(u);
      const share = mass / Math.max(1, its.length);
      for(const i of its) next.set(i, next.get(i)+share);
    }
    // teleport
    for(const i of tele.keys()) next.set(i, next.get(i)*(1-alpha) + alpha);
    for(const i of items) if(!tele.has(i)) next.set(i, next.get(i)*(1-alpha));
    // normalize
    let s=0; for(const v of next.values()) s+=v; s = s||1;
    for(const i of items) next.set(i, next.get(i)/s);
    for(const i of items) score.set(i, next.get(i));
  }
  return score;
}
