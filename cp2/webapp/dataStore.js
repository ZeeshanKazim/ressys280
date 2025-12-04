// dataStore.js
// Fetches static JSON artifacts produced in Colab.
// With your repo tree, JSONs live in ../data/ relative to webapp/index.html.

async function loadJson(path) {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`Failed to load ${path}: ${res.status} ${res.statusText}`);
  }
  return await res.json();
}

export async function loadAllData() {
  // Paths are relative to webapp/index.html
  const flightsByQuery = await loadJson("../data/flights_small.json");
  const baselineScoresArray = await loadJson("../data/baseline_scores.json");
  const routeGraph = await loadJson("../data/route_graph_features.json");
  const reviewStats = await loadJson("../data/review_stats.json");

  // Build a quick lookup map: flight Id -> baseline score
  const baselineScoreById = {};
  baselineScoresArray.forEach((row) => {
    if (row && row.Id !== undefined && row.score !== undefined) {
      baselineScoreById[row.Id] = row.score;
    }
  });

  return {
    flightsByQuery,      // { ranker_id: [flight, ...], ... }
    baselineScoreById,   // { Id: score, ... }
    routeGraph,          // { "SVO-LHR": { route_popularity, ... }, ... }
    reviewStats,         // { airlineKey: { num_reviews, avg_rating }, ... }
  };
}
