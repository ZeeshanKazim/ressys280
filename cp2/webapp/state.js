// state.js
// Holds global app state: loaded data + current UI selection.

export function createInitialState(data) {
  const { flightsByQuery, baselineScoreById, routeGraph, reviewStats } = data;

  // Attach baseline scores to flights if missing
  Object.values(flightsByQuery).forEach((list) => {
    list.forEach((flight) => {
      if (
        (flight.score === undefined || flight.score === null) &&
        baselineScoreById[flight.Id] !== undefined
      ) {
        flight.score = baselineScoreById[flight.Id];
      }
    });
  });

  // Collect distinct origins and destinations for the dropdowns
  const originsSet = new Set();
  const destsSet = new Set();

  Object.values(flightsByQuery).forEach((list) => {
    list.forEach((f) => {
      if (f.origin) originsSet.add(f.origin);
      if (f.dest) destsSet.add(f.dest);
    });
  });

  const allOrigins = Array.from(originsSet).sort();
  const allDests = Array.from(destsSet).sort();

  return {
    flightsByQuery,
    baselineScoreById,
    routeGraph,
    reviewStats,
    allOrigins,
    allDests,

    currentOrigin: "",
    currentDest: "",
    constraints: {
      maxPrice: null,
      maxStops: 3,
      avoidRedEye: false,
      preferredAlliance: "",
    },
  };
}
