// cp2/webapp/dataStore.js
// Central place to load and normalize all precomputed data for the demo.

// In-memory stores
let flightsByRouteKey = {};   // "ORIGIN-DEST" -> [flight, ...]
let allOrigins = [];
let allDestinations = [];
let allRoutes = [];

let reviewStats = {};         // airlineName -> { num_reviews, avg_rating }
let routeGraph = {};          // "ORIGIN-DEST" -> { route_popularity, departure_degree, arrival_degree }
let carrierMeta = {};         // carrierCode -> { name, alliance }

/**
 * Call this once on page load before using any getters.
 * It loads:
 *  - flights_small.json
 *  - review_stats.json
 *  - route_graph_features.json
 *  - carrier_metadata.json
 */
export async function loadAllData() {
  const [
    flightsRes,
    reviewsRes,
    graphRes,
    carrierRes
  ] = await Promise.all([
    fetch("../data/flights_small.json"),
    fetch("../data/review_stats.json"),
    fetch("../data/route_graph_features.json"),
    fetch("../data/carrier_metadata.json")
  ]);

  const flightsJson = await flightsRes.json();
  reviewStats = await reviewsRes.json();
  routeGraph = await graphRes.json();
  carrierMeta = await carrierRes.json();

  // Build flightsByRouteKey from flights_small.json
  flightsByRouteKey = {};

  Object.values(flightsJson).forEach((flightList) => {
    flightList.forEach((raw) => {
      const flight = normalizeFlight(raw);
      const routeKey = flight.routeKey;

      if (!flightsByRouteKey[routeKey]) {
        flightsByRouteKey[routeKey] = [];
      }
      flightsByRouteKey[routeKey].push(flight);
    });
  });

  // Derive sets of origins / destinations from the routes
  const originSet = new Set();
  const destSet = new Set();

  Object.keys(flightsByRouteKey).forEach((routeKey) => {
    const [origin, dest] = routeKey.split("-");
    originSet.add(origin);
    destSet.add(dest);
  });

  allOrigins = Array.from(originSet).sort();
  allDestinations = Array.from(destSet).sort();
  allRoutes = Object.keys(flightsByRouteKey).sort();
}

/**
 * Normalize the raw flight record from flights_small.json into a
 * consistent shape used by the app.
 */
function normalizeFlight(raw) {
  const origin = raw.origin;
  const dest = raw.dest;
  const routeKey = `${origin}-${dest}`;

  return {
    id: raw.Id,
    rankerId: raw.ranker_id,
    carrier: raw.carrier,
    origin,
    dest,
    routeKey,
    departureIso: raw.legs0_departureAt,
    price: Number(raw.totalPrice),
    durationMinutes: Number(raw.duration_minutes),
    stops: Number(raw.num_stops ?? 0),
    redEye: Boolean(raw.red_eye),
    logPrice: Number(raw.log_totalPrice),
    baselineScore: Number(raw.score),
    // Graph features (if missing here, we can fill them from routeGraph later)
    routePopularity: raw.route_popularity ?? null,
    departureDegree: raw.departure_degree ?? null,
    arrivalDegree: raw.arrival_degree ?? null
  };
}

// ---------- Public getters ----------

export function getAllOrigins() {
  return allOrigins.slice();
}

export function getAllDestinations() {
  return allDestinations.slice();
}

export function getAllRoutes() {
  return allRoutes.slice();
}

/**
 * Return all flights for a given (origin, dest) as normalized objects.
 * Graph features are filled from route_graph_features.json if needed.
 */
export function getFlightsForRoute(origin, dest) {
  const routeKey = `${origin}-${dest}`;
  const flights = flightsByRouteKey[routeKey] || [];
  const graphInfo = routeGraph[routeKey];

  if (graphInfo) {
    return flights.map((f) => ({
      ...f,
      routePopularity: f.routePopularity ?? graphInfo.route_popularity,
      departureDegree: f.departureDegree ?? graphInfo.departure_degree,
      arrivalDegree: f.arrivalDegree ?? graphInfo.arrival_degree
    }));
  }

  // No graph info, just return a copy of the list
  return flights.map((f) => ({ ...f }));
}

/**
 * Get combined review info for a carrier:
 *  - airlineName (from carrier_metadata.json)
 *  - alliance
 *  - num_reviews, avg_rating (from review_stats.json)
 */
export function getReviewForCarrier(carrierCode) {
  const meta = carrierMeta[carrierCode];
  if (!meta) return null;

  const stats = reviewStats[meta.name];
  if (!stats) return null;

  return {
    airlineName: meta.name,
    alliance: meta.alliance,
    numReviews: stats.num_reviews,
    avgRating: stats.avg_rating
  };
}

/**
 * Get the alliance for a carrier, or "None" if unknown.
 */
export function getCarrierAlliance(carrierCode) {
  const meta = carrierMeta[carrierCode];
  return meta ? meta.alliance : "None";
}

/**
 * Get the human-readable airline name for a carrier code.
 */
export function getCarrierName(carrierCode) {
  const meta = carrierMeta[carrierCode];
  return meta ? meta.name : carrierCode;
}
