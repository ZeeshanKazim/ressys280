// cp2/webapp/ranking.js
// Core ranking logic:
//  - load flights for a route
//  - apply hard constraints (price, stops, red-eye)
//  - compute baseline and hybrid scores
//  - return two ranked lists for the UI.

import {
  getFlightsForRoute,
  getReviewForCarrier,
  getCarrierAlliance,
  getCarrierName
} from "./dataStore.js";

/**
 * Main entry point.
 *
 * @param {Object} constraints
 * {
 *   origin: "KZN",
 *   dest: "DME",
 *   maxPrice: number | null,
 *   maxStops: number | null,
 *   avoidRedEye: boolean,
 *   preferAlliance: "No preference" | "SkyTeam" | "Star Alliance" | "Oneworld" | "None"
 * }
 *
 * @returns {{
 *   baseline: Array<Object>,
 *   hybrid: Array<Object>,
 *   hardConstraintsSatisfied: boolean
 * }}
 */
export function getRecommendations(constraints) {
  const { origin, dest } = constraints;

  // 1) Get all flights for this route
  const allFlights = getFlightsForRoute(origin, dest);

  if (!allFlights || allFlights.length === 0) {
    return {
      baseline: [],
      hybrid: [],
      hardConstraintsSatisfied: true
    };
  }

  // 2) Apply hard constraints: price, stops, red-eye
  const hardFiltered = allFlights.filter((f) =>
    passesHardConstraints(f, constraints)
  );

  let usedFlights = hardFiltered;
  let hardConstraintsSatisfied = true;

  if (hardFiltered.length === 0) {
    // No flights satisfy all hard constraints — fall back to all flights
    usedFlights = allFlights;
    hardConstraintsSatisfied = false;
  }

  // 3) Enrich flights with review + alliance info
  const enriched = usedFlights.map((f) => enrichFlight(f));

  // 4) Compute hybrid scores using route graph + reviews + alliance preference
  const withScores = computeHybridScores(enriched, constraints);

  // 5) Rank: baseline and hybrid
  const baselineRanked = [...withScores].sort(
    (a, b) => b.baselineScore - a.baselineScore
  );

  const hybridRanked = [...withScores].sort(
    (a, b) => b.hybridScore - a.hybridScore
  );

  return {
    baseline: baselineRanked,
    hybrid: hybridRanked,
    hardConstraintsSatisfied
  };
}

/**
 * Hard constraints: price, stops, red-eye.
 */
function passesHardConstraints(flight, constraints) {
  const { maxPrice, maxStops, avoidRedEye } = constraints;

  // Max price
  if (maxPrice != null && flight.price > maxPrice) {
    return false;
  }

  // Max stops
  if (maxStops != null && flight.stops > maxStops) {
    return false;
  }

  // Red-eye
  if (avoidRedEye && flight.redEye) {
    return false;
  }

  return true;
}

/**
 * Attach airline name, alliance, review stats.
 */
function enrichFlight(flight) {
  const carrier = flight.carrier;

  const reviewInfo = getReviewForCarrier(carrier);
  let airlineName = carrier;
  let alliance = "None";
  let numReviews = null;
  let avgRating = null;

  if (reviewInfo) {
    airlineName = reviewInfo.airlineName;
    alliance = reviewInfo.alliance;
    numReviews = reviewInfo.numReviews;
    avgRating = reviewInfo.avgRating;
  } else {
    // Fallback if we have metadata but no review entry
    airlineName = getCarrierName(carrier);
    alliance = getCarrierAlliance(carrier) || "None";
  }

  return {
    ...flight,
    airlineName,
    alliance,
    reviewNumReviews: numReviews,
    reviewAvgRating: avgRating
  };
}

/**
 * Compute hybrid scores for a list of flights, taking into account:
 *  - baseline score
 *  - route popularity / degrees
 *  - review rating
 *  - alliance preference
 */
function computeHybridScores(flights, constraints) {
  if (!flights.length) return [];

  // Route-graph normalization
  let maxRoutePop = 0;
  let maxDeg = 0;
  let minDuration = Infinity;
  let minPrice = Infinity;

  flights.forEach((f) => {
    const pop = Number(f.routePopularity || 0);
    if (pop > maxRoutePop) maxRoutePop = pop;

    const degSum = Number(f.departureDegree || 0) + Number(f.arrivalDegree || 0);
    if (degSum > maxDeg) maxDeg = degSum;

    if (f.durationMinutes < minDuration) minDuration = f.durationMinutes;
    if (f.price < minPrice) minPrice = f.price;
  });

  const userAlliance = constraints.preferAlliance;

  return flights.map((f) => {
    const baselineScore = Number(f.baselineScore || 0);

    // Normalized route popularity 0–1
    const pop = Number(f.routePopularity || 0);
    const popNorm = maxRoutePop > 0 ? pop / maxRoutePop : 0;

    // Normalized degree 0–1
    const degSum = Number(f.departureDegree || 0) + Number(f.arrivalDegree || 0);
    const degNorm = maxDeg > 0 ? degSum / maxDeg : 0;

    // Review rating: center around 3, scale to roughly [-1, +1]
    let ratingNorm = 0;
    if (f.reviewAvgRating != null) {
      // Assuming rating is roughly 1..5
      ratingNorm = (Number(f.reviewAvgRating) - 3) / 2;
    }

    // Alliance preference bonus: 0 or 1
    let allianceBonus = 0;
    if (
      userAlliance &&
      userAlliance !== "No preference" &&
      f.alliance === userAlliance
    ) {
      allianceBonus = 1;
    }

    // Combine into a hybrid score.
    // Weights are heuristic but small enough to keep baseline dominant.
    const hybridScore =
      baselineScore +
      0.3 * popNorm +
      0.2 * degNorm +
      0.4 * ratingNorm +
      0.5 * allianceBonus;

    // Flags for UI (fastest / cheapest)
    const isFastest = f.durationMinutes === minDuration;
    const isCheapest = f.price === minPrice;

    return {
      ...f,
      baselineScore,
      hybridScore,
      isFastest,
      isCheapest
    };
  });
}
