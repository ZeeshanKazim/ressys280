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
 *  - baseline score (normalized within this route)
 *  - route popularity / degrees
 *  - review rating
 *  - alliance preference
 *
 * The weights are tuned so that:
 *  - Baseline still matters a lot
 *  - But alliance + reviews can change ordering in realistic cases
 */
function computeHybridScores(flights, constraints) {
  if (!flights.length) return [];

  const userAlliance = constraints.preferAlliance;

  // Pre-compute min / max for normalization
  let maxRoutePop = 0;
  let maxDeg = 0;
  let minDuration = Infinity;
  let minPrice = Infinity;
  let minBase = Infinity;
  let maxBase = -Infinity;

  flights.forEach((f) => {
    const pop = Number(f.routePopularity || 0);
    if (pop > maxRoutePop) maxRoutePop = pop;

    const degSum = Number(f.departureDegree || 0) + Number(f.arrivalDegree || 0);
    if (degSum > maxDeg) maxDeg = degSum;

    if (f.durationMinutes < minDuration) minDuration = f.durationMinutes;
    if (f.price < minPrice) minPrice = f.price;

    const base = Number(f.baselineScore || 0);
    if (base < minBase) minBase = base;
    if (base > maxBase) maxBase = base;
  });

  const baseRange = maxBase - minBase;

  return flights.map((f) => {
    const baselineScore = Number(f.baselineScore || 0);

    // Baseline normalized to [0,1]
    let baseNorm = 0.5;
    if (baseRange > 1e-6) {
      baseNorm = (baselineScore - minBase) / baseRange;
    }

    // Normalized route popularity 0–1
    const pop = Number(f.routePopularity || 0);
    const popNorm = maxRoutePop > 0 ? pop / maxRoutePop : 0;

    // Normalized degree 0–1
    const degSum = Number(f.departureDegree || 0) + Number(f.arrivalDegree || 0);
    const degNorm = maxDeg > 0 ? degSum / maxDeg : 0;

    // Review rating: roughly 1..5, center at 3 → [-1,1], then scaled
    let ratingNorm = 0;
    if (f.reviewAvgRating != null) {
      ratingNorm = (Number(f.reviewAvgRating) - 3) / 2; // [-1, +1]
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

    // Hybrid score: weights chosen to give visible impact
    // on alliance + reviews without totally ignoring baseline.
    const hybridScore =
      1.0 * baseNorm + // main backbone
      0.35 * popNorm +
      0.25 * degNorm +
      0.6 * ratingNorm +
      (userAlliance && userAlliance !== "No preference" ? 1.0 * allianceBonus : 0);

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
