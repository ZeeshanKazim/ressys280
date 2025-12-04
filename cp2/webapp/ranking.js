// ranking.js
// Baseline + enhanced ranking using precomputed scores and JSON features.

// Simple carrier → alliance mapping for demo purposes.
const CARRIER_ALLIANCE = {
  SU: "SkyTeam",
  S7: "Oneworld",
  U6: "Star",
  UT: "None",
  DP: "None",
};

function getMinMax(arr) {
  if (!arr.length) return [0, 0];
  let min = arr[0];
  let max = arr[0];
  for (let i = 1; i < arr.length; i++) {
    const v = arr[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return [min, max];
}

function normalizeValue(v, min, max) {
  if (max === min) return 0.5;
  return (v - min) / (max - min);
}

function getCandidatesForRoute(state, origin, dest) {
  const candidates = [];
  Object.values(state.flightsByQuery).forEach((list) => {
    list.forEach((f) => {
      if (f.origin === origin && f.dest === dest) {
        candidates.push(f);
      }
    });
  });
  return candidates;
}

function applyHardConstraints(candidates, constraints) {
  let filtered = candidates;

  // Max price
  if (constraints.maxPrice !== null && !Number.isNaN(constraints.maxPrice)) {
    filtered = filtered.filter((f) => {
      const p = Number(f.totalPrice);
      return Number.isFinite(p) ? p <= constraints.maxPrice : true;
    });
  }

  // Max stops (3 = "any")
  if (constraints.maxStops !== 3) {
    filtered = filtered.filter((f) => {
      const s = Number(f.num_stops || 0);
      return s <= constraints.maxStops;
    });
  }

  // Avoid red-eye
  if (constraints.avoidRedEye) {
    const nonRed = filtered.filter((f) => Number(f.red_eye || 0) === 0);
    if (nonRed.length > 0) {
      filtered = nonRed;
    }
  }

  return filtered.length > 0 ? filtered : candidates;
}

function computeContextStats(flights) {
  let minPrice = Infinity;
  let minDuration = Infinity;

  flights.forEach((f) => {
    const price = Number(f.totalPrice);
    const dur = Number(f.duration_minutes);
    if (Number.isFinite(price) && price < minPrice) minPrice = price;
    if (Number.isFinite(dur) && dur < minDuration) minDuration = dur;
  });

  if (!Number.isFinite(minPrice)) minPrice = null;
  if (!Number.isFinite(minDuration)) minDuration = null;

  return { minPrice, minDuration };
}

function buildBaselineBadges(flight, index, ctx) {
  const badges = [];
  const price = Number(flight.totalPrice);
  const dur = Number(flight.duration_minutes);
  const stops = Number(flight.num_stops || 0);
  const red = Number(flight.red_eye || 0);

  if (index === 0) badges.push("Top pick");
  if (ctx.minPrice !== null && price === ctx.minPrice) badges.push("Cheapest");
  if (ctx.minDuration !== null && dur === ctx.minDuration)
    badges.push("Fastest");
  if (stops === 0) badges.push("Non-stop");
  if (red === 1) badges.push("Red-eye");

  return badges;
}

function buildBaselineExplanation(flight, constraints) {
  const parts = [];

  const price = Number(flight.totalPrice);
  if (
    constraints.maxPrice !== null &&
    Number.isFinite(price) &&
    price <= constraints.maxPrice
  ) {
    parts.push(`within your max price (${price.toFixed(0)})`);
  }

  const stops = Number(flight.num_stops || 0);
  if (constraints.maxStops !== 3 && stops <= constraints.maxStops) {
    parts.push(`${stops} stop(s) within your limit`);
  }

  if (constraints.avoidRedEye && Number(flight.red_eye || 0) === 0) {
    parts.push("avoids red-eye flights");
  }

  if (parts.length === 0) {
    return "Ranked by baseline model score using price, duration, and stops.";
  }

  return (
    "Ranked by baseline model score and because it " + parts.join(", ") + "."
  );
}

export function rankBaseline({ state, origin, dest, constraints }) {
  const candidates = getCandidatesForRoute(state, origin, dest);
  if (!candidates.length) {
    return [];
  }

  const filtered = applyHardConstraints(candidates, constraints);
  const ctx = computeContextStats(filtered);

  const sorted = [...filtered].sort((a, b) => {
    const sa = Number(a.score || 0);
    const sb = Number(b.score || 0);
    return sb - sa;
  });

  const topK = sorted.slice(0, 10);

  return topK.map((flight, index) => ({
    flight,
    explanation: buildBaselineExplanation(flight, constraints),
    badges: buildBaselineBadges(flight, index, ctx),
  }));
}

// ---------- Enhanced ranking (RouteGraph-RAG) ----------

function computeGraphRaw(flight, state) {
  const key = `${flight.origin}-${flight.dest}`;
  const stats = state.routeGraph[key];
  if (!stats) return 0;

  const pop = Number(stats.route_popularity || 0);
  const depDeg = Number(stats.departure_degree || 0);
  const arrDeg = Number(stats.arrival_degree || 0);

  return pop + 0.1 * depDeg + 0.1 * arrDeg;
}

function computeReviewRaw(flight, state) {
  const carrier = flight.carrier;
  if (!carrier) return 0;
  const stats = state.reviewStats[carrier];
  if (!stats) return 0;

  const rating = Number(stats.avg_rating || 0);
  const numReviews = Number(stats.num_reviews || 0);

  const ratingNorm = rating / 5;
  const volumeBoost = Math.log1p(numReviews);

  return ratingNorm + 0.1 * volumeBoost;
}

function buildEnhancedBadges(flight, index, ctx, scoreParts) {
  const badges = [];
  const price = Number(flight.totalPrice);
  const dur = Number(flight.duration_minutes);
  const stops = Number(flight.num_stops || 0);

  if (index === 0) badges.push("Top hybrid pick");
  if (ctx.minPrice !== null && price === ctx.minPrice) badges.push("Cheapest");
  if (ctx.minDuration !== null && dur === ctx.minDuration)
    badges.push("Fastest");
  if (stops === 0) badges.push("Non-stop");

  if (scoreParts.graphNorm > 0.6) badges.push("Graph-favoured");
  if (scoreParts.reviewNorm > 0.6) badges.push("Better reviews");

  return badges;
}

function buildEnhancedExplanation(flight, constraints, scoreParts) {
  const bits = [];

  if (scoreParts.baseNorm >= 0.6) {
    bits.push("strong baseline score");
  }
  if (scoreParts.graphNorm >= 0.6) {
    bits.push("popular or well-connected route");
  }
  if (scoreParts.reviewNorm >= 0.6) {
    bits.push("airline has comparatively better reviews");
  }

  const alliancePref = constraints.preferredAlliance;
  const carrierAlliance = CARRIER_ALLIANCE[flight.carrier] || "None";
  if (alliancePref && carrierAlliance === alliancePref) {
    bits.push(`matches your ${alliancePref} alliance preference`);
  }

  if (!bits.length) {
    return "Ranked by a hybrid of baseline score, route-graph features, and review stats.";
  }

  return (
    "Recommended because of " +
    bits.join(", ") +
    ", combined in the hybrid score."
  );
}

export function rankEnhanced({ state, origin, dest, constraints }) {
  const candidates = getCandidatesForRoute(state, origin, dest);
  if (!candidates.length) {
    return [];
  }

  const filtered = applyHardConstraints(candidates, constraints);
  const ctx = computeContextStats(filtered);

  const baseRaw = filtered.map((f) => Number(f.score || 0));
  const graphRaw = filtered.map((f) => computeGraphRaw(f, state));
  const reviewRaw = filtered.map((f) => computeReviewRaw(f, state));

  const [baseMin, baseMax] = getMinMax(baseRaw);
  const [graphMin, graphMax] = getMinMax(graphRaw);
  const [reviewMin, reviewMax] = getMinMax(reviewRaw);

  const scored = filtered.map((flight, idx) => {
    const baseNorm = normalizeValue(baseRaw[idx], baseMin, baseMax);
    const graphNorm = normalizeValue(graphRaw[idx], graphMin, graphMax);
    const reviewNorm = normalizeValue(reviewRaw[idx], reviewMin, reviewMax);

    let allianceBonus = 0;
    const pref = constraints.preferredAlliance;
    const carrierAlliance = CARRIER_ALLIANCE[flight.carrier] || "None";
    if (pref && carrierAlliance === pref) {
      allianceBonus = 0.08;
    }

    const hybridScore =
      0.6 * baseNorm + 0.25 * graphNorm + 0.15 * reviewNorm + allianceBonus;

    return {
      flight,
      hybridScore,
      baseNorm,
      graphNorm,
      reviewNorm,
    };
  });

  scored.sort((a, b) => b.hybridScore - a.hybridScore);
  const topK = scored.slice(0, 10);

  return topK.map((entry, index) => {
    const scoreParts = {
      baseNorm: entry.baseNorm,
      graphNorm: entry.graphNorm,
      reviewNorm: entry.reviewNorm,
    };
    return {
      flight: entry.flight,
      explanation: buildEnhancedExplanation(
        entry.flight,
        constraints,
        scoreParts
      ),
      badges: buildEnhancedBadges(entry.flight, index, ctx, scoreParts),
    };
  });
}
