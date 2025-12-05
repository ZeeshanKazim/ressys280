// cp2/webapp/ui.js
// Rendering logic for the recommendations:
//  - shows a banner when hard constraints can't be fully satisfied
//  - renders baseline and hybrid result columns with explanatory text

let bannerEl;
let baselineContainer;
let hybridContainer;

export function initUI() {
  bannerEl = document.getElementById("constraints-banner");
  baselineContainer = document.getElementById("baseline-results");
  hybridContainer = document.getElementById("hybrid-results");

  if (!baselineContainer || !hybridContainer) {
    console.error(
      "[ui] Missing baseline-results or hybrid-results container. " +
        "Check IDs in index.html."
    );
  }
}

/**
 * Render both baseline and hybrid columns.
 *
 * @param {Object} opts
 *  - baseline: Array<flight>
 *  - hybrid: Array<flight>
 *  - hardConstraintsSatisfied: boolean
 *  - constraints: { origin, dest, maxPrice, maxStops, avoidRedEye, preferAlliance }
 */
export function renderRecommendations(opts) {
  const { baseline, hybrid, hardConstraintsSatisfied, constraints } = opts;

  if (!baselineContainer || !hybridContainer) {
    initUI(); // try again if not initialized
  }

  // Clear containers
  baselineContainer.innerHTML = "";
  hybridContainer.innerHTML = "";

  // Banner
  renderBanner(hardConstraintsSatisfied, constraints);

  // No results case
  if (!baseline.length && !hybrid.length) {
    const msg = document.createElement("div");
    msg.className = "no-results";
    msg.textContent = "No flights found for this route.";
    baselineContainer.appendChild(msg);
    hybridContainer.appendChild(msg.cloneNode(true));
    return;
  }

  // Render baseline
  baseline.forEach((flight, idx) => {
    const card = createFlightCard({
      flight,
      rank: idx + 1,
      columnType: "baseline",
      constraints
    });
    baselineContainer.appendChild(card);
  });

  // Render hybrid
  hybrid.forEach((flight, idx) => {
    const card = createFlightCard({
      flight,
      rank: idx + 1,
      columnType: "hybrid",
      constraints
    });
    hybridContainer.appendChild(card);
  });
}

function renderBanner(hardConstraintsSatisfied, constraints) {
  if (!bannerEl) {
    bannerEl = document.getElementById("constraints-banner");
    if (!bannerEl) return;
  }

  bannerEl.innerHTML = "";

  if (!hardConstraintsSatisfied) {
    const div = document.createElement("div");
    div.className = "banner banner-warning";
    div.textContent =
      "No flights matched all of your hard constraints. Showing closest available options instead.";
    bannerEl.appendChild(div);
  } else {
    // no banner if fully satisfied
  }
}

function createFlightCard({ flight, rank, columnType, constraints }) {
  const card = document.createElement("article");
  card.className = "flight-card";

  // Header: rank + route
  const header = document.createElement("div");
  header.className = "flight-card-header";

  const rankEl = document.createElement("div");
  rankEl.className = "flight-rank";
  rankEl.textContent = `#${rank}`;

  const titleEl = document.createElement("div");
  titleEl.className = "flight-title";
  titleEl.textContent = `${flight.carrier} ${flight.origin} → ${flight.dest}`;

  header.appendChild(rankEl);
  header.appendChild(titleEl);

  // Badges
  const badgesEl = document.createElement("div");
  badgesEl.className = "flight-badges";

  if (rank === 1) {
    const topBadge = document.createElement("span");
    topBadge.className = "badge badge-primary";
    topBadge.textContent =
      columnType === "baseline" ? "Top pick" : "Top hybrid pick";
    badgesEl.appendChild(topBadge);
  }

  if (flight.isFastest) {
    const b = document.createElement("span");
    b.className = "badge badge-neutral";
    b.textContent = "Fastest";
    badgesEl.appendChild(b);
  }

  if (flight.isCheapest) {
    const b = document.createElement("span");
    b.className = "badge badge-neutral";
    b.textContent = "Cheapest";
    badgesEl.appendChild(b);
  }

  if (flight.stops === 0) {
    const b = document.createElement("span");
    b.className = "badge badge-neutral";
    b.textContent = "Non-stop";
    badgesEl.appendChild(b);
  }

  if (flight.redEye) {
    const b = document.createElement("span");
    b.className = "badge badge-warning";
    b.textContent = "Red-eye";
    badgesEl.appendChild(b);
  }

  header.appendChild(badgesEl);

  // Body: details
  const body = document.createElement("div");
  body.className = "flight-card-body";

  const topRow = document.createElement("div");
  topRow.className = "flight-top-row";

  const priceEl = document.createElement("div");
  priceEl.className = "flight-price";
  priceEl.innerHTML = `<span class="icon">💰</span>${formatPrice(
    flight.price
  )}`;

  const durEl = document.createElement("div");
  durEl.className = "flight-duration";
  durEl.innerHTML = `<span class="icon">⏱</span>${formatDuration(
    flight.durationMinutes
  )}`;

  const stopsEl = document.createElement("div");
  stopsEl.className = "flight-stops";
  stopsEl.textContent = `Stops: ${flight.stops}`;

  topRow.appendChild(priceEl);
  topRow.appendChild(durEl);
  topRow.appendChild(stopsEl);

  const midRow = document.createElement("div");
  midRow.className = "flight-mid-row";

  const depEl = document.createElement("div");
  depEl.className = "flight-departure";
  depEl.innerHTML = `<span class="icon">🛫</span>${formatDateTime(
    flight.departureIso
  )}`;

  const airlineEl = document.createElement("div");
  airlineEl.className = "flight-airline";
  airlineEl.textContent = `${flight.airlineName} (${flight.alliance})`;

  midRow.appendChild(depEl);
  midRow.appendChild(airlineEl);

  // Explanation
  const expl = document.createElement("div");
  expl.className = "flight-explanation";
  expl.textContent = buildExplanation(flight, columnType, constraints);

  body.appendChild(topRow);
  body.appendChild(midRow);
  body.appendChild(expl);

  card.appendChild(header);
  card.appendChild(body);
  return card;
}

function formatPrice(p) {
  if (!Number.isFinite(p)) return String(p);
  return p.toFixed(0);
}

function formatDuration(minutes) {
  if (!Number.isFinite(minutes)) return "";
  const h = Math.floor(minutes / 60);
  const m = Math.round(minutes % 60);
  if (h === 0) return `${m}m`;
  if (m === 0) return `${h}h`;
  return `${h}h ${m}m`;
}

function formatDateTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  });
}

/**
 * Build explanation text for a flight card.
 */
function buildExplanation(flight, columnType, constraints) {
  if (columnType === "baseline") {
    return buildBaselineExplanation(flight, constraints);
  }
  return buildHybridExplanation(flight, constraints);
}

function buildBaselineExplanation(flight, constraints) {
  const parts = [];

  parts.push("Ranked by baseline model score using price, duration, and stops");

  if (constraints.maxPrice != null && flight.price <= constraints.maxPrice) {
    parts.push(`within your max price ${constraints.maxPrice}`);
  }

  if (constraints.maxStops != null && flight.stops <= constraints.maxStops) {
    parts.push(`within your max ${constraints.maxStops} stop(s)`);
  }

  if (constraints.avoidRedEye) {
    if (!flight.redEye) {
      parts.push("avoids red-eye flights as requested");
    } else {
      parts.push(
        "red-eye flight shown because no better options fully met your constraints"
      );
    }
  }

  return sentenceJoin(parts) + ".";
}

function buildHybridExplanation(flight, constraints) {
  const parts = [];

  parts.push("Recommended by hybrid score combining baseline model");

  // Route graph
  if (flight.routePopularity != null && flight.routePopularity > 0) {
    parts.push("route popularity");
  }
  if (
    (flight.departureDegree != null && flight.departureDegree > 0) ||
    (flight.arrivalDegree != null && flight.arrivalDegree > 0)
  ) {
    parts.push("network connectivity (RouteGraph degrees)");
  }

  // Reviews
  if (flight.reviewAvgRating != null) {
    const rating = flight.reviewAvgRating.toFixed(1);
    parts.push(`airline reviews around ${rating}/5`);
  }

  // Alliance
  if (
    constraints.preferAlliance &&
    constraints.preferAlliance !== "No preference"
  ) {
    if (flight.alliance === constraints.preferAlliance) {
      parts.push(`${flight.alliance} alliance matching your preference`);
    } else {
      parts.push(
        `does not match your preferred alliance (${constraints.preferAlliance})`
      );
    }
  }

  return sentenceJoin(parts) + ".";
}

function sentenceJoin(parts) {
  if (!parts.length) return "";
  if (parts.length === 1) return parts[0];
  const [first, ...rest] = parts;
  return first + " and " + rest.join(", ");
}
