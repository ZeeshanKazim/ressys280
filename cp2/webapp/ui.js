// ui.js
// DOM wiring: dropdowns, submit handler, and result rendering.

function populateAirportDropdowns(state) {
  const originSelect = document.getElementById("origin-select");
  const destSelect = document.getElementById("dest-select");

  originSelect.innerHTML = "";
  destSelect.innerHTML = "";

  state.allOrigins.forEach((code) => {
    const opt = document.createElement("option");
    opt.value = code;
    opt.textContent = code;
    originSelect.appendChild(opt);
  });

  state.allDests.forEach((code) => {
    const opt = document.createElement("option");
    opt.value = code;
    opt.textContent = code;
    destSelect.appendChild(opt);
  });

  if (state.allOrigins.length) {
    originSelect.value = state.allOrigins[0];
    state.currentOrigin = state.allOrigins[0];
  }
  if (state.allDests.length) {
    destSelect.value = state.allDests[0];
    state.currentDest = state.allDests[0];
  }
}

function readConstraintsFromForm() {
  const maxPriceRaw = document.getElementById("max-price-input").value;
  const maxStopsRaw = document.getElementById("max-stops-select").value;
  const avoidRedEye = document.getElementById(
    "avoid-red-eye-checkbox"
  ).checked;
  const preferredAlliance = document.getElementById(
    "alliance-select"
  ).value;

  const maxPrice =
    maxPriceRaw === "" ? null : Number.parseFloat(maxPriceRaw || "0");
  const maxStops = Number.parseInt(maxStopsRaw || "3", 10);

  return {
    maxPrice: Number.isFinite(maxPrice) ? maxPrice : null,
    maxStops: Number.isFinite(maxStops) ? maxStops : 3,
    avoidRedEye,
    preferredAlliance,
  };
}

function formatMoney(v) {
  if (!Number.isFinite(Number(v))) return "N/A";
  return Number(v).toFixed(0);
}

function formatDurationMinutes(v) {
  const num = Number(v);
  if (!Number.isFinite(num)) return "N/A";
  const h = Math.floor(num / 60);
  const m = Math.round(num % 60);
  return `${h}h ${m}m`;
}

function renderResultList(containerId, items) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";

  items.forEach((item, index) => {
    const flight = item.flight;
    const explanation = item.explanation;
    const badges = item.badges || [];

    const li = document.createElement("li");
    li.className = "result-item";

    const header = document.createElement("div");
    header.className = "result-header";

    const rankSpan = document.createElement("span");
    rankSpan.className = "result-rank";
    rankSpan.textContent = `#${index + 1}`;

    const titleSpan = document.createElement("span");
    titleSpan.className = "result-title";
    titleSpan.textContent = `${flight.carrier || "??"}  ${
      flight.origin || "?"
    } → ${flight.dest || "?"}`;

    const badgesDiv = document.createElement("div");
    badgesDiv.className = "result-badges";
    badges.forEach((label, i) => {
      const badge = document.createElement("span");
      badge.className = "badge";
      if (index === 0 && i === 0) {
        badge.classList.add("badge-accent");
      }
      badge.textContent = label;
      badgesDiv.appendChild(badge);
    });

    header.appendChild(rankSpan);
    header.appendChild(titleSpan);
    header.appendChild(badgesDiv);

    const meta = document.createElement("div");
    meta.className = "result-meta";

    const priceSpan = document.createElement("span");
    priceSpan.textContent = `Price: ${formatMoney(flight.totalPrice)}`;

    const stopsSpan = document.createElement("span");
    const stops = Number(flight.num_stops || 0);
    stopsSpan.textContent = `Stops: ${stops}`;

    const durationSpan = document.createElement("span");
    durationSpan.textContent = `Duration: ${formatDurationMinutes(
      flight.duration_minutes
    )}`;

    const depSpan = document.createElement("span");
    depSpan.textContent = `Departure: ${
      flight.legs0_departureAt || "unknown"
    }`;

    meta.appendChild(priceSpan);
    meta.appendChild(stopsSpan);
    meta.appendChild(durationSpan);
    meta.appendChild(depSpan);

    const expl = document.createElement("div");
    expl.className = "result-expl";
    expl.textContent = explanation;

    li.appendChild(header);
    li.appendChild(meta);
    li.appendChild(expl);

    container.appendChild(li);
  });
}

export function bindUI({ state, rankBaseline, rankEnhanced }) {
  populateAirportDropdowns(state);

  const form = document.getElementById("search-form");
  const noResultsEl = document.getElementById("no-results-message");

  form.addEventListener("submit", (evt) => {
    evt.preventDefault();

    const originSelect = document.getElementById("origin-select");
    const destSelect = document.getElementById("dest-select");

    const origin = originSelect.value;
    const dest = destSelect.value;
    const constraints = readConstraintsFromForm();

    state.currentOrigin = origin;
    state.currentDest = dest;
    state.constraints = constraints;

    const baselineItems = rankBaseline({
      state,
      origin,
      dest,
      constraints,
    });

    const enhancedItems = rankEnhanced({
      state,
      origin,
      dest,
      constraints,
    });

    const hasAny =
      (baselineItems && baselineItems.length > 0) ||
      (enhancedItems && enhancedItems.length > 0);

    if (hasAny) {
      noResultsEl.classList.add("hidden");
    } else {
      noResultsEl.classList.remove("hidden");
    }

    renderResultList("baseline-results", baselineItems);
    renderResultList("enhanced-results", enhancedItems);
  });
}
