// cp2/webapp/ui.js
// Rendering logic for the recommendations: banner + two tables.

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
 * Render both standard and smart recommendation tables.
 */
export function renderRecommendations(opts) {
  const { baseline, hybrid, hardConstraintsSatisfied } = opts;

  if (!baselineContainer || !hybridContainer) {
    initUI();
  }

  baselineContainer.innerHTML = "";
  hybridContainer.innerHTML = "";

  renderBanner(hardConstraintsSatisfied);

  renderTable(baselineContainer, baseline, "baseline");
  renderTable(hybridContainer, hybrid, "hybrid");
}

function renderBanner(hardConstraintsSatisfied) {
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
  }
}

/**
 * Build a scrollable, fixed-layout table of flights.
 */
function renderTable(container, flights, columnType) {
  if (!flights || flights.length === 0) {
    const msg = document.createElement("div");
    msg.className = "no-results";
    msg.textContent = "No flights found for this route.";
    container.appendChild(msg);
    return;
  }

  const table = document.createElement("table");
  table.className = "flight-table";

  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  const headers = [
    "#",
    "Airline",
    "Route",
    "Departure",
    "Duration",
    "Stops",
    "Price",
    "Tags"
  ];

  headers.forEach((h) => {
    const th = document.createElement("th");
    th.textContent = h;
    headerRow.appendChild(th);
  });

  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");

  flights.forEach((flight, idx) => {
    const tr = document.createElement("tr");

    // Rank
    const rankTd = document.createElement("td");
    rankTd.textContent = `#${idx + 1}`;
    tr.appendChild(rankTd);

    // Airline
    const airlineTd = document.createElement("td");
    airlineTd.textContent = flight.airlineName || flight.carrier;
    tr.appendChild(airlineTd);

    // Route
    const routeTd = document.createElement("td");
    routeTd.textContent = `${flight.origin} → ${flight.dest}`;
    tr.appendChild(routeTd);

    // Departure
    const depTd = document.createElement("td");
    depTd.textContent = formatDateTime(flight.departureIso);
    tr.appendChild(depTd);

    // Duration
    const durTd = document.createElement("td");
    durTd.textContent = formatDuration(flight.durationMinutes);
    tr.appendChild(durTd);

    // Stops
    const stopsTd = document.createElement("td");
    const stops = Number(flight.stops || 0);
    if (stops === 0) {
      stopsTd.textContent = "Non-stop";
    } else if (stops === 1) {
      stopsTd.textContent = "1 stop";
    } else {
      stopsTd.textContent = `${stops} stops`;
    }
    tr.appendChild(stopsTd);

    // Price
    const priceTd = document.createElement("td");
    priceTd.textContent = `${formatPrice(flight.price)} ₽`;
    tr.appendChild(priceTd);

    // Tags
    const tagsTd = document.createElement("td");
    tagsTd.className = "tags-cell";

    const tagsContainer = document.createElement("div");
    tagsContainer.className = "tags-container";

    if (idx === 0) {
      const topBadge = document.createElement("span");
      topBadge.className = "badge badge-primary";
      topBadge.textContent =
        columnType === "baseline" ? "Top pick" : "Top smart pick";
      tagsContainer.appendChild(topBadge);
    }

    if (flight.isCheapest) {
      const b = document.createElement("span");
      b.className = "badge badge-neutral";
      b.textContent = "Cheapest";
      tagsContainer.appendChild(b);
    }

    if (flight.isFastest) {
      const b = document.createElement("span");
      b.className = "badge badge-neutral";
      b.textContent = "Fastest";
      tagsContainer.appendChild(b);
    }

    if (Number(flight.stops || 0) === 0) {
      const b = document.createElement("span");
      b.className = "badge badge-neutral";
      b.textContent = "Non-stop";
      tagsContainer.appendChild(b);
    }

    tagsTd.appendChild(tagsContainer);
    tr.appendChild(tagsTd);

    tbody.appendChild(tr);
  });

  table.appendChild(tbody);

  // Scrollable wrapper to keep layout clean
  const wrapper = document.createElement("div");
  wrapper.className = "table-wrapper";
  wrapper.appendChild(table);

  container.appendChild(wrapper);
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
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  });
}
