// cp2/webapp/state.js
// Reads form inputs and turns them into constraints + sort/filter state.
// Also populates the origin/destination dropdowns from dataStore.

import {
  getAllOrigins,
  getAllDestinations
} from "./dataStore.js";

let onRecommendCallback = null;

// Search form elements
let originSelect;
let destSelect;
let maxStopsSelect;
let allianceSelect;
let recommendButton;

// Sort / filter elements
let sortTabBtn;
let filtersTabBtn;
let sortPanel;
let filtersPanel;
let sortRadios;
let priceRangeInput;
let priceRangeLabel;

// Time-bucket buttons
let depBucketButtons = [];
let arrBucketButtons = [];

let currentSortBy = "model";
let currentDepartureBucket = null; // "early" | "morning" | "afternoon" | "evening" | null
let currentArrivalBucket = null;

/**
 * Initialize UI state:
 *  - grab DOM nodes
 *  - populate origin/destination selects
 *  - wire up "Recommend flights" button
 *  - setup Sort & Filters controls
 */
export function initState(onRecommend) {
  onRecommendCallback = onRecommend;

  originSelect = document.getElementById("origin-select");
  destSelect = document.getElementById("destination-select");
  maxStopsSelect = document.getElementById("max-stops-select");
  allianceSelect = document.getElementById("alliance-select");
  recommendButton = document.getElementById("recommend-button");

  if (
    !originSelect ||
    !destSelect ||
    !maxStopsSelect ||
    !allianceSelect ||
    !recommendButton
  ) {
    console.error(
      "[state] One or more form elements not found. " +
        "Check IDs in index.html match state.js."
    );
    return;
  }

  populateOriginAndDestination();
  setupSortFilterControls();

  // Hook up button
  recommendButton.addEventListener("click", (evt) => {
    evt.preventDefault();
    if (!onRecommendCallback) return;

    const constraints = getCurrentConstraints();
    onRecommendCallback(constraints);
  });
}

/**
 * Populate origin and destination selects using data from dataStore.
 * Adds a disabled placeholder so the user must choose explicitly.
 */
function populateOriginAndDestination() {
  const origins = getAllOrigins();
  const destinations = getAllDestinations();

  // Origin
  originSelect.innerHTML = "";
  const originPlaceholder = document.createElement("option");
  originPlaceholder.value = "";
  originPlaceholder.textContent = "Choose origin";
  originPlaceholder.disabled = true;
  originPlaceholder.selected = true;
  originSelect.appendChild(originPlaceholder);

  origins.forEach((o) => {
    const opt = document.createElement("option");
    opt.value = o;
    opt.textContent = o;
    originSelect.appendChild(opt);
  });

  // Destination
  destSelect.innerHTML = "";
  const destPlaceholder = document.createElement("option");
  destPlaceholder.value = "";
  destPlaceholder.textContent = "Choose destination";
  destPlaceholder.disabled = true;
  destPlaceholder.selected = true;
  destSelect.appendChild(destPlaceholder);

  destinations.forEach((d) => {
    const opt = document.createElement("option");
    opt.value = d;
    opt.textContent = d;
    destSelect.appendChild(opt);
  });
}

/**
 * Setup Sort & Filters card:
 *  - tab switching (Sort by / Filters)
 *  - sort-by radio group
 *  - price range slider label
 *  - departure/arrival time buckets
 */
function setupSortFilterControls() {
  sortTabBtn = document.getElementById("sf-tab-sort");
  filtersTabBtn = document.getElementById("sf-tab-filters");
  sortPanel = document.getElementById("sf-panel-sort");
  filtersPanel = document.getElementById("sf-panel-filters");
  priceRangeInput = document.getElementById("price-range");
  priceRangeLabel = document.getElementById("price-range-label");
  sortRadios = Array.from(
    document.querySelectorAll('input[name="sf-sort-option"]')
  );

  // Tabs
  if (sortTabBtn && filtersTabBtn && sortPanel && filtersPanel) {
    sortTabBtn.addEventListener("click", () => {
      sortTabBtn.classList.add("sort-filter-tab-active");
      filtersTabBtn.classList.remove("sort-filter-tab-active");
      sortPanel.style.display = "block";
      filtersPanel.style.display = "none";
    });

    filtersTabBtn.addEventListener("click", () => {
      filtersTabBtn.classList.add("sort-filter-tab-active");
      sortTabBtn.classList.remove("sort-filter-tab-active");
      filtersPanel.style.display = "block";
      sortPanel.style.display = "none";
    });
  }

  // Sort radios
  if (sortRadios.length > 0) {
    sortRadios.forEach((radio) => {
      radio.addEventListener("change", () => {
        if (radio.checked) {
          currentSortBy = radio.value || "model";
        }
      });
    });
  }

  // Price range slider
  if (priceRangeInput && priceRangeLabel) {
    const max = Number(priceRangeInput.max || "80000");
    priceRangeInput.value = String(max);
    updatePriceLabel();

    priceRangeInput.addEventListener("input", () => {
      updatePriceLabel();
    });
  }

  // Time bucket buttons
  depBucketButtons = [
    { key: "early", el: document.getElementById("dep-bucket-early") },
    { key: "morning", el: document.getElementById("dep-bucket-morning") },
    { key: "afternoon", el: document.getElementById("dep-bucket-afternoon") },
    { key: "evening", el: document.getElementById("dep-bucket-evening") }
  ];

  arrBucketButtons = [
    { key: "early", el: document.getElementById("arr-bucket-early") },
    { key: "morning", el: document.getElementById("arr-bucket-morning") },
    { key: "afternoon", el: document.getElementById("arr-bucket-afternoon") },
    { key: "evening", el: document.getElementById("arr-bucket-evening") }
  ];

  depBucketButtons.forEach(({ key, el }) => {
    if (!el) return;
    el.addEventListener("click", () => {
      currentDepartureBucket =
        currentDepartureBucket === key ? null : key;
      updateBucketStyles();
    });
  });

  arrBucketButtons.forEach(({ key, el }) => {
    if (!el) return;
    el.addEventListener("click", () => {
      currentArrivalBucket = currentArrivalBucket === key ? null : key;
      updateBucketStyles();
    });
  });

  updateBucketStyles();
}

function updatePriceLabel() {
  if (!priceRangeInput || !priceRangeLabel) return;
  const max = Number(priceRangeInput.max || "80000");
  const value = Number(priceRangeInput.value || String(max));

  if (!Number.isFinite(value) || value >= max) {
    priceRangeLabel.textContent = "No max limit";
  } else {
    priceRangeLabel.textContent = `Up to ${value.toFixed(0)} ₽`;
  }
}

function updateBucketStyles() {
  depBucketButtons.forEach(({ key, el }) => {
    if (!el) return;
    if (currentDepartureBucket === key) {
      el.classList.add("time-chip-active");
    } else {
      el.classList.remove("time-chip-active");
    }
  });

  arrBucketButtons.forEach(({ key, el }) => {
    if (!el) return;
    if (currentArrivalBucket === key) {
      el.classList.add("time-chip-active");
    } else {
      el.classList.remove("time-chip-active");
    }
  });
}

/**
 * Parse max-stops value from the dropdown into a number or null.
 */
function parseMaxStops(raw) {
  if (!raw || raw === "Any") return null;

  if (raw === "Non-stop only") return 0;
  if (raw === "Up to 1 stop") return 1;
  if (raw === "Up to 2 stops") return 2;

  const num = Number(raw);
  if (Number.isFinite(num)) return num;

  return null;
}

/**
 * Read and normalize the constraints from the form.
 */
export function getCurrentConstraints() {
  const originValue = originSelect.value;
  const destValue = destSelect.value;

  const origin = originValue === "" ? null : originValue;
  const dest = destValue === "" ? null : destValue;

  const maxStops = parseMaxStops(maxStopsSelect.value);

  let preferAlliance = allianceSelect.value || "No preference";
  if (
    preferAlliance !== "SkyTeam" &&
    preferAlliance !== "Star Alliance" &&
    preferAlliance !== "Oneworld" &&
    preferAlliance !== "None"
  ) {
    preferAlliance = "No preference";
  }

  return {
    origin,
    dest,
    maxPrice: null, // changed via filters
    maxStops,
    avoidRedEye: false,
    preferAlliance
  };
}

/**
 * Return the current sort + filter settings.
 */
export function getSortFilterOptions() {
  let maxPrice = null;

  if (priceRangeInput) {
    const sliderMax = Number(priceRangeInput.max || "80000");
    const value = Number(priceRangeInput.value || String(sliderMax));
    if (Number.isFinite(value) && value < sliderMax) {
      maxPrice = value;
    }
  }

  return {
    sortBy: currentSortBy, // "model" | "price" | "duration"
    maxPrice,
    departureBucket: currentDepartureBucket,
    arrivalBucket: currentArrivalBucket
  };
}
