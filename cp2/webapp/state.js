// cp2/webapp/state.js
// Reads form inputs and turns them into a clean "constraints" object.
// Also populates the origin/destination dropdowns from dataStore.

import {
  getAllOrigins,
  getAllDestinations
} from "./dataStore.js";

let onRecommendCallback = null;

// Cache DOM elements
let originSelect;
let destSelect;
let maxPriceInput;
let maxStopsSelect;
let avoidRedEyeCheckbox;
let allianceSelect;
let recommendButton;

/**
 * Initialize UI state:
 *  - grab DOM nodes
 *  - populate origin/destination selects
 *  - wire up "Recommend flights" button
 *
 * onRecommend(constraints) is called whenever user clicks the button.
 */
export function initState(onRecommend) {
  onRecommendCallback = onRecommend;

  originSelect = document.getElementById("origin-select");
  destSelect = document.getElementById("destination-select");
  maxPriceInput = document.getElementById("max-price-input");
  maxStopsSelect = document.getElementById("max-stops-select");
  avoidRedEyeCheckbox = document.getElementById("avoid-redeye-checkbox");
  allianceSelect = document.getElementById("alliance-select");
  recommendButton = document.getElementById("recommend-button");

  if (
    !originSelect ||
    !destSelect ||
    !maxPriceInput ||
    !maxStopsSelect ||
    !avoidRedEyeCheckbox ||
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
 */
function populateOriginAndDestination() {
  const origins = getAllOrigins();
  const destinations = getAllDestinations();

  // Origin
  originSelect.innerHTML = "";
  origins.forEach((o) => {
    const opt = document.createElement("option");
    opt.value = o;
    opt.textContent = o;
    originSelect.appendChild(opt);
  });

  // Destination
  destSelect.innerHTML = "";
  destinations.forEach((d) => {
    const opt = document.createElement("option");
    opt.value = d;
    opt.textContent = d;
    destSelect.appendChild(opt);
  });

  // Optional: if you want origin change to filter destinations, you can
  // add that later. For CP2 we keep it simple: all destinations shown.
}

/**
 * Parse max-stops value from the dropdown into a number or null.
 *
 * Supported values (from the select's value attribute or text):
 *  - "Any"                -> null
 *  - "Non-stop only"      -> 0
 *  - "Up to 1 stop"       -> 1
 *  - "Up to 2 stops"      -> 2
 *  - numeric string "0", "1", "2" also work
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
 * Returns an object like:
 *
 * {
 *   origin: "KZN",
 *   dest: "DME",
 *   maxPrice: 7000 or null,
 *   maxStops: 0 | 1 | 2 | null,
 *   avoidRedEye: true | false,
 *   preferAlliance: "No preference" | "SkyTeam" | "Star Alliance" | "Oneworld" | "None"
 * }
 */
export function getCurrentConstraints() {
  const origin = originSelect.value;
  const dest = destSelect.value;

  // Max price: empty -> null
  const rawPrice = maxPriceInput.value.trim();
  let maxPrice = null;
  if (rawPrice !== "") {
    const parsed = Number(rawPrice);
    maxPrice = Number.isFinite(parsed) ? parsed : null;
  }

  const maxStops = parseMaxStops(maxStopsSelect.value);

  const avoidRedEye = Boolean(avoidRedEyeCheckbox.checked);

  let preferAlliance = allianceSelect.value || "No preference";
  // Normalize some possible variants
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
    maxPrice,
    maxStops,
    avoidRedEye,
    preferAlliance
  };
}
