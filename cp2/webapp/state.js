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
 *  - populate origin/destination selects (with placeholders)
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
 * Returns an object like:
 *
 * {
 *   origin: "KZN" | null,
 *   dest: "DME" | null,
 *   maxPrice: number | null,
 *   maxStops: 0 | 1 | 2 | null,
 *   avoidRedEye: true | false,
 *   preferAlliance: "No preference" | "SkyTeam" | "Star Alliance" | "Oneworld" | "None"
 * }
 */
export function getCurrentConstraints() {
  const originValue = originSelect.value;
  const destValue = destSelect.value;

  const origin = originValue === "" ? null : originValue;
  const dest = destValue === "" ? null : destValue;

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
