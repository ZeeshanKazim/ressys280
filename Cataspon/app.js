// app.js — wire UI, load, run both tabs
import {$} from './utils.js';
import { loadGraphData, runGraphTab, GraphState } from './graph.js';
import { loadTextData, runTextTab, TextState } from './text.js';

let ready = {graph:false, text:false};

function setStatus(msg){ $('#status').textContent = `Status: ${msg}`; }
function setEnabled(){
  const on = (ready.graph && ready.text);
  $('#btnRunGraph').disabled = !on;
  $('#btnRunText').disabled = !on;
}
function wireTabs(){
  const tabs = Array.from(document.querySelectorAll('.tab'));
  tabs.forEach(btn=>{
    btn.addEventListener('click', ()=>{
      tabs.forEach(b=>b.classList.remove('active'));
      btn.classList.add('active');
      const id = btn.dataset.tab;
      document.querySelectorAll('.tabpane').forEach(p=> p.classList.remove('active'));
      document.getElementById('tab-'+id).classList.add('active');
    });
  });
}
function onTradeoff(){
  const v = Number($('#tradeoff').value);
  $('#tradeoffLabel').textContent = v<0.34 ? "Fastest" : v>0.66 ? "Cheapest" : "Balanced";
}
function fillAirportSelects(){
  const origins = Object.keys(GraphState.airports).sort();
  const selO = $('#origin'), selD = $('#destination');
  selO.innerHTML = ""; selD.innerHTML = "";
  for(const iata of origins){
    const opt1 = document.createElement('option'); opt1.value=iata; opt1.textContent=`${iata} — ${GraphState.airports[iata].city||""}`;
    const opt2 = opt1.cloneNode(true);
    selO.appendChild(opt1); selD.appendChild(opt2);
  }
  if(GraphState.airports.AMS) selO.value="AMS";
  if(GraphState.airports.JFK) selD.value="JFK";
}
function gatherParams(){
  return {
    origin: $('#origin').value,
    destination: $('#destination').value,
    date: $('#date').value || new Date().toISOString().slice(0,10),
    time: $('#time').value || "09:00",
    maxStops: Number($('#stops').value),
    alliance: $('#alliance').value || "",
    tradeoff: Number($('#tradeoff').value),
    freeText: $('#query').value || ""
  };
}

// -------- boot --------
window.addEventListener('DOMContentLoaded', ()=>{
  wireTabs();
  $('#tradeoff').addEventListener('input', onTradeoff);
  onTradeoff();

  $('#btnLoad').addEventListener('click', async ()=>{
    try{
      setStatus("loading data…");
      await loadGraphData();
      await loadTextData();
      ready.graph = GraphState.loaded;
      ready.text  = TextState.loaded;
      setEnabled();
      fillAirportSelects();
      setStatus(`data loaded: ${Object.keys(GraphState.airports).length} airports, ${GraphState.routes.length} routes, ${TextState.reviews.length} reviews.`);
    }catch(e){
      console.error(e);
      setStatus("error: check /data files (case-sensitive).");
    }
  });

  $('#btnRunGraph').addEventListener('click', ()=>{
    const p = gatherParams();
    const rows = runGraphTab(p);
    setStatus(`Tab A done • ${rows.length} candidates.`);
  });

  $('#btnRunText').addEventListener('click', ()=>{
    const p = gatherParams();
    const rows = runTextTab(p);
    setStatus(`Tab B done • ${rows.length} candidates.`);
  });
});
