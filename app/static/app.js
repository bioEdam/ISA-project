const seed = [];
let debounceTimer = null;

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

function init() {
  $("#search-input").addEventListener("input", (e) => {
    clearTimeout(debounceTimer);
    const q = e.target.value.trim();
    if (q.length < 2) { renderSearchResults([]); return; }
    debounceTimer = setTimeout(() => searchTracks(q), 300);
  });

  $("#search-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      clearTimeout(debounceTimer);
      const q = e.target.value.trim();
      if (q) searchTracks(q);
    }
  });

  $("#btn-popular").addEventListener("click", loadPopular);
  $("#btn-recommend").addEventListener("click", getRecommendations);
  $("#btn-clear").addEventListener("click", clearSeed);

  loadPopular();
}

async function searchTracks(query) {
  $("#search-results").innerHTML = '<div class="loading"><span class="spinner"></span>Searching...</div>';
  try {
    const res = await fetch(`/api/search?q=${encodeURIComponent(query)}&max_results=20`);
    const data = await res.json();
    renderSearchResults(data);
  } catch {
    $("#search-results").innerHTML = '<div class="empty-state">Search failed. Try again.</div>';
  }
}

async function loadPopular() {
  $("#search-results").innerHTML = '<div class="loading"><span class="spinner"></span>Loading popular tracks...</div>';
  try {
    const res = await fetch("/api/top?n=20");
    const data = await res.json();
    renderSearchResults(data);
  } catch {
    $("#search-results").innerHTML = '<div class="empty-state">Failed to load popular tracks.</div>';
  }
}

function renderSearchResults(tracks) {
  const list = $("#search-results");
  if (!tracks.length) {
    list.innerHTML = '<div class="empty-state">No tracks found. Try a different search.</div>';
    return;
  }
  list.innerHTML = tracks.map((t, i) => `
    <li class="track-item">
      <span class="rank">${i + 1}</span>
      <div class="info">
        <div class="name">${esc(t.track_name)}</div>
        <div class="artist">${esc(t.artist_name)}</div>
      </div>
      <button class="add-btn" onclick="addToSeed(${t.corpus_idx}, '${escAttr(t.track_name)}', '${escAttr(t.artist_name)}')">+ Add</button>
    </li>
  `).join("");
}

function addToSeed(corpus_idx, track_name, artist_name) {
  if (seed.some((t) => t.corpus_idx === corpus_idx)) return;
  seed.push({ corpus_idx, track_name, artist_name });
  renderSeed();
}

function removeFromSeed(idx) {
  seed.splice(idx, 1);
  renderSeed();
}

function clearSeed() {
  seed.length = 0;
  renderSeed();
  $("#rec-results").innerHTML = '<div class="empty-state">Add tracks to your seed playlist, then click "Get Recommendations".</div>';
}

function renderSeed() {
  const list = $("#seed-list");
  const badge = $("#context-badge");
  const count = $("#seed-count");

  count.textContent = `${seed.length} track${seed.length !== 1 ? "s" : ""}`;

  if (!seed.length) {
    list.innerHTML = '<div class="empty-state">Your seed playlist is empty. Search and add tracks above.</div>';
    badge.className = "context-badge";
    badge.textContent = "";
    return;
  }

  if (seed.length <= 2) {
    badge.className = "context-badge context-weak";
    badge.textContent = "Weak context";
  } else if (seed.length <= 10) {
    badge.className = "context-badge context-good";
    badge.textContent = "Good context";
  } else {
    badge.className = "context-badge context-excellent";
    badge.textContent = "Excellent context";
  }

  list.innerHTML = seed.map((t, i) => `
    <li class="track-item">
      <span class="rank">${i + 1}</span>
      <div class="info">
        <div class="name">${esc(t.track_name)}</div>
        <div class="artist">${esc(t.artist_name)}</div>
      </div>
      <button class="btn-danger" onclick="removeFromSeed(${i})" title="Remove">&#10005;</button>
    </li>
  `).join("");
}

async function getRecommendations() {
  if (!seed.length) return;

  const k = parseInt($("#k-select").value);
  const panel = $("#rec-results");
  panel.innerHTML = '<div class="loading"><span class="spinner"></span>Generating recommendations...</div>';

  try {
    const res = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ seed_idxs: seed.map((t) => t.corpus_idx), k }),
    });
    const recs = await res.json();
    renderRecommendations(recs);
  } catch {
    panel.innerHTML = '<div class="empty-state">Recommendation failed. Try again.</div>';
  }
}

function renderRecommendations(recs) {
  const panel = $("#rec-results");
  if (!recs.length) {
    panel.innerHTML = '<div class="empty-state">No recommendations generated.</div>';
    return;
  }
  panel.innerHTML = recs.map((r) => `
    <div class="rec-item">
      <span class="rank">#${r.rank}</span>
      <div class="info">
        <div class="name">${esc(r.track_name)}</div>
        <div class="artist">${esc(r.artist_name)}</div>
      </div>
      <button class="add-btn" onclick="addToSeed(${r.corpus_idx}, '${escAttr(r.track_name)}', '${escAttr(r.artist_name)}')">+ Add</button>
    </div>
  `).join("");
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function escAttr(s) {
  return s.replace(/\\/g, "\\\\").replace(/'/g, "\\'");
}

document.addEventListener("DOMContentLoaded", init);