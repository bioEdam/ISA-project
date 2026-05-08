const seed = [];
let recommendations = [];

function updateSaveBtn(state) {
    const btn = document.querySelector("#btn-save");
    if (!btn) return;
    if (state === "unsaved") {
        btn.textContent = "Save";
        btn.disabled = false;
        btn.className = "btn-primary btn-small";
    } else if (state === "saved") {
        btn.textContent = "Saved";
        btn.disabled = true;
        btn.className = "btn-secondary btn-small";
    } else {
        btn.textContent = "Save";
        btn.disabled = true;
        btn.className = "btn-secondary btn-small";
    }
}

function addToSeed(track) {
    if (seed.some((t) => t.corpus_idx === track.corpus_idx)) return;
    seed.push({
        corpus_idx: track.corpus_idx,
        track_name: track.track_name,
        artist_name: track.artist_name,
        track_uri: track.track_uri || "",
        artist_uri: track.artist_uri || "",
    });
    updateSaveBtn("unsaved");
    ui.renderSeed(seed, removeFromSeed);
    ui.updateAddButtons(track.corpus_idx, true);
}

function removeFromSeed(idx) {
    const removed = seed.splice(idx, 1)[0];
    updateSaveBtn(seed.length ? "unsaved" : "empty");
    ui.renderSeed(seed, removeFromSeed);
    ui.updateAddButtons(removed.corpus_idx, false);
}

function clearSeed() {
    const removedIdxs = seed.map((t) => t.corpus_idx);
    seed.length = 0;
    recommendations = [];
    updateSaveBtn("empty");
    ui.renderSeed(seed, removeFromSeed);
    removedIdxs.forEach((idx) => ui.updateAddButtons(idx, false));
    ui.setEmpty(document.querySelector("#rec-results"), 'Add tracks to your seed playlist, then click "Get Recommendations".');
}

async function doSearch(q) {
    const list = document.querySelector("#search-results");
    ui.setLoading(list, "Searching...");
    try {
        const tracks = await api.searchTracks(q);
        ui.renderSearchResults(tracks, seed, addToSeed);
    } catch {
        ui.setEmpty(list, "Search failed. Try again.");
    }
}

async function doLoadPopular() {
    const list = document.querySelector("#search-results");
    ui.setLoading(list, "Loading popular tracks...");
    try {
        const tracks = await api.loadPopular();
        ui.renderSearchResults(tracks, seed, addToSeed);
    } catch {
        ui.setEmpty(list, "Failed to load popular tracks.");
    }
}

async function doGetRecommendations() {
    if (!seed.length) return;
    const k = parseInt(document.querySelector("#k-select").value);
    const panel = document.querySelector("#rec-results");
    ui.setLoading(panel, "Generating recommendations...");
    try {
        const recs = await api.getRecommendations(seed.map((t) => t.corpus_idx), k);
        recommendations = recs;
        ui.renderRecommendations(recs, addToSeed);
    } catch {
        ui.setEmpty(panel, "Recommendation failed. Try again.");
    }
}

async function doSavePlaylist() {
    if (!seed.length) return;
    const nameInput = document.querySelector("#playlist-name");
    const saveBtn = document.querySelector("#btn-save");
    const statusEl = document.querySelector("#save-status");

    const name = nameInput.value.trim() || "My Playlist";
    const tracks = [
        ...seed.map((t) => ({ track_uri: t.track_uri, track_name: t.track_name, artist_name: t.artist_name, is_seed: true })),
        ...recommendations.map((r) => ({ track_uri: r.track_uri, track_name: r.track_name, artist_name: r.artist_name, is_seed: false })),
    ];

    saveBtn.textContent = "Saving...";
    saveBtn.disabled = true;
    statusEl.textContent = "";
    try {
        const result = await api.savePlaylist(name, tracks);
        statusEl.textContent = `Saved! Playlist #${result.id}`;
        statusEl.className = "save-status save-status-ok";
        nameInput.value = "";
        updateSaveBtn("saved");
        setTimeout(() => { statusEl.textContent = ""; statusEl.className = "save-status"; }, 4000);
    } catch {
        statusEl.textContent = "Save failed. Try again.";
        statusEl.className = "save-status save-status-err";
        updateSaveBtn("unsaved");
    }
}

function init() {
    let debounceTimer = null;
    const searchInput = document.querySelector("#search-input");

    updateSaveBtn("empty");

    searchInput.addEventListener("input", (e) => {
        clearTimeout(debounceTimer);
        const q = e.target.value.trim();
        if (q.length < 2) { ui.renderSearchResults([], seed, addToSeed); return; }
        debounceTimer = setTimeout(() => doSearch(q), 300);
    });

    searchInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            clearTimeout(debounceTimer);
            const q = e.target.value.trim();
            if (q) doSearch(q);
        }
    });

    document.querySelector("#btn-popular").addEventListener("click", doLoadPopular);
    document.querySelector("#btn-recommend").addEventListener("click", doGetRecommendations);
    document.querySelector("#btn-clear").addEventListener("click", clearSeed);
    document.querySelector("#btn-save").addEventListener("click", doSavePlaylist);

    doLoadPopular();
}

document.addEventListener("DOMContentLoaded", init);
