const seed = [];
let recommendations = [];
let playlistPage = 1;
let playlistsData = { items: [], total: 0, page: 1, pages: 0 };
let activePlaylistId = null;
let editMode = false;

function updateSaveBtn(state) {
    const btn = document.querySelector("#btn-save");
    if (!btn) return;
    if (editMode && activePlaylistId) {
        btn.textContent = "Update";
        btn.disabled = state === "saved";
        btn.className = state === "saved" ? "btn-secondary btn-small" : "btn-primary btn-small";
        return;
    }
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

function updateEditIndicator() {
    const el = document.querySelector("#edit-indicator");
    if (!el) return;
    if (editMode && activePlaylistId) {
        const pl = playlistsData.items.find(p => p.id === activePlaylistId);
        el.textContent = `Editing: ${pl?.name || `Playlist #${activePlaylistId}`}`;
        el.style.display = "";
    } else {
        el.textContent = "";
        el.style.display = "none";
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
    activePlaylistId = null;
    editMode = false;
    updateSaveBtn("empty");
    updateEditIndicator();
    ui.renderSeed(seed, removeFromSeed);
    removedIdxs.forEach((idx) => ui.updateAddButtons(idx, false));
    ui.setEmpty(document.querySelector("#rec-results"), 'Add tracks to your seed playlist, then click "Get Recommendations".');
    refreshPlaylistHighlight();
}

function refreshPlaylistHighlight() {
    ui.renderPlaylistList(playlistsData.items, activePlaylistId, {
        onLoad: loadPlaylistIntoSeed,
        onEdit: enterEditMode,
        onDelete: doDeletePlaylist,
    });
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

async function loadPlaylists(page = 1) {
    playlistPage = page;
    try {
        playlistsData = await api.listPlaylists(page, 20);
    } catch {
        playlistsData = { items: [], total: 0, page: 1, pages: 0 };
    }
    ui.renderPlaylistList(playlistsData.items, activePlaylistId, {
        onLoad: loadPlaylistIntoSeed,
        onEdit: enterEditMode,
        onDelete: doDeletePlaylist,
    });
    ui.renderPagination(playlistsData.page, playlistsData.pages, loadPlaylists);
}

async function loadPlaylistIntoSeed(playlistId) {
    const list = document.querySelector("#seed-list");
    ui.setLoading(list, "Loading playlist...");
    try {
        const playlist = await api.getPlaylist(playlistId);
        const seedTracks = playlist.tracks.filter(t => t.is_seed);
        const uris = seedTracks.map(t => t.track_uri);

        const resolved = uris.length > 0 ? await api.resolveUris(uris) : [];

        const removedIdxs = seed.map(t => t.corpus_idx);
        seed.length = 0;
        recommendations = [];
        removedIdxs.forEach(idx => ui.updateAddButtons(idx, false));

        for (const r of resolved) {
            if (r !== null) addToSeed(r);
        }

        activePlaylistId = playlistId;
        editMode = false;
        document.querySelector("#playlist-name").value = playlist.name;
        updateSaveBtn("saved");
        updateEditIndicator();
        refreshPlaylistHighlight();
        ui.setEmpty(document.querySelector("#rec-results"), 'Click "Get Recommendations" to generate suggestions.');
    } catch {
        ui.setEmpty(list, "Failed to load playlist.");
    }
}

async function enterEditMode(playlistId) {
    if (activePlaylistId !== playlistId) {
        await loadPlaylistIntoSeed(playlistId);
    }
    editMode = true;
    activePlaylistId = playlistId;
    updateSaveBtn("unsaved");
    updateEditIndicator();
    refreshPlaylistHighlight();
}

async function doDeletePlaylist(playlistId) {
    const pl = playlistsData.items.find(p => p.id === playlistId);
    const name = pl?.name || "this playlist";
    if (!confirm(`Delete playlist "${name}"? This cannot be undone.`)) return;
    try {
        await api.deletePlaylist(playlistId);
        if (activePlaylistId === playlistId) {
            activePlaylistId = null;
            editMode = false;
            clearSeed();
        }
        loadPlaylists(playlistPage);
    } catch {
        alert("Failed to delete playlist.");
    }
}

async function doSaveOrUpdate() {
    if (!seed.length) return;
    const nameInput = document.querySelector("#playlist-name");
    const statusEl = document.querySelector("#save-status");
    const name = nameInput.value.trim() || "My Playlist";

    const tracks = [
        ...seed.map(t => ({ track_uri: t.track_uri, track_name: t.track_name, artist_name: t.artist_name, is_seed: true })),
        ...recommendations.map(r => ({ track_uri: r.track_uri, track_name: r.track_name, artist_name: r.artist_name, is_seed: false })),
    ];

    const btn = document.querySelector("#btn-save");
    btn.textContent = editMode ? "Updating..." : "Saving...";
    btn.disabled = true;
    statusEl.textContent = "";

    try {
        if (editMode && activePlaylistId) {
            await api.renamePlaylist(activePlaylistId, name);
            await api.updatePlaylistTracks(activePlaylistId, tracks);
            statusEl.textContent = "Updated!";
        } else {
            const result = await api.savePlaylist(name, tracks);
            activePlaylistId = result.id;
            statusEl.textContent = `Saved! Playlist #${result.id}`;
        }
        statusEl.className = "save-status save-status-ok";
        updateSaveBtn("saved");
        loadPlaylists(playlistPage);
        setTimeout(() => { statusEl.textContent = ""; statusEl.className = "save-status"; }, 4000);
    } catch {
        statusEl.textContent = editMode ? "Update failed. Try again." : "Save failed. Try again.";
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
    document.querySelector("#btn-save").addEventListener("click", doSaveOrUpdate);

    doLoadPopular();
    loadPlaylists();
}

document.addEventListener("DOMContentLoaded", init);
