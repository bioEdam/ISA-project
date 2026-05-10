const ui = (() => {
    function spotifyUrl(uri) {
        if (!uri) return "";
        const parts = uri.split(":");
        return parts.length === 3 ? `https://open.spotify.com/${parts[1]}/${parts[2]}` : "";
    }

    function esc(s) {
        const d = document.createElement("div");
        d.textContent = s ?? "";
        return d.innerHTML;
    }

    function trackLink(name, trackUri) {
        const url = spotifyUrl(trackUri);
        return url
            ? `<a href="${esc(url)}" target="_blank" rel="noopener" class="spotify-link">${esc(name)}</a>`
            : esc(name);
    }

    function artistLink(name, artistUri) {
        const url = spotifyUrl(artistUri);
        return url
            ? `<a href="${esc(url)}" target="_blank" rel="noopener" class="spotify-link artist-link">${esc(name)}</a>`
            : esc(name);
    }

    function setLoading(el, message) {
        el.innerHTML = `<div class="loading"><span class="spinner"></span>${esc(message)}</div>`;
    }

    function setEmpty(el, message) {
        el.innerHTML = `<div class="empty-state">${esc(message)}</div>`;
    }

    function renderSearchResults(tracks, seed, onAdd) {
        const list = document.querySelector("#search-results");
        if (!tracks.length) {
            setEmpty(list, "No tracks found. Try a different search.");
            return;
        }
        list.innerHTML = tracks.map((t, i) => {
            const added = seed.some((s) => s.corpus_idx === t.corpus_idx);
            return `
            <li class="track-item">
              <span class="rank">${i + 1}</span>
              <div class="info">
                <div class="name">${trackLink(t.track_name, t.track_uri)}</div>
                <div class="artist">${artistLink(t.artist_name, t.artist_uri)}</div>
              </div>
              <button class="add-btn${added ? " add-btn-disabled" : ""}" data-idx="${i}" data-corpus-idx="${t.corpus_idx}" ${added ? "disabled" : ""}>${added ? "Added" : "+ Add"}</button>
            </li>`;
        }).join("");
        list.querySelectorAll(".add-btn:not(:disabled)").forEach((btn) => {
            btn.addEventListener("click", () => onAdd(tracks[+btn.dataset.idx]));
        });
    }

    function renderSeed(seed, onRemove) {
        const list = document.querySelector("#seed-list");
        const badge = document.querySelector("#context-badge");
        const count = document.querySelector("#seed-count");

        count.textContent = `${seed.length} track${seed.length !== 1 ? "s" : ""}`;

        if (!seed.length) {
            setEmpty(list, "Your seed playlist is empty. Search and add tracks above.");
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
                <div class="name">${trackLink(t.track_name, t.track_uri)}</div>
                <div class="artist">${artistLink(t.artist_name, t.artist_uri)}</div>
              </div>
              <button class="btn-danger" data-idx="${i}" title="Remove">&#10005;</button>
            </li>
        `).join("");

        list.querySelectorAll(".btn-danger").forEach((btn) => {
            btn.addEventListener("click", () => onRemove(+btn.dataset.idx));
        });
    }

    function renderRecommendations(recs, onAdd) {
        const panel = document.querySelector("#rec-results");
        if (!recs.length) {
            setEmpty(panel, "No recommendations generated.");
            return;
        }
        panel.innerHTML = recs.map((r, i) => `
            <div class="rec-item">
              <span class="rank">#${r.rank}</span>
              <div class="info">
                <div class="name">${trackLink(r.track_name, r.track_uri)}</div>
                <div class="artist">${artistLink(r.artist_name, r.artist_uri)}</div>
              </div>
              <button class="add-btn" data-idx="${i}" data-corpus-idx="${r.corpus_idx}">+ Add</button>
            </div>
        `).join("");
        panel.querySelectorAll(".add-btn").forEach((btn) => {
            btn.addEventListener("click", () => onAdd(recs[+btn.dataset.idx]));
        });
    }

    function updateAddButtons(corpusIdx, added) {
        document.querySelectorAll(`.add-btn[data-corpus-idx="${corpusIdx}"]`).forEach((btn) => {
            if (added) {
                btn.classList.add("add-btn-disabled");
                btn.disabled = true;
                btn.textContent = "Added";
            } else {
                btn.classList.remove("add-btn-disabled");
                btn.disabled = false;
                btn.textContent = "+ Add";
            }
        });
    }

    function renderPlaylistList(playlists, activeId, { onLoad, onEdit, onDelete }) {
        const list = document.querySelector("#playlist-list");
        if (!playlists.length) {
            setEmpty(list, "No playlists saved yet.");
            return;
        }
        list.innerHTML = playlists.map(pl => `
            <li class="playlist-item${pl.id === activeId ? ' active' : ''}" data-id="${pl.id}">
              <span class="pl-name" title="${esc(pl.name)}">${esc(pl.name)}</span>
              <span class="pl-actions">
                <button class="btn-icon pl-edit" data-id="${pl.id}" title="Edit">&#9998;</button>
                <button class="btn-icon pl-delete" data-id="${pl.id}" title="Delete">&#128465;</button>
              </span>
            </li>
        `).join("");

        list.querySelectorAll(".playlist-item").forEach(el => {
            el.addEventListener("click", (e) => {
                if (e.target.closest(".pl-actions")) return;
                onLoad(+el.dataset.id);
            });
        });
        list.querySelectorAll(".pl-edit").forEach(btn => {
            btn.addEventListener("click", () => onEdit(+btn.dataset.id));
        });
        list.querySelectorAll(".pl-delete").forEach(btn => {
            btn.addEventListener("click", () => onDelete(+btn.dataset.id));
        });
    }

    function renderPagination(page, pages, onPageChange) {
        const container = document.querySelector("#playlist-pagination");
        if (pages <= 1) { container.innerHTML = ""; return; }
        container.innerHTML = `
            <button class="btn-secondary btn-small" id="pg-prev" ${page <= 1 ? 'disabled' : ''}>Prev</button>
            <span class="page-info">${page} / ${pages}</span>
            <button class="btn-secondary btn-small" id="pg-next" ${page >= pages ? 'disabled' : ''}>Next</button>
        `;
        const prev = container.querySelector("#pg-prev");
        const next = container.querySelector("#pg-next");
        if (prev) prev.addEventListener("click", () => onPageChange(page - 1));
        if (next) next.addEventListener("click", () => onPageChange(page + 1));
    }

    return {
        setLoading, setEmpty, renderSearchResults, renderSeed,
        renderRecommendations, updateAddButtons,
        renderPlaylistList, renderPagination,
    };
})();
