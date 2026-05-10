const api = {
    async searchTracks(query, maxResults = 20) {
        const res = await fetch(`/api/search?q=${encodeURIComponent(query)}&max_results=${maxResults}`);
        if (!res.ok) throw new Error("Search failed");
        return res.json();
    },

    async loadPopular(n = 20) {
        const res = await fetch(`/api/top?n=${n}`);
        if (!res.ok) throw new Error("Failed to load popular tracks");
        return res.json();
    },

    async getRecommendations(seedIdxs, k) {
        const res = await fetch("/api/recommend", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ seed_idxs: seedIdxs, k }),
        });
        if (!res.ok) throw new Error("Recommendation failed");
        return res.json();
    },

    async savePlaylist(name, tracks) {
        const res = await fetch("/api/playlists", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, tracks }),
        });
        if (!res.ok) throw new Error("Failed to save playlist");
        return res.json();
    },

    async listPlaylists(page = 1, perPage = 20) {
        const res = await fetch(`/api/playlists?page=${page}&per_page=${perPage}`);
        if (!res.ok) throw new Error("Failed to load playlists");
        return res.json();
    },

    async getPlaylist(id) {
        const res = await fetch(`/api/playlists/${id}`);
        if (!res.ok) throw new Error("Failed to load playlist");
        return res.json();
    },

    async renamePlaylist(id, name) {
        const res = await fetch(`/api/playlists/${id}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name }),
        });
        if (!res.ok) throw new Error("Failed to rename playlist");
        return res.json();
    },

    async updatePlaylistTracks(id, tracks) {
        const res = await fetch(`/api/playlists/${id}/tracks`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ tracks }),
        });
        if (!res.ok) throw new Error("Failed to update tracks");
        return res.json();
    },

    async deletePlaylist(id) {
        const res = await fetch(`/api/playlists/${id}`, { method: "DELETE" });
        if (!res.ok) throw new Error("Failed to delete playlist");
    },

    async resolveUris(uris) {
        const res = await fetch("/api/resolve-uris", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ uris }),
        });
        if (!res.ok) throw new Error("Failed to resolve URIs");
        return res.json();
    },
};
