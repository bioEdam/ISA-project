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
};
