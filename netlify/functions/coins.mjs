// Coins per username for the Word Map.
// Each user's record holds the set of words they discovered; coins are computed from that set,
// so saving the same word twice never double-counts.
//
//   GET  /api/coins?user=<name>        -> { user, coins, found }
//   GET  /api/coins?leaderboard=1      -> { players: [{ user, coins, found }] }  (top 10)
//   POST /api/coins  { user, words: [{ w, pts }] }  -> { user, coins, found, added }
import { getStore } from "@netlify/blobs";

const NAME_RE = /^[\p{L}\p{N}_.-]{3,20}$/u;
const WORD_RE = /^[a-z]+(?:[-'][a-z]+)?$/;
const store = () => getStore({ name: "word-map-coins", consistency: "strong" });
const keyFor = (name) => "users/" + name.toLowerCase();
const json = (data, status = 200) => Response.json(data, { status, headers: { "cache-control": "no-store" } });
const summary = (rec) => {
  const pts = Object.values(rec.words || {});
  return { user: rec.name, coins: pts.reduce((a, b) => a + b, 0), found: pts.length };
};

export default async (req) => {
  const url = new URL(req.url);

  if (req.method === "GET" && url.searchParams.has("leaderboard")) {
    const s = store();
    const { blobs } = await s.list({ prefix: "users/" });
    const recs = await Promise.all(blobs.slice(0, 500).map((b) => s.get(b.key, { type: "json" })));
    const players = recs.filter(Boolean).map(summary).sort((a, b) => b.coins - a.coins || b.found - a.found).slice(0, 10);
    return json({ players });
  }

  if (req.method === "GET") {
    const name = (url.searchParams.get("user") || "").trim();
    if (!NAME_RE.test(name)) return json({ error: "Usernames are 3–20 letters, numbers, dots, dashes or underscores." }, 400);
    const rec = (await store().get(keyFor(name), { type: "json" })) || { name, words: {} };
    return json(summary(rec));
  }

  if (req.method === "POST") {
    let body;
    try { body = await req.json(); } catch { return json({ error: "Send JSON." }, 400); }
    const name = String(body?.user || "").trim();
    if (!NAME_RE.test(name)) return json({ error: "Usernames are 3–20 letters, numbers, dots, dashes or underscores." }, 400);
    const incoming = Array.isArray(body?.words) ? body.words.slice(0, 50) : [];
    const s = store();
    const rec = (await s.get(keyFor(name), { type: "json" })) || { name, words: {} };
    let added = 0;
    for (const item of incoming) {
      const w = String(item?.w || "").toLowerCase();
      const pts = item?.pts === 15 ? 15 : 10; // 10 per new word, 15 for rare words
      if (WORD_RE.test(w) && w.length <= 40 && !(w in rec.words)) { rec.words[w] = pts; added++; }
    }
    if (added) { rec.updatedAt = new Date().toISOString(); await s.setJSON(keyFor(name), rec); }
    return json({ ...summary(rec), added });
  }

  return json({ error: "Method not allowed." }, 405);
};

export const config = { path: "/api/coins" };
