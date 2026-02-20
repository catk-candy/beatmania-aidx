#!/usr/bin/env node
/**
 * genChartFromAnalysis.js
 * Usage:
 *   node genChartFromAnalysis.js analysis.json > chart.json
 *
 * Narrow lanes:
 *   node genChartFromAnalysis.js -n analysis.json > chart.json
 *
 * Input analysis.json (minimal):
 * {
 *   "title": "Generated EDM",
 *   "artist": "Generated",
 *   "audio": "song.wav",
 *   "bpm": 174,
 *   "offsetSec": 0.008,
 *   "timeSignature": [4,4],
 *   "bars": 130,
 *   "normalNotesTarget": 1450,
 *   "seed": 174008,
 *   "sections": [
 *     { "fromBar": 0, "toBar": 15, "kind": "intro", "kick": "Light" },
 *     { "fromBar": 16, "toBar": 23, "kind": "build", "kick": "Light" },
 *     { "fromBar": 24, "toBar": 31, "kind": "predrop", "kick": "Light" },
 *     { "fromBar": 32, "toBar": 47, "kind": "drop1", "kick": "Heavy" }
 *   ]
 * }
 *
 * Notes:
 * - bpm / bars / normalNotesTarget / sections(fromBar,toBar,kind,kick) を読んで譜面生成します
 * - shiftNotesTarget は normalNotesTarget の 1/25（四捨五入、最低1）で自動決定します
 * - ★sections の境目（各 fromBar）には必ず Shift を入れます
 * - ★同時押しは「気持ちいい形」のみを優先します（2/3同時押し）
 */

"use strict";

const fs = require("fs");

// -------------------- constants --------------------
const LANES = [
  { index: 0, key: "Shift" },
  { index: 1, key: "KeyZ" },
  { index: 2, key: "KeyS" },
  { index: 3, key: "KeyX" },
  { index: 4, key: "KeyD" },
  { index: 5, key: "KeyC" },
  { index: 6, key: "KeyF" },
  { index: 7, key: "KeyV" },
];

// ★ -n で使うレーン（Shiftは別扱いなのでここは1..7のみ）
const NARROW_PLAY_LANES = [1, 3, 5, 7];

// patterns (lane indices)
const STAIR_UP = [1, 2, 3, 4, 5, 6, 7];
const STAIR_DN = [7, 6, 5, 4, 3, 2, 1];
const ZIGZAG_A = [1, 4, 2, 5, 3, 6, 2, 5];
const ZIGZAG_B = [7, 4, 6, 3, 5, 2, 6, 3];
const TRILL_L = [1, 3, 1, 3, 1, 3, 2, 4];
const TRILL_R = [7, 5, 7, 5, 7, 5, 6, 4];

const DEFAULT_NORMAL_TARGET = 1450;
const DEFAULT_SEED = 174008;

// density control (notes per bar) - normal notes only
const DENSITY_PRESETS = {
  intro: { min: 6, max: 10 },
  build: { min: 10, max: 16 },
  predrop: { min: 14, max: 22 },
  drop1: { min: 18, max: 32 },
  trans: { min: 8, max: 12 },
  groove: { min: 12, max: 18 },
  minidrop: { min: 18, max: 30 },
  breakdown: { min: 6, max: 12 },
  snareroll: { min: 16, max: 28 },
  drop2: { min: 22, max: 36 },
  bridge: { min: 10, max: 16 },
  final: { min: 18, max: 30 },
  outro: { min: 6, max: 10 },
};

const HISTORY_MAX = 10;
const FORBID_SAME_AS_PREV = true;

// ★高密度での縦連（頻発）を強く抑える
const HIGH_DENSE_KINDS = new Set(["drop1", "drop2", "final", "minidrop"]);
const HIGH_DENSE_HARD_WINDOW = 4; // 直近4手
const BAR_SPREAD_STRENGTH = 0.75; // 0.0~1.0

// ★気持ちいい同時押し（Shift含めて0..7の並びで定義、ただし chord は 1..7のみを使う）
// 2同時：13,35,57,24,46,17
const PLEASANT_CHORDS_2 = [
  [1, 3],
  [3, 5],
  [5, 7],
  [2, 4],
  [4, 6],
  [1, 7],
];

// 3同時：135,357,123,567,246,147
const PLEASANT_CHORDS_3 = [
  [1, 3, 5],
  [3, 5, 7],
  [1, 2, 3],
  [5, 6, 7],
  [2, 4, 6],
  [1, 4, 7],
];

// ---- 추가設定（同時押し多すぎ対策：構成に応じて16分乱打を挿入） ----
const STREAM16_KINDS = new Set(["drop1", "drop2", "final", "minidrop", "groove"]);
// 「この小節は乱打寄り」にする確率（kind別）
const STREAM16_BAR_PROB = {
  drop1: 0.55,
  drop2: 0.60,
  final: 0.60,
  minidrop: 0.50,
  groove: 0.35,
};
// 同時押しの基礎確率（kind別）：全体的に控えめにする
const CHORD_PROB = {
  intro: 0.03,
  build: 0.06,
  predrop: 0.12,
  drop1: 0.10,
  trans: 0.03,
  groove: 0.07,
  minidrop: 0.10,
  breakdown: 0.01,
  snareroll: 0.00,
  drop2: 0.12,
  bridge: 0.06,
  final: 0.12,
  outro: 0.03,
};
// 強拍での同時押し上乗せ（ただし乱打小節では無効）
const STRONG_BEAT_CHORD_BONUS = 0.08;

// -------------------- utils --------------------
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function pickWeighted(rng, items) {
  const sum = items.reduce((s, it) => s + it.w, 0);
  if (sum <= 0) return items[0].value;
  let r = rng() * sum;
  for (const it of items) {
    r -= it.w;
    if (r <= 0) return it.value;
  }
  return items[items.length - 1].value;
}

function normalizeKind(kind) {
  const k = String(kind || "").trim().toLowerCase();
  const allowed = new Set([
    "intro", "build", "predrop", "drop1", "trans", "groove", "minidrop",
    "breakdown", "snareroll", "drop2", "bridge", "final", "outro"
  ]);
  if (allowed.has(k)) return k;
  return null;
}

function normalizeKick(kick) {
  const k = String(kick || "").trim().toLowerCase();
  if (k.includes("heavy")) return "Heavy";
  if (k.includes("light")) return "Light";
  return "Unknown";
}

function sectionForBar(sections, bar) {
  for (const s of sections) {
    if (bar >= s.fromBar && bar <= s.toBar) return s;
  }
  return null;
}

function classifyKind(bar, sec, barsTotal) {
  const k = normalizeKind(sec?.kind);
  if (k) return k;

  const p = bar / Math.max(1, barsTotal - 1);
  if (p < 0.12) return "intro";
  if (p < 0.18) return "build";
  if (p < 0.24) return "predrop";
  if (p < 0.36) return "drop1";
  if (p < 0.40) return "trans";
  if (p < 0.50) return "groove";
  if (p < 0.53) return "minidrop";
  if (p < 0.58) return "breakdown";
  if (p < 0.62) return "snareroll";
  if (p < 0.68) return "drop2";
  if (p < 0.74) return "bridge";
  if (p < 0.94) return "final";
  return "outro";
}

function patternForKind(rng, kind) {
  const families = {
    intro: ["zigzag", "stair"],
    build: ["stair", "zigzag"],
    predrop: ["zigzag", "stair"],
    drop1: ["zigzag", "stair", "random"],
    trans: ["zigzag", "random"],
    groove: ["zigzag", "random"],
    minidrop: ["zigzag", "stair"],
    breakdown: ["zigzag", "random"],
    snareroll: ["trill"],
    drop2: ["stair", "zigzag", "trill"],
    bridge: ["zigzag", "stair"],
    final: ["zigzag", "stair", "random"],
    outro: ["zigzag", "stair"],
  };
  const opts = families[kind] || ["zigzag"];
  return opts[Math.floor(rng() * opts.length)];
}

function getPatternSequence(rng, type) {
  if (type === "stair") return rng() < 0.5 ? STAIR_UP : STAIR_DN;
  if (type === "zigzag") return rng() < 0.5 ? ZIGZAG_A : ZIGZAG_B;
  if (type === "trill") return rng() < 0.5 ? TRILL_L : TRILL_R;
  return null;
}

function laneCooldownWeight(kind, lane, history, barLaneCounts) {
  if (FORBID_SAME_AS_PREV && history[0] === lane) return 0;

  if (HIGH_DENSE_KINDS.has(kind)) {
    const window = history.slice(0, HIGH_DENSE_HARD_WINDOW);
    if (window.includes(lane)) return 0;
  }

  let w = 1.0;

  if (history[1] === lane) w *= 0.12;
  if (history[2] === lane) w *= 0.30;
  if (history[3] === lane) w *= 0.55;
  if (history[4] === lane) w *= 0.75;

  const recent = history.slice(0, 8);
  const count = recent.filter(x => x === lane).length;
  if (count >= 2) w *= 0.35;
  if (count >= 3) w *= 0.15;

  if (barLaneCounts) {
    const c = barLaneCounts.get(lane) || 0;
    const spread = 1 / (1 + c * BAR_SPREAD_STRENGTH);
    w *= spread;
  }

  return w;
}

function pickLaneAvoidingRepeats(rng, kind, history, barLaneCounts, playLanes, bias = null) {
  const items = playLanes.map(lane => {
    let w = laneCooldownWeight(kind, lane, history, barLaneCounts);
    if (bias && bias.centerLane != null) {
      const d = Math.abs(lane - bias.centerLane);
      const b = Math.exp(-d * (bias.strength ?? 0.7));
      w *= b;
    }
    return { value: lane, w };
  });
  return pickWeighted(rng, items);
}

function commitLane(history, lane) {
  history.unshift(lane);
  if (history.length > HISTORY_MAX) history.pop();
}

function uniqueSorted(arr) {
  return [...new Set(arr)].sort((a, b) => a - b);
}

function filterChordsByAllowed(chords, allowedSet) {
  return chords.filter(ch => ch.every(l => allowedSet.has(l)));
}

// -------------------- pleasant chord picker --------------------
function getPleasantChordCandidates(size, allowedPlayLaneSet) {
  if (size === 2) return filterChordsByAllowed(PLEASANT_CHORDS_2, allowedPlayLaneSet);
  if (size === 3) return filterChordsByAllowed(PLEASANT_CHORDS_3, allowedPlayLaneSet);
  return [];
}

// 候補の中から「クールダウン/分散に通る」ものを重み付きで選ぶ
function pickPleasantChord(rng, kind, size, laneHistory, barLaneCounts, allowedPlayLaneSet, biasCenterLane = null) {
  const cands = getPleasantChordCandidates(size, allowedPlayLaneSet);
  if (!cands.length) return null;

  const items = [];
  for (const chord of cands) {
    let w = 1.0;
    for (const lane of chord) {
      const lw = laneCooldownWeight(kind, lane, laneHistory, barLaneCounts);
      if (lw <= 0) { w = 0; break; }
      w *= lw;
    }

    if (w > 0 && biasCenterLane != null) {
      const avg = chord.reduce((s, x) => s + x, 0) / chord.length;
      const d = Math.abs(avg - biasCenterLane);
      w *= Math.exp(-d * 0.45);
    }

    if (w > 0) items.push({ value: chord, w });
  }

  if (!items.length) return null;
  return pickWeighted(rng, items);
}

// フォールバック（従来の“押しやすい形”）
// 2点: [0,3] / 3点: [0,2,4]
function makeChordFromBase(baseLane, size, playLanes) {
  // ★ playLanes の範囲で近いレーンに寄せる（-n 対応）
  const sorted = [...playLanes].sort((a, b) => a - b);

  const nearest = (x) => {
    let best = sorted[0];
    let bestD = Math.abs(sorted[0] - x);
    for (const l of sorted) {
      const d = Math.abs(l - x);
      if (d < bestD) { best = l; bestD = d; }
    }
    return best;
  };

  if (size === 2) {
    // base付近 + なるべく離れた（気持ちよい）相方
    const a = nearest(baseLane);
    const b = nearest(baseLane + 2);
    const chord = uniqueSorted([a, b]);
    return chord.length >= 2 ? chord : uniqueSorted([a, sorted[sorted.length - 1]]);
  }

  if (size === 3) {
    const a = nearest(baseLane);
    const b = nearest(baseLane + 2);
    const c = nearest(baseLane + 4);
    const chord = uniqueSorted([a, b, c]);
    if (chord.length >= 3) return chord;
    // 足りない場合は端も混ぜる
    return uniqueSorted([a, sorted[Math.floor(sorted.length / 2)], sorted[sorted.length - 1]]).slice(0, 3);
  }

  return [nearest(baseLane)];
}

// -------------------- shift notes --------------------
function generateShiftBars(bars, target, rng, mandatoryBarsSet) {
  const picks = new Set();

  if (mandatoryBarsSet && mandatoryBarsSet.size) {
    for (const b of mandatoryBarsSet) {
      if (b >= 0 && b < bars) picks.add(b);
    }
  }

  for (let b = 0; b < bars; b += 16) picks.add(b);
  for (let b = 0; b < bars; b += 4) {
    if (rng() < 0.25) picks.add(b);
  }
  picks.add(0);
  picks.add(bars - 1);

  const heads = [];
  for (let b = 0; b < bars; b += 4) heads.push(b);
  while (picks.size < target && heads.length) {
    const b = heads.splice(Math.floor(rng() * heads.length), 1)[0];
    picks.add(b);
  }

  let arr = [...picks].sort((a, b) => a - b);
  if (arr.length > target) {
    const must = new Set([0, bars - 1]);
    if (mandatoryBarsSet && mandatoryBarsSet.size) {
      for (const b of mandatoryBarsSet) must.add(b);
    }
    const keep = arr.filter(b => must.has(b));
    const rest = arr.filter(b => !must.has(b));

    while (keep.length < target && rest.length) {
      keep.push(rest.splice(Math.floor(rng() * rest.length), 1)[0]);
    }
    arr = keep.sort((a, b) => a - b);
  }
  return arr;
}

// -------------------- bar beat grid --------------------
function beatGridForBar(kind, rng) {
  const beats = [];
  const add8th = () => { for (let i = 0; i < 8; i++) beats.push(i * 0.5); };
  const add16th = () => { for (let i = 0; i < 16; i++) beats.push(i * 0.25); };
  const addSparse8th = () => { for (let i = 0; i < 8; i++) if (i % 2 === 0) beats.push(i * 0.5); };
  const addSnareRoll = () => { for (let i = 0; i < 16; i++) beats.push(2 + i * 0.125); };

  if (kind === "intro") {
    (rng() < 0.7) ? addSparse8th() : add8th();
  } else if (kind === "build") {
    (rng() < 0.5) ? add8th() : add16th();
  } else if (kind === "predrop") {
    add16th(); beats.push(0, 1, 2, 3);
  } else if (kind === "drop1" || kind === "minidrop" || kind === "final" || kind === "drop2") {
    add16th();
    const keep = beats.filter(() => rng() > 0.18);
    beats.length = 0; beats.push(...keep);
    [0, 1, 2, 3].forEach(b => beats.push(b));
  } else if (kind === "trans") {
    (rng() < 0.7) ? add8th() : addSparse8th();
  } else if (kind === "groove") {
    add16th();
    const keep = beats.filter((b, idx) => (idx % 4 !== 1) || rng() < 0.25);
    beats.length = 0; beats.push(...keep);
  } else if (kind === "breakdown") {
    (rng() < 0.7) ? addSparse8th() : add8th();
  } else if (kind === "snareroll") {
    add16th(); addSnareRoll();
  } else if (kind === "bridge" || kind === "outro") {
    add8th();
    const keep = beats.filter(() => rng() > 0.35);
    beats.length = 0; beats.push(...keep);
    [0, 2].forEach(b => beats.push(b));
  } else {
    add8th();
  }

  return uniqueSorted(beats).filter(b => b >= 0 && b < 4);
}

// -------------------- density smoothing --------------------
function enforceBarDensity(events, kind, rng) {
  const preset = DENSITY_PRESETS[kind] || { min: 8, max: 16 };
  const count = events.length;

  if (count > preset.max) {
    const keep = [];
    for (const ev of events) {
      const within = ev.beatInBar;
      const isStrong = (within === 0 || within === 1 || within === 2 || within === 3);
      if (isStrong) keep.push(ev);
      else if (rng() < preset.max / count) keep.push(ev);
    }
    while (keep.length > preset.max) keep.splice(Math.floor(rng() * keep.length), 1);
    return keep.sort((a, b) => a.beatInBar - b.beatInBar);
  }

  if (count < preset.min) {
    const pool = [];
    for (let i = 0; i < 16; i++) pool.push(i * 0.25);
    const used = new Set(events.map(e => e.beatInBar));
    const cand = pool.filter(b => !used.has(b));
    const fillN = preset.min - count;
    for (let i = 0; i < fillN && cand.length; i++) {
      const b = cand.splice(Math.floor(rng() * cand.length), 1)[0];
      events.push({ beatInBar: b, type: "fill" });
    }
    return events.sort((a, b) => a.beatInBar - b.beatInBar);
  }

  return events;
}

// ---- 乱打小節用：16分に寄せたグリッド（基本16分 + 少し間引き無し） ----
function beatGridForStream16() {
  const beats = [];
  for (let i = 0; i < 16; i++) beats.push(i * 0.25);
  return beats; // 0..3.75
}

// ---- 小節が「16分乱打モード」か判定 ----
function isStream16Bar(rng, kind) {
  if (!STREAM16_KINDS.has(kind)) return false;
  const p = STREAM16_BAR_PROB[kind] ?? 0;
  return rng() < p;
}

// ---- 同時押しを出すか（乱打小節では極力出さない） ----
function shouldUseChord(rng, kind, isStrong, isStreamBar) {
  if (isStreamBar) return false;
  const base = CHORD_PROB[kind] ?? 0.06;
  const p = base + (isStrong ? STRONG_BEAT_CHORD_BONUS : 0);
  return rng() < p;
}

// -------------------- main generation --------------------
function generateChartFromAnalysis(analysis, opts = {}) {
  const bpm = Number(analysis.bpm);
  const bars = Number(analysis.bars);

  if (!Number.isFinite(bpm) || bpm <= 0) throw new Error("analysis.bpm must be a positive number");
  if (!Number.isFinite(bars) || bars <= 0) throw new Error("analysis.bars must be a positive number");

  const timeSignature = Array.isArray(analysis.timeSignature) ? analysis.timeSignature : [4, 4];
  if (timeSignature[0] !== 4) throw new Error("This generator currently assumes 4/4 (timeSignature[0] must be 4).");

  const normalTarget = Number.isFinite(Number(analysis.normalNotesTarget))
    ? Number(analysis.normalNotesTarget)
    : DEFAULT_NORMAL_TARGET;

  let shiftTarget = Math.max(1, Math.round(normalTarget / 25));

  const seed = Number.isFinite(Number(analysis.seed)) ? Number(analysis.seed) : DEFAULT_SEED;
  const rng = mulberry32(seed);

  const narrow = !!opts.narrowLanes;

  // ★ -n のときだけ 1,3,5,7 に制限
  const PLAY_LANES = narrow ? [...NARROW_PLAY_LANES] : [1, 2, 3, 4, 5, 6, 7];
  const PLAY_LANE_SET = new Set(PLAY_LANES);

  const sectionsRaw = Array.isArray(analysis.sections) ? analysis.sections : [];
  const sections = sectionsRaw
    .map(s => ({
      fromBar: clamp(Number(s.fromBar ?? 0) | 0, 0, bars - 1),
      toBar: clamp(Number(s.toBar ?? 0) | 0, 0, bars - 1),
      kind: normalizeKind(s.kind),
      kick: normalizeKick(s.kick),
    }))
    .map(s => (s.toBar < s.fromBar ? { ...s, fromBar: s.toBar, toBar: s.fromBar } : s))
    .sort((a, b) => a.fromBar - b.fromBar);

  // ★構成変化点（境目）は必須でShiftを置く
  const mandatoryShiftBars = new Set();
  for (const s of sections) mandatoryShiftBars.add(s.fromBar);
  mandatoryShiftBars.add(0);
  mandatoryShiftBars.add(bars - 1);

  if (mandatoryShiftBars.size > shiftTarget) shiftTarget = mandatoryShiftBars.size;

  const laneHistory = [];

  let patType = "zigzag";
  let patSeq = getPatternSequence(rng, patType) || ZIGZAG_A;
  let patIndex = 0;

  // ★ -n のときはパターンが偶数レーンを引くので、基本は random 運用に寄せる
  if (narrow) {
    patType = "random";
    patSeq = null;
    patIndex = 0;
  }

  const notes = [];

  // shift notes
  const shiftBars = generateShiftBars(bars, shiftTarget, rng, mandatoryShiftBars);
  for (const b of shiftBars) notes.push({ lane: 0, beat: b * 4 });

  function refreshPattern(kind) {
    if (narrow) {
      // -n: 偶数レーンが混ざるパターンを避けるため random固定
      patType = "random";
      patSeq = null;
      patIndex = 0;
      return;
    }

    const newType = patternForKind(rng, kind);
    if (newType !== patType || rng() < 0.25) {
      patType = newType;
      patSeq = getPatternSequence(rng, patType) || null;
      patIndex = 0;
    }
  }

  function nextLane(kind, barLaneCounts) {
    if (patType === "random" || !patSeq) {
      return pickLaneAvoidingRepeats(rng, kind, laneHistory, barLaneCounts, PLAY_LANES);
    }

    const raw = patSeq[patIndex % patSeq.length];
    patIndex++;

    // パターンが許可外レーンを返したら即フォールバック
    if (!PLAY_LANE_SET.has(raw)) {
      return pickLaneAvoidingRepeats(rng, kind, laneHistory, barLaneCounts, PLAY_LANES);
    }

    if (FORBID_SAME_AS_PREV && laneHistory[0] === raw) {
      return pickLaneAvoidingRepeats(rng, kind, laneHistory, barLaneCounts, PLAY_LANES, { centerLane: raw, strength: 0.6 });
    }

    const w = laneCooldownWeight(kind, raw, laneHistory, barLaneCounts);
    if (w < 0.25) {
      return pickLaneAvoidingRepeats(rng, kind, laneHistory, barLaneCounts, PLAY_LANES, { centerLane: raw, strength: 0.7 });
    }
    return raw;
  }

  // generate bar-by-bar
  for (let bar = 0; bar < bars; bar++) {
    const sec = sectionForBar(sections, bar);
    const kind = classifyKind(bar, sec, bars);

    // ★同時押し多すぎ対策：構成によっては「16分乱打小節」に切り替える
    const streamBar = isStream16Bar(rng, kind);

    // 乱打小節はパターンをリフレッシュしつつ「random寄り」になりやすくする
    if (streamBar) {
      refreshPattern(kind);
      if (!narrow && rng() < 0.55) { // 乱打中は少しランダム化（-nは常にrandom）
        patType = "random";
        patSeq = null;
        patIndex = 0;
      }
    } else {
      refreshPattern(kind);
    }

    const grid = streamBar ? beatGridForStream16() : beatGridForBar(kind, rng);

    let barEvents = grid.map(b => ({ beatInBar: b, type: streamBar ? "stream16" : "grid" }));
    barEvents = enforceBarDensity(barEvents, kind, rng);

    const barLaneCounts = new Map();

    for (const ev of barEvents) {
      const bInBar = ev.beatInBar;
      const absBeat = bar * 4 + bInBar;
      const isStrong = (bInBar === 0 || bInBar === 1 || bInBar === 2 || bInBar === 3);

      // snareroll は単押し固定
      if (kind === "snareroll") {
        const lane = nextLane(kind, barLaneCounts);
        notes.push({ lane, beat: absBeat });
        commitLane(laneHistory, lane);
        barLaneCounts.set(lane, (barLaneCounts.get(lane) || 0) + 1);
        continue;
      }

      const useChord = shouldUseChord(rng, kind, isStrong, streamBar);

      if (useChord) {
        // 2/3同時は「気持ちいい形」優先（比率を落とすため 3同時はさらに低確率）
        const chordSize = (rng() < 0.25) ? 3 : 2;

        const baseHint = nextLane(kind, barLaneCounts);

        let chord = pickPleasantChord(
          rng,
          kind,
          chordSize,
          laneHistory,
          barLaneCounts,
          PLAY_LANE_SET,
          baseHint
        );

        if (!chord) chord = makeChordFromBase(baseHint, chordSize, PLAY_LANES);

        chord = uniqueSorted(chord).filter(l => PLAY_LANE_SET.has(l));

        if (chord.length < 2) {
          const lane = pickLaneAvoidingRepeats(
            rng, kind, laneHistory, barLaneCounts, PLAY_LANES,
            { centerLane: baseHint, strength: 0.8 }
          );
          notes.push({ lane, beat: absBeat });
          commitLane(laneHistory, lane);
          barLaneCounts.set(lane, (barLaneCounts.get(lane) || 0) + 1);
          continue;
        }

        for (const lane of chord) {
          let useLane = lane;
          if (laneCooldownWeight(kind, useLane, laneHistory, barLaneCounts) === 0) {
            useLane = pickLaneAvoidingRepeats(
              rng, kind, laneHistory, barLaneCounts, PLAY_LANES,
              { centerLane: baseHint, strength: 0.85 }
            );
          }
          // 念のため（-n で絶対に許可外に行かないように）
          if (!PLAY_LANE_SET.has(useLane)) {
            useLane = pickLaneAvoidingRepeats(rng, kind, laneHistory, barLaneCounts, PLAY_LANES);
          }

          notes.push({ lane: useLane, beat: absBeat });
          commitLane(laneHistory, useLane);
          barLaneCounts.set(useLane, (barLaneCounts.get(useLane) || 0) + 1);
        }
      } else {
        // 単押し（16分乱打小節はここが主役）
        const lane = nextLane(kind, barLaneCounts);
        notes.push({ lane, beat: absBeat });
        commitLane(laneHistory, lane);
        barLaneCounts.set(lane, (barLaneCounts.get(lane) || 0) + 1);
      }
    }
  }

  // -------------------- adjust total counts --------------------
  const shiftNotes = notes.filter(n => n.lane === 0);
  let normalNotes = notes.filter(n => n.lane !== 0);

  // ★ -n: 念のため最終フィルタ（これで絶対に 1,3,5,7 以外が残らない）
  if (narrow) {
    normalNotes = normalNotes.filter(n => PLAY_LANE_SET.has(n.lane));
  }

  if (normalNotes.length > normalTarget) {
    const need = normalTarget;
    const scored = normalNotes.map(n => {
      const bInBar = n.beat % 4;
      const isStrong = (bInBar === 0 || bInBar === 1 || bInBar === 2 || bInBar === 3);
      const w = (isStrong ? 3 : 1) + (rng() * 0.25);
      return { n, w };
    });
    scored.sort((a, b) => b.w - a.w);
    normalNotes = scored.slice(0, need).map(x => x.n);
  }

  if (normalNotes.length < normalTarget) {
    const needAdd = normalTarget - normalNotes.length;

    const byBeat = new Map();
    for (const n of normalNotes) {
      const k = n.beat.toFixed(3);
      if (!byBeat.has(k)) byBeat.set(k, []);
      byBeat.get(k).push(n.lane);
    }

    const candidates = [];
    for (let bar = 0; bar < bars; bar++) {
      const sec = sectionForBar(sections, bar);
      const kind = classifyKind(bar, sec, bars);
      const fillBias = (kind === "intro" || kind === "outro" || kind === "breakdown") ? 0.25 : 1.0;

      for (let i = 0; i < 16; i++) {
        const bInBar = i * 0.25;
        const absBeat = bar * 4 + bInBar;
        const key = absBeat.toFixed(3);
        if (byBeat.has(key)) continue;
        candidates.push({ absBeat, fillBias, kind });
      }
    }

    for (let i = candidates.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
    }

    const localHistory = [];
    const localBarLaneCounts = new Map();
    for (let i = 0, added = 0; i < candidates.length && added < needAdd; i++) {
      const c = candidates[i];
      if (rng() > c.fillBias) continue;

      const lane = pickLaneAvoidingRepeats(rng, c.kind, localHistory, localBarLaneCounts, PLAY_LANES);
      normalNotes.push({ lane, beat: c.absBeat });
      commitLane(localHistory, lane);
      localBarLaneCounts.set(lane, (localBarLaneCounts.get(lane) || 0) + 1);
      added++;
    }

    // ★ -n: 追加分も最終フィルタ
    if (narrow) {
      normalNotes = normalNotes.filter(n => PLAY_LANE_SET.has(n.lane));
    }
  }

  const outNotes = [...shiftNotes, ...normalNotes].sort((a, b) => a.beat - b.beat);

  return {
    title: analysis.title || "Generated",
    artist: analysis.artist || "Generated",
    audio: analysis.audio || "song.wav",
    bpm,
    timeSignature,
    bars,
    offsetSec: Number.isFinite(Number(analysis.offsetSec)) ? Number(analysis.offsetSec) : 0,
    lanes: LANES,
    notes: outNotes,
    normalNotesTarget: normalTarget,
    shiftNotesTarget: shiftTarget,
    seed,
  };
}

// -------------------- main --------------------
function parseArgs(argv) {
  const args = argv.slice(2);
  const flags = new Set();
  const positionals = [];

  for (const a of args) {
    if (a === "-n") flags.add("n");
    else positionals.push(a);
  }

  return {
    narrow: flags.has("n"),
    path: positionals[0] || null,
  };
}

function main() {
  const { narrow, path } = parseArgs(process.argv);

  if (!path) {
    console.error("Usage: node genChartFromAnalysis.js [-n] analysis.json > chart.json");
    process.exit(1);
  }

  const analysis = JSON.parse(fs.readFileSync(path, "utf8"));
  const chart = generateChartFromAnalysis(analysis, { narrowLanes: narrow });
  process.stdout.write(JSON.stringify(chart, null, 2));
}

if (require.main === module) main();
