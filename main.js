'use strict';

// ============================================================
//  ハイジャンプ選手権（カメラ版）
//  足ぶみダッシュで助走 → 本当にジャンプして踏み切り → バンザイで空中ジャンプ
//  カメラが使えないときはスペースキー / タップでも遊べる
// ============================================================

const $ = (id) => document.getElementById(id);
const canvas = $('game');
const ctx = canvas.getContext('2d');
const video = $('cam');

// ---------- 定数 ----------
const GRAVITY = 12.8;        // m/s^2（ゲーム用）
const TIME_SCALE = 3;        // 空中の時間の早回し
const RUN_TIME = 4.0;        // 助走の秒数
const TAP_GAIN = 6;          // キーボード1連打あたりのスピード
const STEP_GAIN = 6;         // 足ぶみの動き（胴の長さ1つぶん）あたりのスピード
const TAKEOFF_LIMIT = 4.5;   // 踏み切りの制限時間
const LIFT_THRESHOLD = 0.2;  // 腰がこれだけ（胴の長さ比）上がったらジャンプと判定
const TORSO_CM = 50;         // 胴（肩〜腰）の長さの目安。リアルジャンプの cm 表示用
const RING_TIME = 1.6;       // 空中リングが縮む秒数
const RING_START = 130;
const RING_TARGET = 36;
const HANDS_UP_START = 1.0;  // 両手を上げ続けるとスタートする秒数
const PLAYER_COLORS = ['#ff5a36', '#2f80ed', '#27ae60', '#9b51e0'];
const RANK_KEY = 'highjump.ranking.v2';
const NAME_KEY = 'highjump.names.v1';

const LANDMARKS = [
  { h: 8,    name: '一軒家',       x: 0.12, w: 90,  color: '#e59866', kind: 'house' },
  { h: 25,   name: '大きな木',     x: 0.86, w: 70,  color: '#2e8b57', kind: 'tree' },
  { h: 100,  name: '観覧車',       x: 0.10, w: 160, color: '#d35400', kind: 'wheel' },
  { h: 200,  name: '高層ビル',     x: 0.88, w: 90,  color: '#5d6d7e', kind: 'building' },
  { h: 333,  name: '東京タワー',   x: 0.14, w: 120, color: '#e74c3c', kind: 'tower' },
  { h: 634,  name: 'スカイツリー', x: 0.86, w: 90,  color: '#aab7c4', kind: 'tower' },
  { h: 1000, name: '雲の上',       x: 0.50, w: 0,   color: '#fff',    kind: 'none' },
  { h: 1500, name: '熱気球',       x: 0.20, w: 70,  color: '#f1c40f', kind: 'balloon' },
  { h: 2500, name: 'ジェット機',   x: 0.80, w: 90,  color: '#ecf0f1', kind: 'plane' },
  { h: 3776, name: '富士山',       x: 0.50, w: 0,   color: '#fff',    kind: 'none' },
];

// ---------- 状態 ----------
let W = 0, H = 0, DPR = 1;
let state = 'title';
let players = [];
let rounds = 3;
let turn = { round: 0, idx: 0 };
let t = 0;             // 状態内の経過時間
let lastTime = 0;

const jump = {
  speed: 0, runTime: 0, runPhase: 0, stepAcc: 0,
  marker: 0, markerDir: 1, sweep: 1,
  mult: 1, takeoffLabel: '', realJ: 0, jumpMult: 1, liftMin: 0, liftT: 0, byKey: false,
  py: 0, vy: 0, v0: 0, maxH: 0,
  ringUsed: false, ringActive: false, ringT: 0, ringLabel: '', ringQ: 0, armsDownSeen: false,
  passed: new Set(),
  camH: 0,
  particles: [],
};

let clouds = [];
let stars = [];

// ============================================================
//  カメラ・姿勢検出
// ============================================================
const pose = {
  status: 'loading',   // loading / ok / error
  message: 'カメラを準備中…',
  landmarker: null,
  lastVideoTime: -1,
  lm: null,            // 最新のランドマーク
  seen: false,         // 人が写っているか
  body: false,         // 肩と腰が写っているか
  hipY: 0, torso: 0,
  baseHip: 0, baseTorso: 0.25,
  knees: null,
  motion: 0,           // この1フレームの足ぶみ量（胴の長さ比）
  handsUp: false, handsUpT: 0,
  history: [],         // 直近の腰の高さ
};

async function initCamera() {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error('nomedia');
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
    });
    video.srcObject = stream;
    await new Promise((res) => { video.onloadedmetadata = res; });
    await video.play();
    pose.message = 'AIモデルを読み込み中…';
    updateCamStatus();
    const { FilesetResolver, PoseLandmarker } = await import('./vendor/mediapipe/vision_bundle.mjs');
    const vision = await FilesetResolver.forVisionTasks('./vendor/mediapipe/wasm');
    pose.landmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: './vendor/models/pose_landmarker_lite.task', delegate: 'GPU' },
      runningMode: 'VIDEO', numPoses: 1,
    });
    pose.status = 'ok';
    pose.message = 'カメラOK！';
  } catch (e) {
    console.error(e);
    pose.status = 'error';
    pose.message = e && e.name === 'NotAllowedError'
      ? 'カメラが許可されませんでした（キーボード / タップで遊べます）'
      : 'カメラが使えません（キーボード / タップで遊べます）';
  }
  updateCamStatus();
}
function updateCamStatus() {
  const el = $('cam-status');
  el.textContent = pose.message;
  el.className = `cam-status ${pose.status}`;
}
const camOn = () => pose.status === 'ok';

function detectPose(now) {
  pose.motion = 0;
  if (!camOn() || video.readyState < 2 || video.currentTime === pose.lastVideoTime) return;
  pose.lastVideoTime = video.currentTime;
  let res;
  try { res = pose.landmarker.detectForVideo(video, now); } catch (e) { return; }
  const lm = res && res.landmarks && res.landmarks[0];
  pose.lm = lm || null;
  pose.seen = !!lm;
  if (!lm) { pose.body = false; pose.handsUp = false; pose.knees = null; return; }

  const vis = (i) => (lm[i].visibility ?? 1) > 0.5;
  pose.body = vis(11) && vis(12) && vis(23) && vis(24);
  const shY = (lm[11].y + lm[12].y) / 2;
  pose.hipY = (lm[23].y + lm[24].y) / 2;
  pose.torso = Math.max(0.05, pose.hipY - shY);
  pose.history.push({ t: now, hip: pose.hipY });
  while (pose.history.length && now - pose.history[0].t > 1500) pose.history.shift();

  // 両手が頭より上 = バンザイ
  pose.handsUp = vis(15) && vis(16) && lm[15].y < lm[0].y && lm[16].y < lm[0].y;

  // 足ぶみ量：ひざの上下の動き（見えないときは手の振り）
  const unit = pose.baseTorso || pose.torso;
  const kneesVisible = (lm[25].visibility ?? 1) > 0.4 && (lm[26].visibility ?? 1) > 0.4;
  const cur = kneesVisible
    ? { a: lm[25].y, b: lm[26].y, w: 1 }
    : { a: lm[15].y, b: lm[16].y, w: 0.6 };
  if (pose.knees && pose.knees.w === cur.w) {
    const d = (Math.abs(cur.a - pose.knees.a) + Math.abs(cur.b - pose.knees.b)) / unit;
    pose.motion = clamp(d - 0.02, 0, 0.5) * cur.w; // 小さなブレは無視
  }
  pose.knees = cur;
}

// 立っている姿勢を基準として覚える
function calibrate(dt) {
  if (!pose.body) return;
  const k = 1 - Math.pow(0.02, dt);
  if (!pose.baseHip) { pose.baseHip = pose.hipY; pose.baseTorso = pose.torso; return; }
  pose.baseHip = lerp(pose.baseHip, pose.hipY, k);
  pose.baseTorso = lerp(pose.baseTorso, pose.torso, k);
}
function recentHipMedian(ms) {
  const now = performance.now();
  const v = pose.history.filter((h) => now - h.t <= ms).map((h) => h.hip).sort((a, b) => a - b);
  return v.length ? v[Math.floor(v.length / 2)] : pose.baseHip;
}

// ============================================================
//  サウンド（WebAudio のみ）
// ============================================================
let actx = null;
function audio() {
  if (!actx) {
    try { actx = new (window.AudioContext || window.webkitAudioContext)(); } catch (e) { actx = null; }
  }
  if (actx && actx.state === 'suspended') actx.resume();
  return actx;
}
function tone(freq, dur, type = 'square', vol = 0.08, slideTo = null) {
  const a = audio(); if (!a) return;
  const o = a.createOscillator(); const g = a.createGain();
  o.type = type; o.frequency.setValueAtTime(freq, a.currentTime);
  if (slideTo) o.frequency.exponentialRampToValueAtTime(slideTo, a.currentTime + dur);
  g.gain.setValueAtTime(vol, a.currentTime);
  g.gain.exponentialRampToValueAtTime(0.0001, a.currentTime + dur);
  o.connect(g); g.connect(a.destination);
  o.start(); o.stop(a.currentTime + dur);
}
const sfx = {
  tap: (s) => tone(300 + s * 6, 0.05, 'square', 0.05),
  count: () => tone(660, 0.15, 'sine', 0.12),
  go: () => tone(990, 0.35, 'sine', 0.14),
  jump: () => tone(250, 0.5, 'sawtooth', 0.08, 1200),
  good: () => { tone(880, 0.12, 'triangle', 0.1); setTimeout(() => tone(1320, 0.2, 'triangle', 0.1), 90); },
  miss: () => tone(200, 0.3, 'square', 0.07, 100),
  pass: () => tone(1568, 0.15, 'triangle', 0.08),
  fanfare: () => [523, 659, 784, 1047].forEach((f, i) => setTimeout(() => tone(f, 0.25, 'triangle', 0.1), i * 110)),
};

// ============================================================
//  ユーティリティ
// ============================================================
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, k) => a + (b - a) * k;
const fmt = (h) => h.toFixed(1);

function resize() {
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  W = window.innerWidth; H = window.innerHeight;
  canvas.width = W * DPR; canvas.height = H * DPR;
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
}
window.addEventListener('resize', resize);

function showScreen(id) {
  document.querySelectorAll('.screen').forEach((el) => el.classList.add('hidden'));
  if (id) $(id).classList.remove('hidden');
}
function show(id, on) { $(id).classList.toggle('hidden', !on); }

function setPrompt(text, pulse = false) {
  const el = $('prompt');
  el.textContent = text;
  el.classList.toggle('pulse', pulse);
  show('prompt', !!text);
}
function setGuide(text) {
  $('guide').textContent = text;
  show('guide', !!text);
}
function popup(text) {
  const el = $('popup');
  el.classList.add('hidden');
  void el.offsetWidth; // アニメーションをリセット
  el.textContent = text;
  el.classList.remove('hidden');
}

function makeWorld() {
  clouds = [];
  for (let i = 0; i < 110; i++) {
    clouds.push({ h: 60 + Math.random() * 3200, x: Math.random(), s: 0.6 + Math.random() * 1.2 });
  }
  stars = [];
  for (let i = 0; i < 160; i++) stars.push({ x: Math.random(), y: Math.random(), r: Math.random() * 1.5 + 0.3 });
}

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen?.().catch(() => {});
}

// ============================================================
//  ランキング
// ============================================================
function loadRanking() {
  try { return JSON.parse(localStorage.getItem(RANK_KEY)) || []; } catch (e) { return []; }
}
function saveRanking(list) {
  try { localStorage.setItem(RANK_KEY, JSON.stringify(list)); } catch (e) { /* 保存できなくても続行 */ }
}
function addToRanking(name, h) {
  const list = loadRanking();
  const entry = { name, h, d: new Date().toLocaleDateString('ja-JP'), id: Math.random() };
  list.push(entry);
  list.sort((a, b) => b.h - a.h);
  const top = list.slice(0, 10);
  saveRanking(top);
  const pos = top.indexOf(entry);
  return pos >= 0 ? pos + 1 : 0;
}
function allTimeBest() {
  const list = loadRanking();
  return list.length ? list[0] : null;
}

// ============================================================
//  タイトル画面
// ============================================================
let playerCount = 1;
function loadNames() {
  try { return JSON.parse(localStorage.getItem(NAME_KEY)) || []; } catch (e) { return []; }
}
function renderNameInputs() {
  const saved = loadNames();
  const box = $('names');
  const current = [...box.querySelectorAll('input')].map((i) => i.value);
  box.innerHTML = '';
  for (let i = 0; i < playerCount; i++) {
    const inp = document.createElement('input');
    inp.maxLength = 8;
    inp.placeholder = `プレイヤー${i + 1}`;
    inp.value = current[i] ?? saved[i] ?? '';
    inp.style.borderColor = PLAYER_COLORS[i];
    box.appendChild(inp);
  }
}
function segSelect(groupId, cb) {
  $(groupId).addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    $(groupId).querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b));
    cb(Number(b.dataset.n));
  });
}
segSelect('count-btns', (n) => { playerCount = n; renderNameInputs(); });
segSelect('round-btns', (n) => { rounds = n; });

$('start-btn').addEventListener('click', () => {
  audio();
  const names = [...$('names').querySelectorAll('input')].map((inp, i) => inp.value.trim() || `プレイヤー${i + 1}`);
  try { localStorage.setItem(NAME_KEY, JSON.stringify(names)); } catch (e) { /* 無視 */ }
  players = names.map((name, i) => ({ name, color: PLAYER_COLORS[i], jumps: [], best: 0 }));
  turn = { round: 0, idx: 0 };
  goReady();
});
$('rank-btn').addEventListener('click', () => { renderRanking(); showScreen('ranking'); });
$('rank-back').addEventListener('click', () => showScreen('title'));
$('rank-clear').addEventListener('click', () => {
  if (confirm('ランキングを全部消しますか？')) { saveRanking([]); renderRanking(); }
});
$('again-btn').addEventListener('click', () => {
  players.forEach((p) => { p.jumps = []; p.best = 0; });
  turn = { round: 0, idx: 0 };
  goReady();
});
$('title-btn').addEventListener('click', () => { state = 'title'; resetJump(); showScreen('title'); });
$('fs-btn').addEventListener('click', toggleFullscreen);

function renderRanking() {
  const list = loadRanking();
  const ol = $('rank-list');
  ol.innerHTML = '';
  if (!list.length) { ol.innerHTML = '<li class="empty">まだ記録がありません</li>'; return; }
  list.forEach((r, i) => {
    const li = document.createElement('li');
    li.innerHTML = `<span class="pos">${medal(i + 1)}</span><span class="nm"></span><span class="dt">${r.d}</span><span class="h">${fmt(r.h)}m</span>`;
    li.querySelector('.nm').textContent = r.name;
    ol.appendChild(li);
  });
}
function medal(n) { return ['🥇', '🥈', '🥉'][n - 1] || n; }

// ============================================================
//  ターン進行
// ============================================================
function currentPlayer() { return players[turn.idx]; }

function goReady() {
  state = 'ready'; t = 0;
  resetJump();
  hideGameUI();
  pose.handsUpT = 0;
  const p = currentPlayer();
  $('ready-round').textContent = rounds > 1 ? `${turn.round + 1} / ${rounds} 回目` : '1回勝負';
  $('ready-name').textContent = p.name;
  $('ready-name').style.color = p.color;
  $('ready-best').textContent = p.best > 0 ? `自己ベスト ${fmt(p.best)}m` : '';
  $('ready-cam').classList.toggle('hidden', !camOn());
  showScreen('ready');
}
$('ready-btn').addEventListener('click', () => { audio(); beginTurn(); });

function beginTurn() {
  if (state !== 'ready') return;
  document.activeElement && document.activeElement.blur();
  showScreen(null);
  startCountdown();
}

function resetJump() {
  Object.assign(jump, {
    speed: 0, runTime: 0, runPhase: 0, stepAcc: 0,
    marker: 0, markerDir: 1, sweep: 1,
    mult: 1, takeoffLabel: '', realJ: 0, jumpMult: 1, liftMin: 0, liftT: 0, byKey: false,
    py: 0, vy: 0, v0: 0, maxH: 0,
    ringUsed: false, ringActive: false, ringT: 0, ringLabel: '', ringQ: 0, armsDownSeen: false,
    passed: new Set(), camH: 0, particles: [],
  });
}

function hideGameUI() {
  ['hud', 'speed-wrap', 'timing-wrap', 'prompt', 'popup', 'guide'].forEach((id) => show(id, false));
}

function startCountdown() {
  state = 'countdown'; t = 0;
  const p = currentPlayer();
  $('hud-player').textContent = p.name;
  $('hud-height').innerHTML = '0.0<span>m</span>';
  const best = allTimeBest();
  $('hud-best').textContent = best ? `歴代1位 ${best.name} ${fmt(best.h)}m` : '';
  show('hud', true);
  show('speed-wrap', true);
  updateSpeedUI();
  setPrompt('3');
  setGuide(camOn() ? 'まっすぐ立って…' : '');
  sfx.count();
}

function updateSpeedUI() {
  $('speed-fill').style.width = `${jump.speed}%`;
  $('speed-time').textContent = state === 'run' ? `のこり ${Math.max(0, RUN_TIME - jump.runTime).toFixed(1)}秒` : '';
}

// ============================================================
//  入力（キーボード / タップ。カメラが無いとき用）
// ============================================================
function press() {
  switch (state) {
    case 'ready': beginTurn(); break;
    case 'run': {
      jump.speed = Math.min(100, jump.speed + TAP_GAIN);
      sfx.tap(jump.speed);
      for (let i = 0; i < 2; i++) spawnDust();
      break;
    }
    case 'takeoff': judgeTakeoff(false); jump.byKey = true; launch(0.5); break;
    case 'flight': if (jump.ringActive) doRing(false); break;
    default: break;
  }
}
window.addEventListener('keydown', (e) => {
  if (e.code === 'KeyF' && !(e.target instanceof HTMLInputElement)) { toggleFullscreen(); return; }
  if (e.code === 'Space' || e.code === 'Enter') {
    if (['ready', 'run', 'takeoff', 'flight', 'countdown', 'apex'].includes(state)) {
      e.preventDefault();
      if (!e.repeat) press();
    }
  }
});
canvas.addEventListener('pointerdown', (e) => { e.preventDefault(); audio(); press(); });

// ============================================================
//  各フェーズ
// ============================================================
function judgeTakeoff(timeout) {
  const d = Math.abs(jump.marker - 0.5); // 0 = ど真ん中
  let mult, label;
  if (timeout) { mult = 0.6; label = 'おそい…'; }
  else if (d <= 0.06) { mult = 1.6; label = 'PERFECT!!'; }
  else if (d <= 0.15) { mult = 1.3; label = 'GREAT!'; }
  else if (d <= 0.28) { mult = 1.0; label = 'GOOD'; }
  else { mult = 0.6; label = 'MISS…'; }
  jump.mult = mult; jump.takeoffLabel = label;
  mult >= 1.3 ? sfx.good() : mult < 1 ? sfx.miss() : null;
  popup(label);
  show('timing-wrap', false);
  setPrompt('');
}

// realJ = 本当のジャンプで腰が上がった量（胴の長さ比）
function launch(realJ) {
  jump.realJ = realJ;
  jump.jumpMult = clamp(0.5 + realJ, 0.6, 1.6);
  sfx.jump();
  setGuide('');
  jump.v0 = (10 + 0.9 * jump.speed) * jump.mult * jump.jumpMult;
  jump.vy = jump.v0;
  jump.py = 0;
  state = 'flight'; t = 0;
  for (let i = 0; i < 20; i++) spawnDust(true);
}

function startRing() {
  jump.ringUsed = true; jump.ringActive = true; jump.ringT = 0;
  jump.armsDownSeen = !pose.handsUp;
  setPrompt(camOn() ? 'バンザイで空中ジャンプ！' : 'いまだ！', true);
}

function doRing(timeout) {
  jump.ringActive = false;
  setPrompt('');
  setGuide('');
  const r = ringRadius();
  const diff = Math.abs(r - RING_TARGET);
  let q = 0, label = '';
  if (timeout) { q = 0; label = 'のがした…'; }
  else if (diff < 10) { q = 1; label = '空中 PERFECT!!'; }
  else if (diff < 24) { q = 0.7; label = '空中 GREAT!'; }
  else if (diff < 40) { q = 0.4; label = '空中 GOOD'; }
  else { q = 0; label = 'はやすぎ…'; }
  jump.ringQ = q; jump.ringLabel = label;
  if (q > 0) {
    jump.vy += jump.v0 * 0.4 * q;
    q >= 0.7 ? sfx.good() : sfx.pass();
    sfx.jump();
    for (let i = 0; i < 24; i++) spawnSpark();
  } else sfx.miss();
  popup(label);
}
function ringRadius() { return lerp(RING_START, 0, clamp(jump.ringT / RING_TIME, 0, 1)); }

function finishJump() {
  const p = currentPlayer();
  const h = jump.maxH;
  const isNew = h > p.best;
  p.jumps.push(h);
  p.best = Math.max(p.best, h);

  const passed = LANDMARKS.filter((l) => h >= l.h);
  const next = LANDMARKS.find((l) => h < l.h);
  $('jr-name').textContent = p.name;
  $('jr-name').style.color = p.color;
  $('jr-height').innerHTML = `${fmt(h)}<span>m</span>`;
  const real = jump.byKey || !camOn() ? '' : `リアルジャンプ <b>約${Math.round(jump.realJ * TORSO_CM)}cm</b> ／ `;
  $('jr-detail').innerHTML =
    `助走スピード <b>${Math.round(jump.speed)}</b> ／ 踏み切り <b>${jump.takeoffLabel}</b><br>` +
    `${real}空中ジャンプ <b>${jump.ringLabel || 'なし'}</b>`;
  $('jr-landmark').textContent =
    (passed.length ? `${passed[passed.length - 1].name}を超えた！` : '') +
    (next ? ` あと${fmt(next.h - h)}mで${next.name}` : ' 最高到達！');
  show('jr-new', isNew && rounds > 1 && p.jumps.length > 1);

  const last = turn.round === rounds - 1 && turn.idx === players.length - 1;
  $('jr-next').textContent = last ? 'けっか発表へ' : 'つぎへ';
  hideGameUI();
  showScreen('jump-result');
  state = 'jresult';
  sfx.fanfare();
}

$('jr-next').addEventListener('click', () => {
  turn.idx++;
  if (turn.idx >= players.length) { turn.idx = 0; turn.round++; }
  if (turn.round >= rounds) showFinal();
  else goReady();
});

function showFinal() {
  state = 'final';
  resetJump();
  const sorted = [...players].sort((a, b) => b.best - a.best);
  const ol = $('final-list');
  ol.innerHTML = '';
  sorted.forEach((p, i) => {
    const li = document.createElement('li');
    if (i === 0) li.classList.add('first');
    li.innerHTML = `<span class="pos">${medal(i + 1)}</span><span class="nm"></span><span class="h">${fmt(p.best)}m</span>`;
    li.querySelector('.nm').textContent = p.name;
    li.querySelector('.nm').style.color = p.color;
    ol.appendChild(li);
  });
  const msgs = [];
  players.forEach((p) => {
    const pos = addToRanking(p.name, p.best);
    if (pos) msgs.push(`${p.name} が歴代${pos}位にランクイン！`);
  });
  $('final-rank-msg').innerHTML = msgs.map((m) => m.replace(/</g, '&lt;')).join('<br>');
  showScreen('final');
  sfx.fanfare();
}

// ============================================================
//  パーティクル
// ============================================================
function spawnDust(big = false) {
  jump.particles.push({
    x: W / 2 + (Math.random() - 0.5) * 30, y: 0, world: true,
    vx: (Math.random() - 0.5) * (big ? 300 : 120) - (big ? 0 : 80), vy: -Math.random() * (big ? 200 : 80),
    life: 0.6, max: 0.6, color: '#c8a27a', r: big ? 7 : 4,
  });
}
function spawnSpark() {
  const a = Math.random() * Math.PI * 2, s = 100 + Math.random() * 200;
  jump.particles.push({
    x: W / 2, y: H * 0.45, world: false,
    vx: Math.cos(a) * s, vy: Math.sin(a) * s,
    life: 0.7, max: 0.7, color: ['#ffe14d', '#ff5a36', '#fff'][Math.floor(Math.random() * 3)], r: 4,
  });
}

// ============================================================
//  更新
// ============================================================
function update(dt) {
  t += dt;
  switch (state) {
    case 'ready': {
      calibrate(dt);
      // 両手を上げ続けるとスタート
      if (camOn() && pose.body && pose.handsUp) pose.handsUpT += dt; else pose.handsUpT = 0;
      $('ready-hold-fill').style.width = `${clamp(pose.handsUpT / HANDS_UP_START, 0, 1) * 100}%`;
      $('ready-cam-msg').textContent = !pose.seen ? 'カメラの前に立ってください'
        : !pose.body ? 'もう少し下がって、腰まで写してください'
        : '両手を上げるとスタート！🙌';
      if (pose.handsUpT >= HANDS_UP_START) { sfx.go(); beginTurn(); }
      break;
    }
    case 'countdown': {
      calibrate(dt);
      const n = 3 - Math.floor(t / 0.8);
      if (n <= 0) {
        state = 'run'; t = 0; jump.runTime = 0;
        setPrompt(camOn() ? 'その場で全力ダッシュ！！' : '連打！！', true);
        setGuide(camOn() ? 'ももを高く上げて足ぶみ！' : '');
        sfx.go();
      } else if ($('prompt').textContent !== String(n)) { setPrompt(String(n)); sfx.count(); }
      break;
    }
    case 'run': {
      jump.runTime += dt;
      // 足ぶみでスピードアップ
      if (pose.motion > 0) {
        jump.speed = Math.min(100, jump.speed + pose.motion * STEP_GAIN);
        jump.stepAcc += pose.motion;
        if (jump.stepAcc > 0.4) { jump.stepAcc = 0; sfx.tap(jump.speed); spawnDust(); }
      }
      // 速いほど落ちやすい減速
      jump.speed = Math.max(0, jump.speed - (45 * jump.speed / 100 + 10) * dt);
      jump.runPhase += dt * (4 + jump.speed / 8);
      updateSpeedUI();
      if (jump.runTime >= RUN_TIME) {
        state = 'takeoff'; t = 0;
        if (pose.body) pose.baseHip = recentHipMedian(800);
        show('speed-wrap', false);
        show('timing-wrap', true);
        setPrompt(camOn() ? 'まん中でジャンプ！' : '踏み切り！', true);
        setGuide(camOn() ? 'マーカーがまん中に来たら思いっきり跳べ！' : '');
        jump.marker = 0; jump.markerDir = 1;
        // 速いほどマーカーも速い（カメラのときは体が反応できるよう遅め）
        jump.sweep = camOn() ? 0.45 + jump.speed / 100 * 0.35 : 0.8 + jump.speed / 100;
      }
      break;
    }
    case 'takeoff': {
      jump.marker += jump.markerDir * jump.sweep * dt;
      if (jump.marker >= 1) { jump.marker = 1; jump.markerDir = -1; }
      if (jump.marker <= 0) { jump.marker = 0; jump.markerDir = 1; }
      $('timing-marker').style.left = `${jump.marker * 100}%`;
      // 腰が上がったらジャンプした
      if (pose.body && pose.baseHip - pose.hipY > LIFT_THRESHOLD * pose.baseTorso) {
        judgeTakeoff(false);
        state = 'liftoff'; t = 0; jump.liftMin = pose.hipY;
        setGuide('');
      } else if (t >= TAKEOFF_LIMIT) {
        judgeTakeoff(true);
        launch(0);
      }
      break;
    }
    case 'liftoff': {
      // 本当のジャンプの最高点を測る（腰が下がり始めたら終わり）
      if (pose.body) jump.liftMin = Math.min(jump.liftMin, pose.hipY);
      if (t > 0.8 || (pose.body && pose.hipY > jump.liftMin + 0.08 * pose.baseTorso)) {
        launch(Math.max(0, (pose.baseHip - jump.liftMin) / pose.baseTorso));
        popup(`リアル ${Math.round(jump.realJ * TORSO_CM)}cm！`);
      }
      break;
    }
    case 'flight': {
      let scale = TIME_SCALE;
      if (jump.ringActive) {
        scale = 0.3; // スローモーション
        jump.ringT += dt;
        if (!pose.handsUp) jump.armsDownSeen = true;
        if (camOn() && jump.armsDownSeen && pose.handsUp) doRing(false);
        else if (jump.ringT >= RING_TIME) doRing(true);
        else if (camOn()) setGuide(jump.armsDownSeen ? '輪がキャラに重なったら両手を上げろ！' : 'いったん手を下ろして…');
      }
      const sdt = dt * scale;
      jump.vy -= GRAVITY * sdt;
      jump.py += jump.vy * sdt;
      if (jump.py > jump.maxH) jump.maxH = jump.py;

      if (!jump.ringUsed && jump.v0 > 15 && jump.vy <= jump.v0 * 0.35) startRing();

      for (const l of LANDMARKS) {
        if (!jump.passed.has(l.h) && jump.py >= l.h) {
          jump.passed.add(l.h);
          popup(`${l.name}（${l.h}m）越え！`);
          sfx.pass();
        }
      }
      if (jump.vy <= 0 && !jump.ringActive) {
        state = 'apex'; t = 0;
        setPrompt(`${fmt(jump.maxH)}m！`);
      }
      $('hud-height').innerHTML = `${fmt(Math.max(0, jump.py))}<span>m</span>`;
      break;
    }
    case 'apex': {
      if (t > 1.8) { setPrompt(''); finishJump(); }
      break;
    }
    default: break;
  }

  // カメラ：キャラが画面の 45% の高さに来るよう追従
  const target = cameraFor(jump.py);
  jump.camH = state === 'flight' || state === 'apex' ? target : lerp(jump.camH, target, 1 - Math.pow(0.001, dt));

  // パーティクル
  for (const p of jump.particles) {
    p.life -= dt;
    p.x += p.vx * dt; p.y += p.vy * dt; p.vy += 300 * dt;
  }
  jump.particles = jump.particles.filter((p) => p.life > 0);
}

// ============================================================
//  描画
// ============================================================
const groundLine = () => H * 0.82;
function ppmAt(camH) { return Math.max(1.2, 30 / (1 + camH / 30)); }
function cameraFor(py) {
  let cam = 0;
  for (let i = 0; i < 4; i++) cam = Math.max(0, py - (groundLine() - H * 0.45) / ppmAt(cam));
  return cam;
}
function worldY(h) { return groundLine() - (h - jump.camH) * ppmAt(jump.camH); }

function skyColors(h) {
  // 高度で空の色を変える
  const stops = [
    [0, [126, 200, 255], [210, 238, 255]],
    [500, [70, 140, 230], [150, 205, 255]],
    [1200, [30, 60, 150], [80, 140, 230]],
    [2200, [10, 16, 60], [40, 70, 160]],
    [3500, [2, 2, 12], [15, 20, 60]],
  ];
  let a = stops[0], b = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (h >= stops[i][0] && h <= stops[i + 1][0]) { a = stops[i]; b = stops[i + 1]; break; }
  }
  const k = h >= b[0] ? 1 : clamp((h - a[0]) / (b[0] - a[0] || 1), 0, 1);
  const mix = (c1, c2) => `rgb(${c1.map((v, i) => Math.round(lerp(v, c2[i], k))).join(',')})`;
  return [mix(a[1], b[1]), mix(a[2], b[2])];
}

function draw() {
  const cam = jump.camH;
  const ppm = ppmAt(cam);
  const [top, bottom] = skyColors(cam);
  const g = ctx.createLinearGradient(0, 0, 0, H);
  g.addColorStop(0, top); g.addColorStop(1, bottom);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);

  // 星
  const starA = clamp((cam - 1200) / 800, 0, 1);
  if (starA > 0) {
    ctx.fillStyle = `rgba(255,255,255,${starA})`;
    for (const s of stars) { ctx.beginPath(); ctx.arc(s.x * W, s.y * H, s.r, 0, Math.PI * 2); ctx.fill(); }
  }

  // 富士山（はるか遠くの山として、高度に合わせて見える）
  drawFuji(ppm);

  // 雲
  for (const c of clouds) {
    const y = worldY(c.h);
    if (y < -80 || y > H + 80) continue;
    drawCloud(c.x * W, y, 40 * c.s);
  }

  // ランドマーク
  for (const l of LANDMARKS) drawLandmark(l, ppm);

  // 地面
  const gy = worldY(0);
  if (gy < H + 10) {
    ctx.fillStyle = '#6ab04c';
    ctx.fillRect(0, gy, W, H - gy + 10);
    ctx.fillStyle = '#c8a27a';
    ctx.fillRect(0, gy, W, 14);
    // 助走中はトラックの線を流す
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    const off = (jump.runPhase * 40) % 80;
    for (let x = -off; x < W; x += 80) ctx.fillRect(x, gy + 5, 40, 3);
  }

  // 目盛り
  drawRuler(ppm);
  // 他プレイヤーの記録ライン
  drawRecordLines();

  // キャラクター
  const p = players[turn.idx];
  if (p && state !== 'title' && state !== 'final') {
    const y = worldY(jump.py);
    drawRunner(W / 2, y, p.color);
  }

  // パーティクル
  for (const pt of jump.particles) {
    const y = pt.world ? worldY(0) + pt.y : pt.y;
    ctx.globalAlpha = clamp(pt.life / pt.max, 0, 1);
    ctx.fillStyle = pt.color;
    ctx.beginPath(); ctx.arc(pt.x, y, pt.r, 0, Math.PI * 2); ctx.fill();
  }
  ctx.globalAlpha = 1;

  // 空中リング
  if (state === 'flight' && jump.ringActive) {
    const cy = worldY(jump.py) - 40;
    ctx.lineWidth = 4;
    ctx.strokeStyle = 'rgba(255,255,255,0.8)';
    ctx.setLineDash([6, 6]);
    ctx.beginPath(); ctx.arc(W / 2, cy, RING_TARGET, 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);
    ctx.lineWidth = 8;
    ctx.strokeStyle = '#ffe14d';
    ctx.beginPath(); ctx.arc(W / 2, cy, ringRadius(), 0, Math.PI * 2); ctx.stroke();
  }

  // スピード線
  if (state === 'flight' && jump.vy > 20) {
    ctx.strokeStyle = `rgba(255,255,255,${clamp(jump.vy / 150, 0.1, 0.6)})`;
    ctx.lineWidth = 2;
    for (let i = 0; i < 14; i++) {
      const x = (Math.sin(i * 91.7 + performance.now() / 70) * 0.5 + 0.5) * W;
      const y = ((i * 137 + performance.now() * Math.min(jump.vy, 200) / 60) % (H + 200)) - 100;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y + 60); ctx.stroke();
    }
  }

  drawCamera();
}

// カメラ映像（鏡映し）＋骨格
function camRect() {
  const vw = video.videoWidth || 16, vh = video.videoHeight || 9;
  let w;
  if (state === 'ready') w = W > H ? W * 0.42 : W - 32;
  else if (state === 'flight' || state === 'apex') w = Math.min(W * 0.2, 320);
  else w = Math.min(W * 0.3, 520);
  const h = w * vh / vw;
  if (state === 'ready') {
    return W > H ? { x: W - w - 24, y: (H - h) / 2, w, h } : { x: 16, y: H - h - 16, w, h };
  }
  return { x: W - w - 16, y: H - h - 16, w, h };
}
function drawCamera() {
  if (!camOn() || state === 'title' || state === 'final' || state === 'jresult') return;
  const r = camRect();
  ctx.save();
  ctx.beginPath();
  ctx.roundRect ? ctx.roundRect(r.x, r.y, r.w, r.h, 14) : ctx.rect(r.x, r.y, r.w, r.h);
  ctx.clip();
  ctx.translate(r.x + r.w, r.y);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, r.w, r.h);
  ctx.restore();

  const lm = pose.lm;
  if (lm) {
    const P = (i) => [r.x + (1 - lm[i].x) * r.w, r.y + lm[i].y * r.h];
    const bones = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28]];
    ctx.strokeStyle = pose.handsUp ? '#ffe14d' : '#4cd964';
    ctx.lineWidth = 4; ctx.lineCap = 'round';
    for (const [a, b] of bones) {
      if ((lm[a].visibility ?? 1) < 0.4 || (lm[b].visibility ?? 1) < 0.4) continue;
      const [x1, y1] = P(a), [x2, y2] = P(b);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    }
    // 基準の腰の高さとジャンプ判定ライン
    if (pose.baseHip && ['countdown', 'run', 'takeoff', 'liftoff'].includes(state)) {
      const by = r.y + pose.baseHip * r.h;
      const jy = r.y + (pose.baseHip - LIFT_THRESHOLD * pose.baseTorso) * r.h;
      ctx.setLineDash([8, 6]);
      ctx.strokeStyle = 'rgba(255,255,255,0.7)'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(r.x, by); ctx.lineTo(r.x + r.w, by); ctx.stroke();
      if (state === 'takeoff' || state === 'liftoff') {
        ctx.strokeStyle = '#ff5a36';
        ctx.beginPath(); ctx.moveTo(r.x, jy); ctx.lineTo(r.x + r.w, jy); ctx.stroke();
      }
      ctx.setLineDash([]);
    }
  }
  ctx.lineWidth = 4;
  ctx.strokeStyle = pose.body ? '#fff' : '#ff3b30';
  ctx.beginPath();
  ctx.roundRect ? ctx.roundRect(r.x, r.y, r.w, r.h, 14) : ctx.rect(r.x, r.y, r.w, r.h);
  ctx.stroke();
  if (!pose.body) {
    ctx.font = `bold ${Math.max(14, r.w / 18)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillStyle = '#ff3b30';
    ctx.fillText('体が写っていません', r.x + r.w / 2, r.y + r.h - 12);
  }
}

function drawFuji(ppm) {
  // 3776m の山頂が高度に合わせて近づいてくる背景
  const top = worldY(3776);
  const base = Math.max(worldY(0), H + 20);
  if (top > H + 20) return;
  const half = Math.max(W * 0.6, (base - top) * 1.4);
  ctx.fillStyle = 'rgba(60,80,140,0.55)';
  ctx.beginPath();
  ctx.moveTo(W / 2 - half, base); ctx.lineTo(W / 2 - 40, top); ctx.lineTo(W / 2 + 40, top); ctx.lineTo(W / 2 + half, base);
  ctx.fill();
  ctx.fillStyle = 'rgba(255,255,255,0.85)';
  const snow = Math.min(90, 400 * ppm);
  ctx.beginPath();
  ctx.moveTo(W / 2 - 40 - snow * 1.4, top + snow); ctx.lineTo(W / 2 - 40, top); ctx.lineTo(W / 2 + 40, top); ctx.lineTo(W / 2 + 40 + snow * 1.4, top + snow);
  ctx.fill();
  if (top > -20 && top < H) {
    ctx.font = 'bold 14px sans-serif'; ctx.textAlign = 'center';
    ctx.fillStyle = '#fff';
    ctx.fillText('富士山 3776m', W / 2, top - 10);
  }
}

function drawCloud(x, y, r) {
  ctx.fillStyle = 'rgba(255,255,255,0.9)';
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.arc(x + r * 0.9, y + r * 0.2, r * 0.75, 0, Math.PI * 2);
  ctx.arc(x - r * 0.9, y + r * 0.25, r * 0.7, 0, Math.PI * 2);
  ctx.fill();
}

function drawLandmark(l, ppm) {
  if (l.kind === 'none') return;
  const baseY = worldY(0);
  const topY = worldY(l.h);
  const floating = l.kind === 'balloon' || l.kind === 'plane';
  if (!floating && topY > H + 50) return;
  const x = l.x * W;
  const w = l.w * Math.min(1, 0.3 + ppm / 30) * Math.min(1, W / 800);
  ctx.fillStyle = l.color;
  switch (l.kind) {
    case 'house': {
      ctx.fillRect(x - w / 2, lerp(topY, baseY, 0.45), w, baseY - lerp(topY, baseY, 0.45));
      ctx.fillStyle = '#a93226';
      ctx.beginPath(); ctx.moveTo(x - w / 2 - 8, lerp(topY, baseY, 0.45)); ctx.lineTo(x, topY); ctx.lineTo(x + w / 2 + 8, lerp(topY, baseY, 0.45)); ctx.fill();
      break;
    }
    case 'tree': {
      ctx.fillStyle = '#8e5b3a';
      ctx.fillRect(x - w * 0.1, lerp(topY, baseY, 0.5), w * 0.2, baseY - lerp(topY, baseY, 0.5));
      ctx.fillStyle = l.color;
      ctx.beginPath(); ctx.ellipse(x, lerp(topY, baseY, 0.3), w * 0.6, (baseY - topY) * 0.32, 0, 0, Math.PI * 2); ctx.fill();
      break;
    }
    case 'wheel': {
      const r = (baseY - topY) / 2;
      const cy = topY + r;
      ctx.strokeStyle = l.color; ctx.lineWidth = 4;
      ctx.beginPath(); ctx.arc(x, cy, r * 0.95, 0, Math.PI * 2); ctx.stroke();
      for (let i = 0; i < 8; i++) {
        const a = i * Math.PI / 4 + performance.now() / 4000;
        ctx.beginPath(); ctx.moveTo(x, cy); ctx.lineTo(x + Math.cos(a) * r * 0.95, cy + Math.sin(a) * r * 0.95); ctx.stroke();
        ctx.fillStyle = ['#e74c3c', '#f1c40f', '#3498db', '#2ecc71'][i % 4];
        ctx.fillRect(x + Math.cos(a) * r * 0.95 - 6, cy + Math.sin(a) * r * 0.95, 12, 10);
      }
      break;
    }
    case 'building': {
      ctx.fillRect(x - w / 2, topY, w, baseY - topY);
      ctx.fillStyle = 'rgba(255,255,200,0.6)';
      const step = Math.max(6, 4 * ppm);
      for (let yy = Math.max(topY + 6, -10); yy < Math.min(baseY, H) - 4; yy += step) {
        for (let xx = x - w / 2 + 8; xx < x + w / 2 - 8; xx += 18) ctx.fillRect(xx, yy, 9, Math.min(step * 0.5, 8));
      }
      break;
    }
    case 'tower': {
      ctx.beginPath();
      ctx.moveTo(x - w / 2, baseY); ctx.lineTo(x - 3, topY); ctx.lineTo(x + 3, topY); ctx.lineTo(x + w / 2, baseY);
      ctx.closePath(); ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.fillRect(x - w * 0.18, lerp(topY, baseY, 0.35), w * 0.36, 6);
      break;
    }
    case 'balloon': {
      const by = topY + Math.sin(performance.now() / 700) * 6;
      if (by < -100 || by > H + 100) return;
      ctx.beginPath(); ctx.arc(x, by - 30, 30, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#e67e22'; ctx.fillRect(x - 3, by - 60, 6, 60);
      ctx.fillStyle = '#8e5b3a'; ctx.fillRect(x - 10, by + 8, 20, 14);
      break;
    }
    case 'plane': {
      const px = x + Math.sin(performance.now() / 3000) * 40;
      if (topY < -100 || topY > H + 100) return;
      ctx.beginPath(); ctx.ellipse(px, topY - 20, 55, 10, 0, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.moveTo(px - 10, topY - 20); ctx.lineTo(px + 15, topY + 10); ctx.lineTo(px + 25, topY - 20); ctx.fill();
      ctx.beginPath(); ctx.moveTo(px + 40, topY - 22); ctx.lineTo(px + 55, topY - 45); ctx.lineTo(px + 55, topY - 20); ctx.fill();
      break;
    }
    default: break;
  }
  // ラベル
  const ly = floating ? topY - 70 : topY - 10;
  if (ly > -20 && ly < H) {
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(0,0,0,0.5)';
    ctx.strokeText(`${l.name} ${l.h}m`, x, ly);
    ctx.fillStyle = '#fff';
    ctx.fillText(`${l.name} ${l.h}m`, x, ly);
  }
}

function drawRuler(ppm) {
  const steps = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];
  const step = steps.find((s) => s * ppm >= 45) || 2000;
  const topH = jump.camH + (groundLine()) / ppm;
  const start = Math.max(0, Math.floor(jump.camH / step) * step - step);
  ctx.font = 'bold 13px sans-serif';
  ctx.textAlign = 'right';
  for (let h = start; h <= topH + step; h += step) {
    const y = worldY(h);
    if (y > groundLine() + 2 || y < -10) continue;
    ctx.fillStyle = 'rgba(255,255,255,0.8)';
    ctx.fillRect(W - 26, y, 22, 2);
    ctx.fillText(`${h}m`, W - 30, y + 4);
  }
}

function drawRecordLines() {
  if (state === 'title' || state === 'final') return;
  const lines = [];
  players.forEach((p, i) => { if (i !== turn.idx && p.best > 0) lines.push({ h: p.best, label: p.name, color: p.color }); });
  const me = players[turn.idx];
  if (me && me.best > 0) lines.push({ h: me.best, label: '自己ベスト', color: '#fff' });
  const at = allTimeBest();
  if (at) lines.push({ h: at.h, label: `歴代1位 ${at.name}`, color: '#ffd700' });
  ctx.font = 'bold 15px sans-serif';
  ctx.textAlign = 'left';
  for (const l of lines) {
    const y = worldY(l.h);
    if (y < -10 || y > H) continue;
    ctx.strokeStyle = l.color; ctx.lineWidth = 2;
    ctx.setLineDash([10, 6]);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W - 70, y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = l.color;
    ctx.fillText(`${l.label} ${fmt(l.h)}m`, 8, y - 5);
  }
}

// キャラクター（足元が (x, y)）
function drawRunner(x, y, color) {
  const now = performance.now() / 1000;
  let pose2 = 'stand';
  if (state === 'run') pose2 = 'run';
  else if (state === 'takeoff' || state === 'liftoff') pose2 = 'crouch';
  else if (state === 'flight' || state === 'apex') pose2 = jump.vy > 0 ? 'jump' : 'apex';

  ctx.save();
  ctx.translate(x, y);
  ctx.scale(1.3, 1.3);
  // 影
  if (jump.py < 30) {
    ctx.fillStyle = 'rgba(0,0,0,0.2)';
    const gy = (worldY(0) - y) / 1.3;
    ctx.beginPath(); ctx.ellipse(0, gy, 22, 6, 0, 0, Math.PI * 2); ctx.fill();
  }
  let bodyTop = -58, hip = -26;
  let legA = 0, legB = 0, armA = 0, armB = 0;
  if (pose2 === 'run') {
    const ph = jump.runPhase * 2.5;
    legA = Math.sin(ph) * 0.9; legB = -legA;
    armA = -legA; armB = legA;
  } else if (pose2 === 'crouch') {
    const k = 0.5 + 0.5 * Math.sin(now * 12);
    bodyTop = -44 - k * 2; hip = -16;
    legA = 0.9; legB = -0.9; armA = 2.2; armB = 2.4;
  } else if (pose2 === 'jump') {
    legA = 0.2; legB = -0.2; armA = Math.PI - 0.3; armB = Math.PI + 0.3;
  } else if (pose2 === 'apex') {
    legA = 0.6; legB = -0.6; armA = Math.PI / 2 + 0.8; armB = -Math.PI / 2 - 0.8;
  } else {
    legA = 0.15; legB = -0.15; armA = 0.3; armB = -0.3;
  }
  ctx.lineCap = 'round';
  ctx.strokeStyle = '#1c2340';
  ctx.lineWidth = 7;
  const limb = (ox, oy, ang, len) => {
    ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox + Math.sin(ang) * len, oy + Math.cos(ang) * len); ctx.stroke();
  };
  limb(0, hip, legA, -hip);
  limb(0, hip, legB, -hip);
  // 胴体
  ctx.strokeStyle = color;
  ctx.lineWidth = 16;
  ctx.beginPath(); ctx.moveTo(0, bodyTop + 10); ctx.lineTo(0, hip); ctx.stroke();
  // 腕
  ctx.strokeStyle = '#1c2340'; ctx.lineWidth = 6;
  limb(0, bodyTop + 14, armA, 22);
  limb(0, bodyTop + 14, armB, 22);
  // 頭
  ctx.fillStyle = '#ffd9b3';
  ctx.beginPath(); ctx.arc(0, bodyTop - 4, 13, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(0, bodyTop - 8, 13, Math.PI, Math.PI * 2); ctx.fill(); // 帽子
  ctx.fillStyle = '#1c2340';
  ctx.beginPath(); ctx.arc(4, bodyTop - 3, 2, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
}

// ============================================================
//  メインループ
// ============================================================
function loop(now) {
  const dt = Math.min(0.05, (now - lastTime) / 1000 || 0);
  lastTime = now;
  detectPose(now);
  update(dt);
  draw();
  requestAnimationFrame(loop);
}

resize();
makeWorld();
renderNameInputs();
updateCamStatus();
initCamera();
requestAnimationFrame((n) => { lastTime = n; loop(n); });
