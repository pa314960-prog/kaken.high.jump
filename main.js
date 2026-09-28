'use strict';

// ============================================================
//  最高到達点チャレンジ
//  カメラの前で本当にジャンプして、指先が届いた高さ（cm）を競う。
//  壁に映した目盛りに、跳んでいる人の影が重なるように表示する。
// ============================================================

const $ = (id) => document.getElementById(id);
const canvas = $('game');
const ctx = canvas.getContext('2d');
const video = $('cam');

// ---------- 定数 ----------
const HEAD_FACTOR = 1.13;     // 鼻〜足首の長さ × これ ≒ 身長
const ANKLE_FACTOR = 0.045;   // 床〜足首の高さ（身長比）
const CALIB_HOLD = 1.5;       // 両手を上げて止まる秒数
const AIR_CM = 8;             // 腰がこれだけ上がったら「跳んでいる」
const LAND_CM = 3;            // ここまで戻ったら着地
const PLAYER_COLORS = ['#e8452c', '#2f80ed', '#27ae60', '#9b51e0'];
const RANK_KEY = 'reachjump.ranking.v1';
const NAME_KEY = 'reachjump.players.v1';
const MODES = [
  { key: 'reach', label: '到達点', unit: 'cm' },
  { key: 'jump', label: 'ジャンプ力', unit: 'cm' },
];
const REFERENCES = [
  { cm: 200, label: 'ドアの高さ' },
  { cm: 224, label: 'バレーネット（女子）' },
  { cm: 243, label: 'バレーネット（男子）' },
  { cm: 305, label: 'バスケットゴール' },
];

// ---------- 状態 ----------
let W = 0, H = 0, DPR = 1;
let state = 'title';
let players = [];
let turnIdx = 0;
let timeLimit = 10;
let mode = 0;
let t = 0;
let lastTime = 0;

const run = {
  calibT: 0,
  cal: null,          // { cmPerUnit, floorY, standReach, baseHip, heightCm }
  airborne: false, maxRise: 0,
  jumps: [],          // 1回ごとのジャンプ力(cm)
  bestJump: 0,
  timeLeft: 0,
  flash: [],          // 記録ラインの光るエフェクト
};

// ============================================================
//  カメラ・姿勢検出
// ============================================================
const pose = {
  status: 'loading', message: 'カメラを準備中…',
  landmarker: null, lastVideoTime: -1,
  lm: null, seen: false, full: false, hipsOk: false, handsUp: false,
  noseY: 0, ankleY: 0, hipY: 0, hipX: 0.5, handTopY: 0, handsInFrame: false,
  hist: [],
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
      ? 'カメラが許可されませんでした。ブラウザの設定でカメラを許可して再読み込みしてください'
      : 'カメラが使えません（https で開いているか、カメラがつながっているか確認してください）';
  }
  updateCamStatus();
}
function updateCamStatus() {
  const el = $('cam-status');
  el.textContent = pose.message;
  el.className = `cam-status ${pose.status}`;
}

function detectPose(now) {
  if (pose.status !== 'ok' || video.readyState < 2 || video.currentTime === pose.lastVideoTime) return;
  pose.lastVideoTime = video.currentTime;
  let res;
  try { res = pose.landmarker.detectForVideo(video, now); } catch (e) { return; }
  processLandmarks((res && res.landmarks && res.landmarks[0]) || null, now);
}

function processLandmarks(lm, now) {
  pose.lm = lm;
  pose.seen = !!lm;
  if (!lm) { pose.full = pose.hipsOk = pose.handsUp = false; return; }
  const vis = (i, th = 0.5) => (lm[i].visibility ?? 1) > th;
  pose.hipsOk = vis(23) && vis(24);
  pose.full = pose.hipsOk && vis(0) && vis(27) && vis(28);
  pose.noseY = lm[0].y;
  pose.ankleY = (lm[27].y + lm[28].y) / 2;
  pose.hipY = (lm[23].y + lm[24].y) / 2;
  pose.hipX = (lm[23].x + lm[24].x) / 2;
  pose.handsUp = vis(15) && vis(16) && lm[15].y < lm[0].y && lm[16].y < lm[0].y;
  // 指先（人さし指の付け根）と手首のうち一番上
  let top = Infinity;
  for (const i of [15, 16, 19, 20]) if (vis(i, 0.3)) top = Math.min(top, lm[i].y);
  pose.handTopY = top;
  pose.handsInFrame = top > 0.01 && top < 1;
  pose.hist.push({ t: now, hip: pose.hipY });
  while (pose.hist.length && now - pose.hist[0].t > 800) pose.hist.shift();
}

// 腰がほぼ止まっているか
function isStill() {
  if (pose.hist.length < 5) return false;
  const v = pose.hist.map((h) => h.hip);
  return Math.max(...v) - Math.min(...v) < 0.015;
}

// いまのフレームから、立っている前提でスケールを出す
function liveScale(heightCm) {
  const span = (pose.ankleY - pose.noseY) * HEAD_FACTOR;
  if (!pose.full || span < 0.1) return null;
  const cmPerUnit = heightCm / span;
  return { cmPerUnit, floorY: pose.ankleY + ANKLE_FACTOR * heightCm / cmPerUnit };
}

function calibrate(p) {
  const s = liveScale(p.heightCm);
  if (!s) return null;
  let standReach = pose.handsInFrame
    ? (s.floorY - pose.handTopY) * s.cmPerUnit + 2
    : p.heightCm * 1.33;
  standReach = clamp(standReach, p.heightCm * 1.18, p.heightCm * 1.45);
  return { ...s, standReach, baseHip: pose.hipY, heightCm: p.heightCm };
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
  count: () => tone(660, 0.15, 'sine', 0.12),
  go: () => tone(990, 0.4, 'sine', 0.14),
  ok: () => { tone(880, 0.1, 'triangle', 0.1); setTimeout(() => tone(1320, 0.15, 'triangle', 0.1), 80); },
  jump: () => tone(400, 0.25, 'triangle', 0.1, 900),
  best: () => [784, 988, 1175, 1568].forEach((f, i) => setTimeout(() => tone(f, 0.18, 'triangle', 0.1), i * 70)),
  tick: () => tone(1200, 0.05, 'square', 0.05),
  end: () => tone(1500, 0.6, 'sawtooth', 0.08, 1400),
  fanfare: () => [523, 659, 784, 1047].forEach((f, i) => setTimeout(() => tone(f, 0.25, 'triangle', 0.1), i * 110)),
};

// ============================================================
//  ユーティリティ
// ============================================================
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, k) => a + (b - a) * k;
const cm = (v) => Math.round(v);

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
function setPrompt(text) { $('prompt').textContent = text; show('prompt', !!text); }
function popup(text) {
  const el = $('popup');
  el.classList.add('hidden');
  void el.offsetWidth;
  el.textContent = text;
  el.classList.remove('hidden');
}
function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen?.().catch(() => {});
}
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
function medal(n) { return ['🥇', '🥈', '🥉'][n - 1] || n; }
const score = (p) => (mode === 0 ? p.reach : p.jump);

// ============================================================
//  ランキング
// ============================================================
function loadRanking() {
  try { return JSON.parse(localStorage.getItem(RANK_KEY)) || []; } catch (e) { return []; }
}
function saveRanking(list) {
  try { localStorage.setItem(RANK_KEY, JSON.stringify(list)); } catch (e) { /* 保存できなくても続行 */ }
}
function sortedRanking(m) {
  const k = MODES[m].key;
  return loadRanking().sort((a, b) => b[k] - a[k]);
}
function addToRanking(p) {
  const list = loadRanking();
  const entry = { name: p.name, reach: p.reach, jump: p.jump, d: new Date().toLocaleDateString('ja-JP'), id: Math.random() };
  list.push(entry);
  // 到達点・ジャンプ力それぞれの上位20件だけ残す
  const keep = new Set();
  for (const k of ['reach', 'jump']) [...list].sort((a, b) => b[k] - a[k]).slice(0, 20).forEach((e) => keep.add(e));
  saveRanking(list.filter((e) => keep.has(e)));
  const pos = sortedRanking(mode).findIndex((e) => e.id === entry.id);
  return pos >= 0 && pos < 10 ? pos + 1 : 0;
}
function renderRanking(m) {
  const list = sortedRanking(m).slice(0, 10);
  const k = MODES[m].key;
  const ol = $('rank-list');
  ol.innerHTML = '';
  if (!list.length) { ol.innerHTML = '<li class="empty">まだ記録がありません</li>'; return; }
  list.forEach((r, i) => {
    const li = document.createElement('li');
    const other = k === 'reach' ? `ジャンプ力 ${cm(r.jump)}cm` : `到達点 ${cm(r.reach)}cm`;
    li.innerHTML = `<span class="pos">${medal(i + 1)}</span><span class="nm">${esc(r.name)}</span>` +
      `<span class="sub2">${other}<br>${esc(r.d)}</span><span class="h">${cm(r[k])}cm</span>`;
    ol.appendChild(li);
  });
}

// ============================================================
//  タイトル画面
// ============================================================
let playerCount = 1;
function loadSaved() {
  try { return JSON.parse(localStorage.getItem(NAME_KEY)) || []; } catch (e) { return []; }
}
function renderNameInputs() {
  const saved = loadSaved();
  const box = $('names');
  const cur = [...box.querySelectorAll('.pl')].map((row) => ({ name: row.children[0].value, h: row.children[1].value }));
  box.innerHTML = '';
  for (let i = 0; i < playerCount; i++) {
    const row = document.createElement('div');
    row.className = 'pl';
    row.style.display = 'contents';
    const n = document.createElement('input');
    n.maxLength = 8; n.placeholder = `プレイヤー${i + 1}`;
    n.value = cur[i]?.name ?? saved[i]?.name ?? '';
    n.style.borderColor = PLAYER_COLORS[i];
    const h = document.createElement('input');
    h.type = 'number'; h.min = 90; h.max = 220; h.placeholder = '160';
    h.value = cur[i]?.h ?? saved[i]?.h ?? '';
    row.append(n, h);
    box.appendChild(row);
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
segSelect('time-btns', (n) => { timeLimit = n; });
segSelect('mode-btns', (n) => { mode = n; });
segSelect('rank-tabs', (n) => renderRanking(n));

$('start-btn').addEventListener('click', () => {
  audio();
  const rows = [...$('names').querySelectorAll('.pl')];
  const saved = rows.map((row) => ({ name: row.children[0].value.trim(), h: row.children[1].value }));
  try { localStorage.setItem(NAME_KEY, JSON.stringify(saved)); } catch (e) { /* 無視 */ }
  players = saved.map((s, i) => ({
    name: s.name || `プレイヤー${i + 1}`,
    heightCm: clamp(Number(s.h) || 160, 90, 220),
    color: PLAYER_COLORS[i],
    reach: 0, jump: 0, standReach: 0, jumps: [], done: false,
  }));
  turnIdx = 0;
  goReady();
});
$('rank-btn').addEventListener('click', () => {
  $('rank-tabs').querySelectorAll('button').forEach((x) => x.classList.toggle('on', Number(x.dataset.n) === mode));
  renderRanking(mode);
  showScreen('ranking');
});
$('rank-back').addEventListener('click', () => showScreen('title'));
$('rank-clear').addEventListener('click', () => {
  if (confirm('ランキングを全部消しますか？')) { saveRanking([]); renderRanking(mode); }
});
$('again-btn').addEventListener('click', () => {
  players.forEach((p) => Object.assign(p, { reach: 0, jump: 0, standReach: 0, jumps: [], done: false }));
  turnIdx = 0;
  goReady();
});
$('title-btn').addEventListener('click', () => { state = 'title'; showScreen('title'); });
$('fs-btn').addEventListener('click', toggleFullscreen);
window.addEventListener('keydown', (e) => {
  if (e.code === 'KeyF' && !(e.target instanceof HTMLInputElement)) toggleFullscreen();
});

// ============================================================
//  ターン進行
// ============================================================
const cur = () => players[turnIdx];

function goReady() {
  state = 'ready'; t = 0;
  Object.assign(run, { calibT: 0, cal: null, airborne: false, maxRise: 0, jumps: [], bestJump: 0, flash: [] });
  show('hud', false); setPrompt('');
  const p = cur();
  $('ready-name').textContent = p.name;
  $('ready-name').style.color = p.color;
  $('ready-sub').textContent = `身長 ${p.heightCm}cm で計算します`;
  showScreen('ready');
}
$('skip-btn').addEventListener('click', () => { cur().done = true; nextPlayer(); });

function startCountdown() {
  state = 'countdown'; t = 0;
  const p = cur();
  p.standReach = run.cal.standReach;
  showScreen(null);
  $('hud-name').textContent = p.name;
  $('hud-name').style.color = p.color;
  $('hud-sec').textContent = timeLimit;
  $('hud-timer').classList.remove('hurry');
  $('hud-reach-val').textContent = '---';
  show('hud', true);
  setPrompt('3');
  sfx.count();
}

function recordJump(rise) {
  const p = cur();
  if (rise < AIR_CM) return;
  run.jumps.push(rise);
  run.flash.push({ cm: p.standReach + rise, life: 1.2 });
  if (rise > run.bestJump) {
    const first = run.bestJump === 0;
    run.bestJump = rise;
    $('hud-reach-val').textContent = cm(p.standReach + rise);
    popup(first ? `${cm(p.standReach + rise)}cm！` : `記録更新！ ${cm(p.standReach + rise)}cm`);
    first ? sfx.jump() : sfx.best();
  } else {
    popup(`${cm(p.standReach + rise)}cm`);
    sfx.jump();
  }
}

function finishTurn() {
  const p = cur();
  p.jump = run.bestJump;
  p.reach = run.bestJump > 0 ? p.standReach + run.bestJump : 0;
  p.jumps = run.jumps.slice();
  p.done = true;
  state = 'jresult';
  show('hud', false); setPrompt('');
  $('jr-name').textContent = p.name;
  $('jr-name').style.color = p.color;
  $('jr-reach').innerHTML = p.reach ? `${cm(p.reach)}<span>cm</span>` : '<span>記録なし</span>';
  $('jr-detail').innerHTML =
    `ジャンプ力（垂直跳び） <b>${cm(p.jump)}cm</b><br>` +
    `立ったときの指先 ${cm(p.standReach)}cm ／ ジャンプ ${p.jumps.length}回`;
  const passed = REFERENCES.filter((r) => p.reach >= r.cm);
  const next = REFERENCES.find((r) => p.reach < r.cm);
  $('jr-pass').textContent =
    (passed.length ? `${passed[passed.length - 1].label}にタッチ！` : '') +
    (next && p.reach ? ` あと${cm(next.cm - p.reach)}cmで${next.label}` : '');
  const last = players.every((x) => x.done);
  $('jr-next').textContent = last ? 'けっか発表へ' : 'つぎの人へ';
  showScreen('jump-result');
  sfx.fanfare();
}
$('jr-retry').addEventListener('click', () => goReady());
$('jr-next').addEventListener('click', nextPlayer);

function nextPlayer() {
  const i = players.findIndex((p) => !p.done);
  if (i < 0) showFinal();
  else { turnIdx = i; goReady(); }
}

function showFinal() {
  state = 'final';
  show('hud', false);
  const sorted = [...players].sort((a, b) => score(b) - score(a));
  const ol = $('final-list');
  ol.innerHTML = '';
  sorted.forEach((p, i) => {
    const li = document.createElement('li');
    if (i === 0) li.classList.add('first');
    const other = mode === 0 ? `ジャンプ力 ${cm(p.jump)}cm` : `到達点 ${cm(p.reach)}cm`;
    li.innerHTML = `<span class="pos">${medal(i + 1)}</span><span class="nm" style="color:${p.color}">${esc(p.name)}</span>` +
      `<span class="sub2">${other}</span><span class="h">${cm(score(p))}cm</span>`;
    ol.appendChild(li);
  });
  const msgs = [];
  players.filter((p) => p.reach > 0).forEach((p) => {
    const pos = addToRanking(p);
    if (pos) msgs.push(`${esc(p.name)} が${MODES[mode].label}の歴代${pos}位にランクイン！`);
  });
  $('final-rank-msg').innerHTML = msgs.join('<br>');
  showScreen('final');
  sfx.fanfare();
}

// ============================================================
//  更新
// ============================================================
function update(dt) {
  t += dt;
  const p = cur();
  switch (state) {
    case 'ready': {
      let msg = '', warn = true;
      if (pose.status !== 'ok') msg = 'カメラの準備を待っています…';
      else if (!pose.seen) msg = 'カメラの前に立ってください';
      else if (!pose.full) msg = '頭から足まで全身が写るように<br>下がってください';
      else if (!pose.handsUp) { msg = '両手をまっすぐ上に<br>伸ばしてください 🙌'; warn = false; }
      else if (!isStill()) { msg = 'そのまま止まって…'; warn = false; }
      else { msg = 'はかっています…'; warn = false; }
      const measuring = pose.full && pose.handsUp && isStill();
      run.calibT = measuring ? run.calibT + dt : Math.max(0, run.calibT - dt * 2);
      $('ready-msg').innerHTML = msg;
      $('ready-msg').classList.toggle('warn', warn);
      $('ready-hold-fill').style.width = `${clamp(run.calibT / CALIB_HOLD, 0, 1) * 100}%`;
      if (run.calibT >= CALIB_HOLD) {
        run.cal = calibrate(p);
        if (run.cal) {
          sfx.ok();
          popup(`指先 ${cm(run.cal.standReach)}cm`);
          startCountdown();
        } else run.calibT = 0;
      }
      break;
    }
    case 'countdown': {
      const n = 3 - Math.floor(t / 0.8);
      if (n <= 0) {
        state = 'jumping'; t = 0; run.timeLeft = timeLimit;
        setPrompt('ジャンプ！');
        setTimeout(() => { if (state === 'jumping') setPrompt(''); }, 900);
        sfx.go();
      } else if ($('prompt').textContent !== String(n)) { setPrompt(String(n)); sfx.count(); }
      break;
    }
    case 'jumping':
    case 'ending': {
      trackJump(dt);
      if (state === 'jumping') {
        const before = Math.ceil(run.timeLeft);
        run.timeLeft = Math.max(0, run.timeLeft - dt);
        const sec = Math.ceil(run.timeLeft);
        $('hud-sec').textContent = sec;
        $('hud-timer').classList.toggle('hurry', run.timeLeft <= 3);
        if (sec !== before && sec <= 3 && sec > 0) sfx.tick();
        if (run.timeLeft <= 0) { state = 'ending'; t = 0; sfx.end(); setPrompt('しゅうりょう！'); }
      } else if (!run.airborne || t > 1.5) {
        if (run.airborne) { run.airborne = false; recordJump(run.maxRise); }
        if (t > 1.2) finishTurn();
      }
      break;
    }
    default: break;
  }
  run.flash.forEach((f) => { f.life -= dt; });
  run.flash = run.flash.filter((f) => f.life > 0);
}

// 腰の上がり方で1回ごとのジャンプを検出する
function trackJump(dt) {
  if (!run.cal || !pose.hipsOk) return;
  const rise = (run.cal.baseHip - pose.hipY) * run.cal.cmPerUnit;
  if (!run.airborne) {
    if (rise > AIR_CM) { run.airborne = true; run.maxRise = rise; }
    else if (Math.abs(rise) < 4) run.cal.baseHip = lerp(run.cal.baseHip, pose.hipY, 1 - Math.pow(0.5, dt)); // ゆっくり基準を追従
  } else {
    run.maxRise = Math.max(run.maxRise, rise);
    if (rise < LAND_CM) { run.airborne = false; recordJump(run.maxRise); }
  }
}
function liveReach() {
  if (!run.cal || !pose.hipsOk) return 0;
  return cur().standReach + Math.max(0, (run.cal.baseHip - pose.hipY) * run.cal.cmPerUnit);
}

// ============================================================
//  描画
// ============================================================
function wallLayout() {
  const p = cur();
  const floor = H * 0.9;
  let topCm = 320;
  if (p) topCm = Math.max(topCm, (p.standReach || p.heightCm * 1.33) + 90);
  players.forEach((q) => { if (q.reach) topCm = Math.max(topCm, q.reach + 30); });
  const pxPerCm = (floor - H * 0.16) / topCm; // 上はタイマーなどの表示ぶん空ける
  const figX = W > H && W >= 900 ? W * 0.6 : W / 2;
  return { floor, topCm, pxPerCm, figX, y: (c) => floor - c * pxPerCm };
}

function draw() {
  // 壁
  ctx.fillStyle = '#f4f5f7';
  ctx.fillRect(0, 0, W, H);
  if (state === 'title' || state === 'ranking' || state === 'final' || !cur()) { drawIdleWall(); drawCamera(); return; }
  const L = wallLayout();
  const p = cur();

  // 方眼
  ctx.strokeStyle = 'rgba(40,60,90,0.07)'; ctx.lineWidth = 1;
  for (let c = 0; c <= L.topCm; c += 10) { const y = L.y(c); ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
  const gx = 10 * L.pxPerCm;
  for (let x = L.figX % gx; x < W; x += gx) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, L.floor); ctx.stroke(); }

  // 目盛り（左右）
  ctx.font = `bold ${Math.max(12, L.pxPerCm * 6)}px sans-serif`;
  for (let c = 0; c <= L.topCm; c += 10) {
    const y = L.y(c), big = c % 50 === 0;
    ctx.fillStyle = big ? '#1c2340' : 'rgba(28,35,64,0.5)';
    ctx.fillRect(0, y - 1, big ? 40 : 20, 2);
    ctx.fillRect(W - (big ? 40 : 20), y - 1, big ? 40 : 20, 2);
    if (big && c > 0) {
      ctx.textAlign = 'left'; ctx.fillText(`${c}`, 46, y + 5);
      ctx.textAlign = 'right'; ctx.fillText(`${c}`, W - 46, y + 5);
    }
  }

  // 床
  ctx.fillStyle = '#c8633a';
  ctx.fillRect(0, L.floor, W, H - L.floor);
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, L.floor, W, 4);
  ctx.fillStyle = '#e8452c';
  ctx.beginPath(); ctx.ellipse(L.figX, L.floor + (H - L.floor) / 2, 60, (H - L.floor) / 2 - 6, 0, 0, Math.PI * 2); ctx.fill();

  const labelX = L.figX - 30 * L.pxPerCm - 120;
  const line = (c, color, width, label, dash = null, align = 'right', lx = labelX) => {
    const y = L.y(c);
    if (y < 0 || y > L.floor) return;
    ctx.strokeStyle = color; ctx.lineWidth = width;
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath(); ctx.moveTo(60, y); ctx.lineTo(W - 60, y); ctx.stroke();
    ctx.setLineDash([]);
    if (label) {
      ctx.font = `bold ${Math.max(14, Math.min(24, L.pxPerCm * 6))}px sans-serif`;
      ctx.textAlign = align;
      ctx.lineWidth = 4; ctx.strokeStyle = 'rgba(255,255,255,0.9)';
      ctx.strokeText(label, lx, y - 6);
      ctx.fillStyle = color; ctx.fillText(label, lx, y - 6);
    }
  };

  // 目安の高さ
  REFERENCES.forEach((r) => line(r.cm, '#7a869a', 2, `${r.label} ${r.cm}`, [4, 6], 'left', 70));
  // 歴代1位・ほかの人の記録
  const top = sortedRanking(mode)[0];
  if (top && mode === 0) line(top.reach, '#d4a300', 3, `歴代1位 ${top.name} ${cm(top.reach)}`, [14, 8], 'right', W - 70);
  players.forEach((q, i) => {
    if (i !== turnIdx && q.reach) line(q.reach, q.color, 3, `${q.name} ${cm(q.reach)}`, [10, 6], 'right', W - 70);
  });

  if (run.cal) {
    // 立ったときの指先
    line(p.standReach, '#2f80ed', 2, `指先 ${cm(p.standReach)}`, [6, 6]);
    // これまでのジャンプ
    for (const j of run.jumps) line(p.standReach + j, 'rgba(232,69,44,0.35)', 2, '');
    // ベスト
    if (run.bestJump > 0) line(p.standReach + run.bestJump, '#e8452c', 5, `${cm(p.standReach + run.bestJump)}cm`);
    // 記録した瞬間に光る
    for (const f of run.flash) {
      const y = L.y(f.cm);
      ctx.fillStyle = `rgba(255,210,63,${clamp(f.life, 0, 1) * 0.6})`;
      ctx.fillRect(0, y - 10, W, 20);
    }
    // いまの高さ
    if (state === 'jumping' || state === 'ending') {
      const r = liveReach();
      if (r) {
        const y = L.y(r);
        ctx.fillStyle = '#1c2340';
        ctx.beginPath(); ctx.moveTo(L.figX + 60, y); ctx.lineTo(L.figX + 80, y - 10); ctx.lineTo(L.figX + 80, y + 10); ctx.fill();
      }
    }
  }

  drawShadow(L, p);
  drawCamera();
}

function drawIdleWall() {
  ctx.strokeStyle = 'rgba(40,60,90,0.07)'; ctx.lineWidth = 1;
  for (let y = 0; y < H; y += 24) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
  for (let x = 0; x < W; x += 24) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
  ctx.fillStyle = '#c8633a';
  ctx.fillRect(0, H * 0.9, W, H * 0.1);
}

// カメラの人を「壁に映る影」として実寸で描く（鏡映し）
function drawShadow(L, p) {
  const lm = pose.lm;
  if (!lm || !pose.hipsOk) return;
  let s = run.cal;
  if (!s) { s = liveScale(p.heightCm); if (!s) return; }
  const aspect = (video.videoWidth || 16) / (video.videoHeight || 9);
  const P = (i) => [
    L.figX - (lm[i].x - pose.hipX) * aspect * s.cmPerUnit * L.pxPerCm,
    L.y((s.floorY - lm[i].y) * s.cmPerUnit),
  ];
  const k = L.pxPerCm;
  ctx.save();
  ctx.strokeStyle = 'rgba(20,24,40,0.55)'; ctx.fillStyle = 'rgba(20,24,40,0.55)';
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  const seg = (a, b, w) => {
    const [x1, y1] = P(a), [x2, y2] = P(b);
    ctx.lineWidth = w * k; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  };
  // 胴体
  const [a1, b1] = P(11), [a2, b2] = P(12), [a3, b3] = P(24), [a4, b4] = P(23);
  ctx.beginPath(); ctx.moveTo(a1, b1); ctx.lineTo(a2, b2); ctx.lineTo(a3, b3); ctx.lineTo(a4, b4); ctx.closePath();
  ctx.lineWidth = 10 * k; ctx.stroke(); ctx.fill();
  // 手足
  seg(11, 13, 10); seg(13, 15, 8); seg(12, 14, 10); seg(14, 16, 8);
  seg(15, 19, 6); seg(16, 20, 6);
  seg(23, 25, 15); seg(25, 27, 11); seg(24, 26, 15); seg(26, 28, 11);
  seg(27, 31, 7); seg(28, 32, 7);
  // 首と頭
  const [nx, ny] = P(0);
  const mx = (a1 + a2) / 2, my = (b1 + b2) / 2;
  ctx.lineWidth = 9 * k; ctx.beginPath(); ctx.moveTo(mx, my); ctx.lineTo(nx, ny); ctx.stroke();
  ctx.beginPath(); ctx.arc(nx, ny - 3 * k, 11 * k, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
}

// 操作する人向けの小さなカメラ映像
function drawCamera() {
  if (pose.status !== 'ok' || state === 'title' || state === 'ranking') return;
  const vw = video.videoWidth || 16, vh = video.videoHeight || 9;
  const w = Math.min(W * 0.18, 300), h = w * vh / vw;
  const x = W - w - 16, y = H - h - 16;
  ctx.save();
  ctx.translate(x + w, y); ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, w, h);
  ctx.restore();
  const lm = pose.lm;
  if (lm) {
    ctx.strokeStyle = pose.full ? '#4cd964' : '#ffd23f'; ctx.lineWidth = 2;
    const bones = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28]];
    for (const [a, b] of bones) {
      ctx.beginPath();
      ctx.moveTo(x + (1 - lm[a].x) * w, y + lm[a].y * h);
      ctx.lineTo(x + (1 - lm[b].x) * w, y + lm[b].y * h);
      ctx.stroke();
    }
  }
  ctx.strokeStyle = pose.full ? '#fff' : '#e8452c'; ctx.lineWidth = 3;
  ctx.strokeRect(x, y, w, h);
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

// テスト用：合成したランドマークを流し込めるようにしておく
window.__feedPose = (lm) => processLandmarks(lm, performance.now());

resize();
renderNameInputs();
updateCamStatus();
initCamera();
requestAnimationFrame((n) => { lastTime = n; loop(n); });
