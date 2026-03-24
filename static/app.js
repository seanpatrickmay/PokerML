/* ===============================================================
   Poker — CFR Solver Frontend
   Left panel : N-seat table with live solver
   Right panel: 13x13 range grid + per-hand strategy detail
   =============================================================== */

let sessionId = null;
let handsPlayed = 0;
let totalProfit = 0;
let rangeData = null;
let selectedCell = null;
let myGridPos = null;
let myCards = null;
let gameData = null;
let playerSeat = 0;
let livePolling = null;
let animating = false;
let autoDealTimer = null;
let numPlayers = 6;

const RANKS = ['A','K','Q','J','T','9','8','7','6','5','4','3','2'];
const SUIT_SYMBOLS = {'s':'\u2660','h':'\u2665','d':'\u2666','c':'\u2663'};
const SUIT_HTML = {'s':'\u2660','h':'<span class="red">\u2665</span>','d':'<span class="red">\u2666</span>','c':'\u2663'};

const POSITION_NAMES = {
  2: {0:'BTN', 1:'BB'},
  3: {0:'BTN', 1:'SB', 2:'BB'},
  4: {0:'BTN', 1:'SB', 2:'BB', 3:'CO'},
  5: {0:'BTN', 1:'SB', 2:'BB', 3:'MP', 4:'CO'},
  6: {0:'BTN', 1:'SB', 2:'BB', 3:'UTG', 4:'MP', 5:'CO'},
  7: {0:'BTN', 1:'SB', 2:'BB', 3:'UTG', 4:'MP', 5:'HJ', 6:'CO'},
  8: {0:'BTN', 1:'SB', 2:'BB', 3:'UTG', 4:'UTG1', 5:'MP', 6:'HJ', 7:'CO'},
  9: {0:'BTN', 1:'SB', 2:'BB', 3:'UTG', 4:'UTG1', 5:'UTG2', 6:'MP', 7:'HJ', 8:'CO'},
};

let actionLabels = {};
let actionChips = {};

function actLabel(a) {
  if (actionLabels[a]) return actionLabels[a];
  if (a==='f') return 'Fold';
  if (a==='k') return 'Check';
  if (a==='c') return 'Call';
  if (a==='a') return 'All In';
  if (a.startsWith('b')) {
    const pct = parseInt(a.slice(1));
    if (pct === 100) return 'Bet Pot';
    return `Bet ${pct}%`;
  }
  return a;
}

// Short label for range grid tooltips (no chip amounts)
function actLabelShort(a) {
  if (a==='f') return 'Fold';
  if (a==='k') return 'Check';
  if (a==='c') return 'Call';
  if (a==='a') return 'All In';
  if (a.startsWith('b')) {
    const pct = parseInt(a.slice(1));
    if (pct === 100) return 'Pot';
    return `${pct}%`;
  }
  return a;
}
function actColor(a) {
  if (a==='f') return '#ef4444';
  if (a==='k') return '#22c55e';
  if (a==='c') return '#3b82f6';
  if (a==='a') return '#a855f7';
  if (a.startsWith('b')) {
    const pct = parseInt(a.slice(1));
    if (pct <= 50)  return '#f97316';
    if (pct <= 100) return '#eab308';
    return '#f59e0b';
  }
  return '#6b7280';
}
function actBtnClass(a) {
  if (a==='f') return 'fold';
  if (a==='k') return 'check';
  if (a==='c') return 'call';
  if (a==='a') return 'allin';
  return 'raise';
}
const ACT_LBL = new Proxy({}, {get:(_,k)=>actLabel(k)});
const ACT_CLR = new Proxy({}, {get:(_,k)=>actColor(k)});

// ---- Error handling ----

let errorToast = null;
function showError(msg) {
  if (!errorToast) {
    errorToast = document.createElement('div');
    errorToast.id = 'error-toast';
    errorToast.setAttribute('role', 'alert');
    document.body.appendChild(errorToast);
  }
  errorToast.textContent = msg;
  errorToast.classList.add('visible');
  setTimeout(() => errorToast.classList.remove('visible'), 4000);
}

// ---- API ----

async function api(method, path, body) {
  const opts = {method, headers:{'Content-Type':'application/json'}};
  if (body) opts.body = JSON.stringify(body);
  let res;
  try {
    res = await fetch(path, opts);
  } catch (e) {
    showError('Connection failed — is the server running?');
    throw e;
  }
  if (!res.ok) {
    const e = await res.json().catch(()=>({}));
    const msg = e.detail || res.statusText;
    showError(msg);
    throw new Error(msg);
  }
  return res.json();
}

// ---- Card rendering ----

function cardEl(text, animate) {
  if (!text) return emptyCardEl();
  const rank = text.slice(0, -1), suit = text.slice(-1);
  const red = '\u2662\u2661\u2666\u2665'.includes(suit);
  const el = document.createElement('div');
  el.className = 'card face' + (red ? ' red' : '') + (animate ? ' deal-in' : '');
  el.innerHTML = `<span class="rank">${rank}</span><span class="suit">${suit}</span>`;
  return el;
}
function backCardEl() { const d=document.createElement('div');d.className='card back';return d; }
function emptyCardEl(){ const d=document.createElement('div');d.className='card empty';return d; }

function renderBoard(cards, animate) {
  const el=document.getElementById('board'); el.innerHTML='';
  for(let i=0;i<5;i++) {
    const c = cards&&i<cards.length ? cardEl(cards[i], animate) : emptyCardEl();
    if (animate && cards && i < cards.length) c.style.animationDelay = `${i * 0.06}s`;
    el.appendChild(c);
  }
}

// ---- Card text helpers ----

function cardToSuitChar(card) {
  const s = card.slice(-1);
  if (s==='\u2660'||s==='s') return 's';
  if (s==='\u2665'||s==='\u2661'||s==='h') return 'h';
  if (s==='\u2666'||s==='\u2662'||s==='d') return 'd';
  if (s==='\u2663'||s==='c') return 'c';
  return s;
}

function suitHtml(suitChar) {
  return SUIT_HTML[suitChar] || suitChar;
}

// ---- Determine grid position for player's hand ----

function handToGrid(cards) {
  if (!cards||cards.length<2) return null;
  const r1 = RANKS.indexOf(cards[0][0]);
  const r2 = RANKS.indexOf(cards[1][0]);
  const s1 = cardToSuitChar(cards[0]), s2 = cardToSuitChar(cards[1]);
  const suited = s1===s2;
  if (r1===r2) return {row:r1,col:r1};
  const hi=Math.min(r1,r2), lo=Math.max(r1,r2);
  return suited ? {row:hi,col:lo} : {row:lo,col:hi};
}

// ---- Dynamic seat generation ----

function seatAngle(seat, n, heroSeat) {
  const relIdx = (seat - heroSeat + n) % n;
  return Math.PI/2 + 2*Math.PI * relIdx / n;
}

function getSeatPosition(seat, n, heroSeat) {
  const angle = seatAngle(seat, n, heroSeat);
  const x = 50 + 46 * Math.cos(angle);
  const y = 50 + 44 * Math.sin(angle);
  return { left: x+'%', top: y+'%' };
}

function buildSeats() {
  const felt = document.getElementById('table-felt');
  felt.querySelectorAll('.seat').forEach(el => el.remove());
  felt.querySelectorAll('.bet-chip-abs').forEach(el => el.remove());

  const n = numPlayers;
  for (let seat = 0; seat < n; seat++) {
    const pos = getSeatPosition(seat, n, playerSeat);
    const posName = POSITION_NAMES[n]?.[seat] || String(seat);
    const angle = seatAngle(seat, n, playerSeat);

    const seatEl = document.createElement('div');
    seatEl.className = 'seat';
    seatEl.id = `seat-${seat}`;
    seatEl.dataset.seat = seat;
    seatEl.style.left = pos.left;
    seatEl.style.top = pos.top;

    seatEl.innerHTML = `
      <div class="seat-info">
        <div class="seat-label">${posName}</div>
        <div class="stack">100.0</div>
      </div>
      <div class="hand" aria-label="${posName} cards"></div>
      <div class="status-badge" aria-live="polite"></div>
      <div class="seat-hud"></div>`;

    felt.appendChild(seatEl);

    const chipEl = document.createElement('div');
    chipEl.className = 'bet-chip-abs';
    chipEl.id = `chip-${seat}`;
    const cx = 50 + 30 * Math.cos(angle);
    const cy = 50 + 28 * Math.sin(angle);
    chipEl.style.left = cx + '%';
    chipEl.style.top = cy + '%';
    felt.appendChild(chipEl);
  }

  const btnAngle = seatAngle(0, n, playerSeat);
  const dBtn = document.getElementById('dealer-btn');
  const bx = 50 + 38 * Math.cos(btnAngle + 0.15);
  const by = 50 + 36 * Math.sin(btnAngle + 0.15);
  dBtn.style.left = bx + '%';
  dBtn.style.top = by + '%';
}

// ---- Settings ----

function populatePositionSelect() {
  const sel = document.getElementById('position-select');
  const names = POSITION_NAMES[numPlayers] || {};
  sel.innerHTML = '';
  for (const [seat, name] of Object.entries(names)) {
    const opt = document.createElement('option');
    opt.value = seat;
    opt.textContent = name;
    sel.appendChild(opt);
  }
  sel.value = '0';
}

function onNumPlayersChange() {
  numPlayers = parseInt(document.getElementById('num-players-select').value);
  populatePositionSelect();
  playerSeat = parseInt(document.getElementById('position-select').value);
  buildSeats();
  newGame();
}

function onPositionChange() {
  playerSeat = parseInt(document.getElementById('position-select').value);
  buildSeats();
  newGame();
}

// ---- Animation helpers ----

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function highlightSeat(seat, action) {
  const seatEl = document.getElementById(`seat-${seat}`);
  if (!seatEl) return;
  const bubble = document.createElement('div');
  bubble.className = 'action-bubble';
  bubble.textContent = actLabel(action);
  bubble.style.color = actColor(action);
  seatEl.appendChild(bubble);
  requestAnimationFrame(() => bubble.classList.add('show'));
  setTimeout(() => {
    bubble.classList.remove('show');
    bubble.classList.add('fade');
    setTimeout(() => bubble.remove(), 250);
  }, 350);
}

// ---- Game UI update ----

function updateSeats(data) {
  for (const s of data.seats) {
    const seatEl = document.getElementById(`seat-${s.seat}`);
    if (!seatEl) continue;

    seatEl.querySelector('.seat-label').textContent = s.position;
    seatEl.querySelector('.stack').textContent = s.stack.toFixed(1);

    const chipEl = document.getElementById(`chip-${s.seat}`);
    if (chipEl) chipEl.textContent = s.bet > 0 ? s.bet.toFixed(1) : '';

    const handEl = seatEl.querySelector('.hand');
    handEl.innerHTML = '';
    if (s.cards) {
      for (const c of s.cards) handEl.appendChild(cardEl(c));
    } else if (s.active) {
      handEl.appendChild(backCardEl());
      handEl.appendChild(backCardEl());
    }

    const badge = seatEl.querySelector('.status-badge');
    if (!s.active) {
      badge.textContent = 'Folded';
      badge.style.display = 'block';
    } else if (s.all_in) {
      badge.textContent = 'All In';
      badge.style.display = 'block';
    } else {
      badge.style.display = 'none';
    }

    seatEl.classList.toggle('folded', !s.active);
    seatEl.classList.toggle('all-in', s.all_in);
    seatEl.classList.toggle('is-human', s.is_human);
    seatEl.classList.toggle('is-acting',
      !data.is_over && data.current_player === s.seat && s.active);
  }
}

async function updateGame(data, skipAnim=false) {
  gameData = data;
  playerSeat = data.player_seat;

  if (autoDealTimer) { clearTimeout(autoDealTimer); autoDealTimer = null; }

  // Update board, seats, pot FIRST so player sees new street immediately
  updateSeats(data);
  renderBoard(data.board, !skipAnim);
  document.getElementById('pot-display').textContent = `Pot: ${data.pot}`;
  document.getElementById('street-display').textContent = data.street_name || 'Preflop';

  // Then animate bot actions (non-blocking for UI)
  const botActions = data.bot_actions || [];
  if (botActions.length > 0) {
    if (skipAnim) {
      for (const ba of botActions)
        addLog(`${ba.position}: ${ACT_LBL[ba.action]||ba.action}`, 'bot');
    } else {
      animating = true;
      for (let i = 0; i < botActions.length; i++) {
        const ba = botActions[i];
        highlightSeat(ba.seat, ba.action);
        addLog(`${ba.position}: ${ACT_LBL[ba.action]||ba.action}`, 'bot');
        await sleep(350);
      }
      animating = false;
    }
  }

  // Action buttons
  actionLabels = data.action_labels || {};
  actionChips = data.action_chips || {};
  const actBox = document.getElementById('actions');
  actBox.innerHTML = '';
  const legal = data.legal_actions || [];

  if (legal.length > 0) {
    for (const a of legal) {
      const btn = document.createElement('button');
      btn.className = 'act ' + actBtnClass(a);
      btn.textContent = actLabel(a);
      btn.type = 'button';
      btn.onclick = () => doAction(a);
      if (data.is_over) btn.disabled = true;
      actBox.appendChild(btn);
    }
  } else if (!data.is_over) {
    const w = document.createElement('span');
    w.className = 'waiting-text';
    w.textContent = 'Waiting for opponent...';
    actBox.appendChild(w);
  }

  // Info bar: equity, pot odds, SPR
  updateInfoBar(data);

  // Track player's cards
  const humanSeat = data.seats.find(s => s.is_human);
  myCards = humanSeat?.cards || null;
  myGridPos = handToGrid(myCards);

  // Result
  const banner = document.getElementById('result-banner');
  if (data.is_over) {
    stopLivePolling();
    banner.classList.remove('hidden','win','lose','tie');
    const p = data.player_profit || 0;
    const cls = p > 0 ? 'win' : p < 0 ? 'lose' : 'tie';
    banner.classList.add(cls);
    banner.textContent = `${data.result_text}  (${p>=0?'+':''}${p} BB)`;
    if (data.full_board) renderBoard(data.full_board, true);
    addLog(data.result_text, 'info');
    handsPlayed++;
    totalProfit += p;
    document.getElementById('stat-hands').textContent = handsPlayed;
    const sp = document.getElementById('stat-profit');
    sp.textContent = totalProfit.toFixed(1);
    sp.className = totalProfit > 0 ? 'pos' : totalProfit < 0 ? 'neg' : '';
    document.getElementById('live-solver').classList.add('hidden');

    // Show Deal Next button instead of auto-dealing
    const dealBtn = document.createElement('button');
    dealBtn.className = 'act check';
    dealBtn.textContent = 'Deal Next (N)';
    dealBtn.type = 'button';
    dealBtn.onclick = () => newGame();
    actBox.appendChild(dealBtn);
  } else {
    banner.classList.add('hidden');
    if (data.current_turn === 'player') {
      startLivePolling();
    } else {
      stopLivePolling();
      document.getElementById('live-solver').classList.add('hidden');
    }
  }
}

function addLog(t, cls='') {
  const d = document.createElement('div');
  d.className = 'log-entry ' + cls;
  d.textContent = t;
  const l = document.getElementById('log');
  l.appendChild(d);
  l.scrollTop = l.scrollHeight;
}

// ---- Info Bar (equity, pot odds, SPR) ----

function updateInfoBar(data) {
  const bar = document.getElementById('info-bar');
  if (!data || data.is_over) {
    bar.classList.add('hidden');
    return;
  }

  bar.classList.remove('hidden');
  const equity = data.hand_equity;
  const fill = document.getElementById('equity-fill');
  const txt = document.getElementById('equity-text');

  if (equity != null) {
    fill.style.width = equity + '%';
    txt.textContent = equity.toFixed(0) + '%';
    // Color: red < 30, yellow 30-60, green > 60
    if (equity < 30) fill.style.background = '#ef4444';
    else if (equity < 60) fill.style.background = '#eab308';
    else fill.style.background = '#22c55e';
  } else {
    fill.style.width = '0%';
    txt.textContent = '—';
  }

  // Pot info: pot, to-call, pot odds, SPR
  const parts = [];
  parts.push(`<span class="pot-stat"><b>${data.pot}</b> pot</span>`);
  if (data.to_call > 0) {
    parts.push(`<span class="pot-stat"><b>${data.to_call}</b> to call</span>`);
    const potOdds = data.pot / data.to_call;
    const neededPct = (data.to_call / (data.pot + data.to_call) * 100).toFixed(0);
    parts.push(`<span class="pot-stat"><b>${potOdds.toFixed(1)}:1</b> odds (need ${neededPct}%)</span>`);
  }
  if (data.spr != null && data.spr < 100) {
    parts.push(`<span class="pot-stat">SPR <b>${data.spr}</b></span>`);
  }
  document.getElementById('pot-info').innerHTML = parts.join('');
}

// ---- HUD (VPIP/PFR on bot seats) ----

async function pollHudStats() {
  try {
    const stats = await api('GET', '/api/stats');
    for (const [seat, s] of Object.entries(stats)) {
      const seatNum = parseInt(seat);
      if (seatNum === playerSeat) continue; // skip human
      const hud = document.querySelector(`#seat-${seatNum} .seat-hud`);
      if (!hud) continue;
      if (s.hands < 3) { hud.classList.remove('visible'); continue; }

      const vpip = s.vpip.toFixed(0);
      const pfr = s.pfr.toFixed(0);
      let tag = '', tagClass = '';
      if (s.vpip < 18) { tag = 'Tight'; tagClass = 'range-tight'; }
      else if (s.vpip < 28) { tag = 'Reg'; tagClass = 'range-medium'; }
      else { tag = 'Wide'; tagClass = 'range-wide'; }

      hud.innerHTML = `V:${vpip} P:${pfr} <span class="range-tag ${tagClass}">${tag}</span>`;
      hud.classList.add('visible');
    }
  } catch(e) { /* ignore */ }
}

// ---- Live Solver ----

function startLivePolling() {
  stopLivePolling();
  pollLive();
  livePolling = setInterval(pollLive, 1500);
}

function stopLivePolling() {
  if (livePolling) { clearInterval(livePolling); livePolling = null; }
}

async function pollLive() {
  if (!sessionId) return;
  try {
    const d = await api('GET', `/api/game/${sessionId}/live`);
    updateLiveSolver(d);
    // Always refresh range when solver is active (solving or just finished)
    if (d && (d.status === 'solving' || d.status === 'done')) await fetchRange();
  } catch(e) { /* ignore polling errors */ }
}

function updateLiveSolver(data) {
  const panel = document.getElementById('live-solver');

  if (!data || data.status === 'idle' || data.status === 'unavailable') {
    panel.classList.add('hidden');
    return;
  }

  panel.classList.remove('hidden');
  const bars = document.getElementById('live-solver-bars');
  bars.innerHTML = '';

  const actions = data.actions || [];
  const probs = data.probs || [];

  for (let i = 0; i < actions.length; i++) {
    const pct = (probs[i] * 100).toFixed(1);
    const clr = actColor(actions[i]);
    const d = document.createElement('div');
    d.className = 'lsbar';
    d.innerHTML = `
      <span class="lsbar-label" style="color:${clr}">${actLabelShort(actions[i])}</span>
      <span class="lsbar-track"><span class="lsbar-fill" style="width:${pct}%;background:${clr}"></span></span>
      <span class="lsbar-pct">${pct}%</span>`;
    bars.appendChild(d);
  }

  // Mode and confidence display
  const metaEl = document.getElementById('live-solver-meta');
  const label = document.getElementById('live-solver-label');
  if (data.mode === 'preflop_ranges') {
    label.textContent = 'GTO Preflop Ranges';
    metaEl.textContent = '';
  } else {
    label.textContent = 'Live Solver';
    const conf = data.confidence || 0;
    const confLabel = conf >= 80 ? 'high' : conf >= 40 ? 'medium' : 'low';
    const status = data.status === 'solving' ? 'solving...' : 'done';
    metaEl.textContent = `${data.iterations} iters | ${confLabel} confidence (${status})`;
  }
}

// ---- Range Grid ----

function buildGrid() {
  const g = document.getElementById('range-grid'); g.innerHTML='';
  for (let row=0;row<13;row++) {
    const rh=document.createElement('div');rh.className='row-hdr';rh.textContent=RANKS[row];g.appendChild(rh);
    for (let col=0;col<13;col++) {
      const cell=document.createElement('div');
      cell.className='rcell';
      cell.dataset.rc=`${row},${col}`;
      cell.tabIndex = 0;
      cell.setAttribute('role', 'gridcell');
      let lbl;
      if(row===col) lbl=RANKS[row]+RANKS[col];
      else if(row<col) lbl=RANKS[row]+RANKS[col]+'s';
      else lbl=RANKS[col]+RANKS[row]+'o';
      cell.setAttribute('aria-label', lbl);
      cell.innerHTML=`<span class="lbl">${lbl}</span><span class="bar"></span>`;
      cell.onclick=()=>selectGridCell(row,col);
      cell.onkeydown=(e)=>{
        if (e.key==='Enter'||e.key===' ') { e.preventDefault(); selectGridCell(row,col); }
        else if (e.key==='ArrowRight'&&col<12) focusCell(row,col+1);
        else if (e.key==='ArrowLeft'&&col>0) focusCell(row,col-1);
        else if (e.key==='ArrowDown'&&row<12) focusCell(row+1,col);
        else if (e.key==='ArrowUp'&&row>0) focusCell(row-1,col);
      };
      g.appendChild(cell);
    }
  }
}

function focusCell(row,col) {
  const el = document.querySelector(`.rcell[data-rc="${row},${col}"]`);
  if (el) el.focus();
}

function updateGrid() {
  if (!rangeData||rangeData.status!=='ok') return;
  const actions = rangeData.actions;
  document.querySelectorAll('.rcell').forEach(cell=>{
    const rc=cell.dataset.rc;
    const hd=rangeData.hands[rc];
    cell.classList.remove('selected','mine','dead');

    if (!hd||!hd.probs) { cell.classList.add('dead'); cell.style.background=''; cell.removeAttribute('data-tooltip'); return; }

    const bar = cell.querySelector('.bar');
    bar.innerHTML='';
    let tooltip = hd.label + ': ';
    for(let i=0;i<hd.probs.length;i++){
      if(hd.probs[i]<0.005) continue;
      const seg=document.createElement('span');seg.className='bar-seg';
      seg.style.background=ACT_CLR[actions[i]]||'#6b7280';
      seg.style.width=(hd.probs[i]*100)+'%';
      bar.appendChild(seg);
      tooltip += `${actLabelShort(actions[i])} ${(hd.probs[i]*100).toFixed(0)}% `;
    }
    cell.dataset.tooltip = tooltip.trim();

    let maxP=0,maxI=0;
    for(let i=0;i<hd.probs.length;i++) if(hd.probs[i]>maxP){maxP=hd.probs[i];maxI=i;}
    const baseClr = ACT_CLR[actions[maxI]]||'#374151';
    cell.style.background = hexToRgba(baseClr, 0.12 + maxP*0.4);

    const [cr,cc] = rc.split(',').map(Number);
    if (myGridPos && myGridPos.row===cr && myGridPos.col===cc)
      cell.classList.add('mine');
  });

  // Auto-select player's hand cell
  if (myGridPos) {
    selectedCell = `${myGridPos.row},${myGridPos.col}`;
    const el = document.querySelector(`.rcell[data-rc="${selectedCell}"]`);
    if (el) el.classList.add('selected');
  }

  if (selectedCell && !myGridPos) {
    const el = document.querySelector(`.rcell[data-rc="${selectedCell}"]`);
    if (el) el.classList.add('selected');
  }

  const who = rangeData.current_player==='player' ? 'Your' : "Bot's";
  const posName = rangeData.position_name || '';
  document.getElementById('strat-context').textContent =
    `${rangeData.street_name} | ${who} Turn (${posName}) | Pot: ${rangeData.pot}`;

  showDetail();
}

function selectGridCell(row,col) {
  selectedCell = `${row},${col}`;
  document.querySelectorAll('.rcell.selected').forEach(e=>e.classList.remove('selected'));
  const el = document.querySelector(`.rcell[data-rc="${selectedCell}"]`);
  if (el) el.classList.add('selected');
  showDetail();
}

function showDetail() {
  const panel = document.getElementById('detail');
  if (!rangeData||rangeData.status!=='ok'||!selectedCell) { panel.classList.add('hidden'); return; }
  const hd = rangeData.hands[selectedCell];
  if (!hd||!hd.probs) { panel.classList.add('hidden'); return; }

  panel.classList.remove('hidden');
  document.getElementById('detail-title').textContent = hd.label;

  const meta = hd.bucket != null
    ? `Bucket ${hd.bucket} | ${hd.combos} combo${hd.combos!==1?'s':''}`
    : `${hd.combos} combo${hd.combos!==1?'s':''}`;
  document.getElementById('detail-meta').textContent = meta;

  const bars = document.getElementById('detail-bars'); bars.innerHTML='';
  const actions = rangeData.actions;
  for (let i=0;i<actions.length;i++) {
    const pct = (hd.probs[i]*100).toFixed(1);
    const clr = ACT_CLR[actions[i]]||'#6b7280';
    const d = document.createElement('div'); d.className='dbar';
    d.innerHTML = `
      <span class="dbar-label" style="color:${clr}">${ACT_LBL[actions[i]]||actions[i]}</span>
      <span class="dbar-track"><span class="dbar-fill" style="width:${pct}%;background:${clr}"></span></span>
      <span class="dbar-pct">${pct}%</span>`;
    bars.appendChild(d);
  }

  const combosEl = document.getElementById('detail-combos');
  combosEl.innerHTML = '';

  if (hd.suit_combos && hd.suit_combos.length > 0) {
    const sorted = [...hd.suit_combos];
    if (myCards) {
      sorted.sort((a, b) => {
        const aMatch = isMyCombo(a);
        const bMatch = isMyCombo(b);
        if (aMatch && !bMatch) return -1;
        if (!aMatch && bMatch) return 1;
        return 0;
      });
    }

    for (const combo of sorted) {
      const mine = myCards && isMyCombo(combo);
      const el = document.createElement('div');
      el.className = 'combo-item' + (mine ? ' mine' : '');
      el.innerHTML = comboHtml(combo);
      combosEl.appendChild(el);
    }
  }
}

function isMyCombo(combo) {
  if (!myCards || combo.length < 2) return false;
  return (combo[0]===myCards[0] && combo[1]===myCards[1]) ||
         (combo[0]===myCards[1] && combo[1]===myCards[0]);
}

function comboHtml(combo) {
  let html = '';
  for (const card of combo) {
    const rank = card.slice(0,-1);
    const suit = card.slice(-1);
    const sChar = cardToSuitChar(card);
    const isRed = sChar==='h'||sChar==='d';
    html += `<span class="combo-card${isRed?' red':''}">${rank}${suit}</span>`;
  }
  return html;
}

// ---- Actions ----

async function newGame() {
  stopLivePolling();
  if (autoDealTimer) { clearTimeout(autoDealTimer); autoDealTimer = null; }

  const seat = playerSeat;
  try {
    const d = await api('POST','/api/game/new',{
      player_seat: seat,
      num_players: numPlayers,
    });
    sessionId = d.session_id;
    buildSeats();
    document.getElementById('log').innerHTML='';
    addLog('--- New Hand ---','info');
    document.getElementById('result-banner').classList.add('hidden');
    await updateGame(d);
    await fetchRange();
    pollHudStats();
  } catch(e) { addLog('Error: '+e.message,'info'); }
}

async function doAction(action) {
  if (!sessionId || animating) return;
  stopLivePolling();

  if (gameData) {
    const mySeat = gameData.seats.find(s => s.is_human);
    if (mySeat) {
      const chips = computeChips(action, mySeat);
      if (chips > 0) {
        const chipEl = document.getElementById(`chip-${mySeat.seat}`);
        if (chipEl) chipEl.textContent = (mySeat.bet + chips).toFixed(1);
        const seatEl = document.getElementById(`seat-${mySeat.seat}`);
        if (seatEl) seatEl.querySelector('.stack').textContent = (mySeat.stack - chips).toFixed(1);
      }
    }
    document.querySelectorAll('#actions .act').forEach(b => b.disabled = true);
  }

  try {
    addLog(`You: ${ACT_LBL[action]||action}`,'player');
    const d = await api('POST',`/api/game/${sessionId}/action`,{action});
    await updateGame(d, action === 'f');
    await fetchRange();
  } catch(e) { addLog('Error: '+e.message,'info'); }
}

function computeChips(action, seat) {
  if (action === 'f' || action === 'k') return 0;
  if (!gameData) return 0;
  const maxBet = Math.max(...gameData.seats.map(s => s.bet));
  const toCall = maxBet - seat.bet;
  if (action === 'c') return Math.min(toCall, seat.stack);
  if (action === 'a') return seat.stack;
  if (action.startsWith('b')) {
    const frac = parseInt(action.slice(1)) / 100;
    if (toCall > 0) {
      const effPot = gameData.pot + toCall;
      const raiseAmt = Math.round(frac * effPot * 10) / 10;
      return Math.min(toCall + raiseAmt, seat.stack);
    } else {
      return Math.min(Math.round(frac * gameData.pot * 10) / 10, seat.stack);
    }
  }
  return 0;
}

async function fetchRange() {
  if (!sessionId) return;
  try {
    const d = await api('GET',`/api/game/${sessionId}/range`);
    if (d.status==='ok') { rangeData=d; updateGrid(); }
  } catch(e) { /* hand may be over */ }
}

// ---- Helpers ----

function hexToRgba(hex,a) {
  const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${a})`;
}

// ---- Keyboard shortcuts ----

document.addEventListener('keydown', (e) => {
  // Number keys 1-9 map to action buttons
  if (e.key >= '1' && e.key <= '9' && !e.ctrlKey && !e.metaKey && !e.altKey) {
    const idx = parseInt(e.key) - 1;
    const btns = document.querySelectorAll('#actions .act:not(:disabled)');
    if (idx < btns.length) {
      e.preventDefault();
      btns[idx].click();
    }
  }
  // 'n' for new hand
  if (e.key === 'n' && !e.ctrlKey && !e.metaKey && !e.altKey) {
    const active = document.activeElement;
    if (active.tagName !== 'SELECT' && active.tagName !== 'INPUT') {
      e.preventDefault();
      newGame();
    }
  }
});

// ---- Init ----

window.addEventListener('DOMContentLoaded', () => {
  const npSel = document.getElementById('num-players-select');
  npSel.addEventListener('change', onNumPlayersChange);
  document.getElementById('position-select').addEventListener('change', onPositionChange);
  document.getElementById('new-hand-btn').addEventListener('click', () => newGame());

  numPlayers = parseInt(npSel.value);
  populatePositionSelect();
  playerSeat = parseInt(document.getElementById('position-select').value);

  buildGrid();
  buildSeats();
  newGame();
});
