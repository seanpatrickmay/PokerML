/* ===============================================================
   HUNL Poker — CFR Solver Frontend
   Left panel : play against the bot
   Right panel: 13x13 range grid + per-hand strategy detail
   =============================================================== */

let sessionId = null;
let handsPlayed = 0;
let totalProfit = 0;
let rangeData = null;      // latest /range response
let selectedCell = null;   // "row,col"
let myGridPos = null;      // {row,col} of player's actual hand

const RANKS  = ['A','K','Q','J','T','9','8','7','6','5','4','3','2'];
function actLabel(a) {
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
function actColor(a) {
  if (a==='f') return '#c62828';
  if (a==='k') return '#2e7d32';
  if (a==='c') return '#1565c0';
  if (a==='a') return '#6a1b9a';
  if (a.startsWith('b')) {
    const pct = parseInt(a.slice(1));
    if (pct <= 50)  return '#e65100';
    if (pct <= 100) return '#f57c00';
    return '#ff6d00';  // overbets
  }
  return '#666';
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

// ---- API ----

async function api(method, path, body) {
  const opts = {method, headers:{'Content-Type':'application/json'}};
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  if (!res.ok) { const e = await res.json().catch(()=>({})); throw new Error(e.detail||res.statusText); }
  return res.json();
}

// ---- Card rendering ----

function cardEl(text) {
  if (!text) return emptyCardEl();
  const rank = text.slice(0, -1), suit = text.slice(-1);
  const red = '\u2662\u2661\u2666\u2665'.includes(suit);
  const el = document.createElement('div');
  el.className = 'card face' + (red ? ' red' : '');
  el.innerHTML = `<span class="rank">${rank}</span><span class="suit">${suit}</span>`;
  return el;
}
function backCardEl() { const d=document.createElement('div');d.className='card back';return d; }
function emptyCardEl(){ const d=document.createElement('div');d.className='card empty';return d; }

function renderHand(id, cards, opts={}) {
  const el = document.getElementById(id); el.innerHTML='';
  if (!cards||!cards.length) { if(opts.backs){el.appendChild(backCardEl());el.appendChild(backCardEl());} return; }
  for (const c of cards) el.appendChild(cardEl(c));
}
function renderBoard(cards) {
  const el=document.getElementById('board'); el.innerHTML='';
  for(let i=0;i<5;i++) el.appendChild(cards&&i<cards.length ? cardEl(cards[i]) : emptyCardEl());
}

// ---- Determine grid position for player's hand ----

function handToGrid(cards) {
  if (!cards||cards.length<2) return null;
  const r1 = RANKS.indexOf(cards[0][0]);
  const r2 = RANKS.indexOf(cards[1][0]);
  const s1 = cards[0].slice(-1), s2 = cards[1].slice(-1);
  const suited = s1===s2;
  if (r1===r2) return {row:r1,col:r1};
  const hi=Math.min(r1,r2), lo=Math.max(r1,r2);
  return suited ? {row:hi,col:lo} : {row:lo,col:hi};
}

// ---- Game UI update ----

function updateGame(data) {
  renderHand('player-hand', data.player_hand);
  renderBoard(data.board);
  document.getElementById('pot-display').textContent = `Pot: ${data.pot}`;
  document.getElementById('street-display').textContent = data.street_name||'Preflop';
  document.getElementById('player-stack').textContent = data.player_stack;
  document.getElementById('bot-stack').textContent = data.bot_stack;
  document.getElementById('player-bet-display').textContent = data.player_bet ? `${data.player_bet}` : '';
  document.getElementById('bot-bet-display').textContent = data.bot_bet ? `${data.bot_bet}` : '';

  if (data.is_over && data.bot_hand) renderHand('bot-hand', data.bot_hand);
  else renderHand('bot-hand', null, {backs:true});

  // Render action buttons dynamically from legal_actions
  const actBox = document.getElementById('actions'); actBox.innerHTML = '';
  const legal = data.legal_actions || [];
  for (const a of legal) {
    const btn = document.createElement('button');
    btn.className = 'act ' + actBtnClass(a);
    btn.textContent = actLabel(a);
    btn.onclick = () => doAction(a);
    if (data.is_over) btn.disabled = true;
    actBox.appendChild(btn);
  }

  if (data.bot_actions) for (const a of data.bot_actions) addLog(`Bot: ${ACT_LBL[a]||a}`,'bot');

  const banner = document.getElementById('result-banner');
  const nb = document.getElementById('new-hand-btn');
  if (data.is_over) {
    banner.classList.remove('hidden','win','lose','tie');
    const p = data.player_profit||0;
    banner.classList.add(data.winner==='player'?'win':data.winner==='bot'?'lose':'tie');
    banner.textContent = `${data.result_text}  (${p>=0?'+':''}${p} BB)`;
    nb.classList.remove('hidden');
    if (data.full_board) renderBoard(data.full_board);
    addLog(data.result_text,'info');
    handsPlayed++; totalProfit+=p;
    document.getElementById('stat-hands').textContent=handsPlayed;
    const sp=document.getElementById('stat-profit');
    sp.textContent=totalProfit.toFixed(1);
    sp.className=totalProfit>0?'pos':totalProfit<0?'neg':'';
  } else {
    banner.classList.add('hidden'); nb.classList.add('hidden');
  }

  myGridPos = handToGrid(data.player_hand);
}

function addLog(t,cls=''){
  const d=document.createElement('div');d.className='log-entry '+cls;d.textContent=t;
  const l=document.getElementById('log');l.appendChild(d);l.scrollTop=l.scrollHeight;
}

// ---- Range Grid ----

function buildGrid() {
  const g = document.getElementById('range-grid'); g.innerHTML='';
  for (let row=0;row<13;row++) {
    // row header
    const rh=document.createElement('div');rh.className='row-hdr';rh.textContent=RANKS[row];g.appendChild(rh);
    for (let col=0;col<13;col++) {
      const cell=document.createElement('div');
      cell.className='rcell';
      cell.dataset.rc=`${row},${col}`;
      // label
      let lbl;
      if(row===col) lbl=RANKS[row]+RANKS[col];
      else if(row<col) lbl=RANKS[row]+RANKS[col]+'s';
      else lbl=RANKS[col]+RANKS[row]+'o';
      cell.innerHTML=`<span class="lbl">${lbl}</span><span class="bar"></span>`;
      cell.onclick=()=>selectGridCell(row,col);
      g.appendChild(cell);
    }
  }
}

function updateGrid() {
  if (!rangeData||rangeData.status!=='ok') return;
  const actions = rangeData.actions;
  document.querySelectorAll('.rcell').forEach(cell=>{
    const rc=cell.dataset.rc;
    const hd=rangeData.hands[rc];
    cell.classList.remove('selected','mine','dead');

    if (!hd||!hd.probs) { cell.classList.add('dead'); cell.style.background=''; return; }

    // mini bar
    const bar = cell.querySelector('.bar');
    bar.innerHTML='';
    let pos=0;
    for(let i=0;i<hd.probs.length;i++){
      if(hd.probs[i]<0.005) continue;
      const seg=document.createElement('span');seg.className='bar-seg';
      seg.style.background=ACT_CLR[actions[i]]||'#666';
      seg.style.width=(hd.probs[i]*100)+'%';
      bar.appendChild(seg);
    }

    // tint background by dominant action
    let maxP=0,maxI=0;
    for(let i=0;i<hd.probs.length;i++) if(hd.probs[i]>maxP){maxP=hd.probs[i];maxI=i;}
    const baseClr = ACT_CLR[actions[maxI]]||'#333';
    cell.style.background = hexToRgba(baseClr, 0.15 + maxP*0.45);

    // highlight player's hand
    const [cr,cc] = rc.split(',').map(Number);
    if (myGridPos && myGridPos.row===cr && myGridPos.col===cc)
      cell.classList.add('mine');
  });

  if (selectedCell) {
    const el = document.querySelector(`.rcell[data-rc="${selectedCell}"]`);
    if (el) el.classList.add('selected');
  }

  // context
  const who = rangeData.current_player==='player' ? 'Your' : "Bot's";
  document.getElementById('strat-context').textContent =
    `${rangeData.street_name} | ${who} Turn | Pot: ${rangeData.pot}`;
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
  document.getElementById('detail-meta').textContent =
    `Bucket ${hd.bucket} | ${hd.combos} combo${hd.combos!==1?'s':''}`;

  const bars = document.getElementById('detail-bars'); bars.innerHTML='';
  const actions = rangeData.actions;
  for (let i=0;i<actions.length;i++) {
    const pct = (hd.probs[i]*100).toFixed(1);
    const clr = ACT_CLR[actions[i]]||'#666';
    const d = document.createElement('div'); d.className='dbar';
    d.innerHTML = `
      <span class="dbar-label" style="color:${clr}">${ACT_LBL[actions[i]]||actions[i]}</span>
      <span class="dbar-track"><span class="dbar-fill" style="width:${pct}%;background:${clr}"></span></span>
      <span class="dbar-pct">${pct}%</span>`;
    bars.appendChild(d);
  }
}

// ---- Actions ----

async function newGame() {
  try {
    const d = await api('POST','/api/game/new');
    sessionId = d.session_id;
    document.getElementById('log').innerHTML='';
    addLog('--- New Hand ---','info');
    updateGame(d);
    await fetchRange();
  } catch(e) { addLog('Error: '+e.message,'info'); }
}

async function doAction(action) {
  if (!sessionId) return;
  try {
    addLog(`You: ${ACT_LBL[action]||action}`,'player');
    const d = await api('POST',`/api/game/${sessionId}/action`,{action});
    updateGame(d);
    await fetchRange();
  } catch(e) { addLog('Error: '+e.message,'info'); }
}

async function fetchRange() {
  if (!sessionId) return;
  try {
    const d = await api('GET',`/api/game/${sessionId}/range`);
    if (d.status==='ok') { rangeData=d; updateGrid(); showDetail(); }
  } catch(e) { /* hand may be over */ }
}

// ---- Helpers ----

function hexToRgba(hex,a) {
  const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${a})`;
}

// ---- Init ----

window.addEventListener('DOMContentLoaded', ()=>{ buildGrid(); newGame(); });
