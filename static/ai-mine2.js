/* AI Mine 2 — isolated universal-miner dashboard page. */
let _mine2Results = null;
let _mine2Running = false;
let _mine2ProgressPoll = null;

function dashNavMine2(){
  dashPage = 'mine2';
  document.querySelectorAll('#dash-nav .dn-item').forEach(el=>{
    el.classList.toggle('active', el.dataset.page === 'mine2');
  });
  dashPageMine2(document.getElementById('dash-content'));
  _renderDashCollections();
}

function dashPageMine2(c){
  c.innerHTML = `
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
      <h2 style="margin:0">AI Mine 2</h2>
      <span style="font-size:10px;color:#00d4aa;border:1px solid rgba(0,212,170,.35);padding:3px 7px;border-radius:4px">UNIVERSAL PIPELINE</span>
      <span id="mine2-model" style="font-size:10px;color:var(--text-dim)">model: loading…</span>
    </div>
    <div class="dpanel" style="margin-bottom:14px">
      <div class="dpanel-title"><span class="dpanel-icon">⚡</span> Source Input</div>
      <div style="font-size:10px;color:var(--text-dim);margin-bottom:10px">One ontology and evidence contract for chat, narrative, documents, and papers. Auto only adapts source interpretation and segmentation.</div>
      <div id="mine2-drop" style="border:1px dashed var(--border);border-radius:6px;padding:18px;text-align:center;color:var(--text-dim);font-size:10px;cursor:pointer;margin-bottom:10px" onclick="document.getElementById('mine2-file-input').click()" ondragover="event.preventDefault()" ondrop="event.preventDefault();_mine2HandleFile(event.dataTransfer.files[0])">
        Drop a file here or click to browse · .json · .jsonl · .md · .txt · .pdf · .docx
        <input id="mine2-file-input" type="file" style="display:none" onchange="_mine2HandleFile(this.files[0])">
      </div>
      <div id="mine2-file-chip" style="display:none;font-size:10px;color:var(--text-dim);margin-bottom:7px"></div>
      <textarea id="mine2-text" placeholder="Paste any source here…" style="width:100%;min-height:230px;resize:vertical;background:rgba(255,255,255,.025);border:1px solid var(--border);border-radius:6px;color:var(--text);padding:12px;font-family:inherit;font-size:11px;box-sizing:border-box"></textarea>
      <div style="display:flex;flex-wrap:wrap;gap:10px;align-items:end;margin-top:10px;font-size:10px;color:var(--text-dim)">
        <label>Collection<br><select id="mine2-collection" class="mine2-input"><option value="default">default</option></select></label>
        <label>Source label<br><input id="mine2-source" class="mine2-input" value="universal"></label>
        <label>Profile<br><select id="mine2-kind" class="mine2-input"><option value="auto">Auto detect</option><option value="chat">Chat</option><option value="narrative">Narrative</option><option value="document">Document</option><option value="paper">Paper</option></select></label>
        <label>Min confidence<br><input id="mine2-conf" class="mine2-input" type="number" value="0.5" min="0" max="1" step="0.1" style="width:65px"></label>
        <label>Segment chars<br><input id="mine2-segment" class="mine2-input" type="number" value="0" min="0" max="20000" step="1000" style="width:80px" title="0 selects a model-aware default"></label>
        <button class="dbtn dbtn-grey" onclick="_mine2NewCollection()">+ New Collection</button>
        <div style="flex:1"></div>
        <button id="mine2-run" class="dbtn dbtn-blue" onclick="_mine2Run()">⚡ Run Universal Mine</button>
      </div>
    </div>
    <div id="mine2-progress" class="dpanel" style="display:none;margin-bottom:14px">
      <div style="display:flex;justify-content:space-between;font-size:10px;margin-bottom:8px"><span id="mine2-progress-label">Starting…</span><span id="mine2-progress-detail" style="color:var(--text-dim)"></span></div>
      <div style="height:7px;background:rgba(255,255,255,.05);border-radius:4px;overflow:hidden"><div id="mine2-progress-fill" style="height:100%;width:0;background:linear-gradient(90deg,#0AAC00,#00d4aa);transition:width .5s"></div></div>
    </div>
    <div id="mine2-status" style="font-size:10px;color:var(--text-dim);margin-bottom:10px">Ready. Mining creates proposals; nothing enters the collection until review and promotion.</div>
    <div id="mine2-results"></div>`;
  c.querySelectorAll('.mine2-input').forEach(el=>{
    el.style.cssText += ';background:rgba(255,255,255,.04);border:1px solid var(--border);border-radius:4px;color:var(--text);padding:5px 7px;font-size:10px;font-family:inherit';
  });
  _mine2LoadCollections();
  fetch(`${API}/mine-v2/config`).then(r=>r.json()).then(data=>{
    const el=document.getElementById('mine2-model');
    if(el)el.textContent=`model: ${data.extract_model}${data.hosted?' · hosted':''}`;
  }).catch(()=>{});
}

async function _mine2LoadCollections(){
  const sel=document.getElementById('mine2-collection');
  if(!sel)return;
  try{
    const data=await (await fetch(`${API}/collections`)).json();
    sel.innerHTML='';
    for(const col of data.collections||[]){
      const opt=document.createElement('option');
      opt.value=col.id; opt.textContent=`${col.id} (${col.anchors}a/${col.slabs}s/${col.bundles}b)${col.active?'':' [OFF]'}`;
      opt.selected=col.id===_selectedCollection; sel.appendChild(opt);
    }
    sel.onchange=()=>{_selectedCollection=sel.value};
  }catch(error){console.warn('AI Mine 2 collections',error)}
}

async function _mine2NewCollection(){
  const name=prompt('Collection name:');
  if(!name)return;
  try{
    const response=await fetch(`${API}/collections`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});
    if(!response.ok)throw new Error((await response.json()).detail||'Create failed');
    const result=await response.json(); _selectedCollection=result.id;
    await _mine2LoadCollections(); _renderDashCollections();
  }catch(error){alert(error.message)}
}

async function _mine2HandleFile(file){
  if(!file)return;
  if(file.size>10*1024*1024){alert('File too large (max 10 MB)');return}
  const chip=document.getElementById('mine2-file-chip');
  const textarea=document.getElementById('mine2-text');
  chip.style.display='block'; chip.textContent=`${file.name} — loading…`; chip.style.color='';
  const ext=(file.name.split('.').pop()||'').toLowerCase();
  if(new Set(['pdf','docx','doc']).has(ext)){
    try{
      const body=new FormData(); body.append('file',file);
      const response=await fetch(`${API}/upload?full=true`,{method:'POST',body});
      if(!response.ok)throw new Error((await response.json()).detail||'Extraction failed');
      const data=await response.json(); textarea.value=data.content||data.text||'';
      chip.textContent=`${file.name} · ${textarea.value.length.toLocaleString()} chars${data.truncated?' · TRUNCATED':''}`;
      if(data.truncated)chip.style.color='#ffc040';
    }catch(error){chip.textContent=`${file.name} — ${error.message}`;chip.style.color='#ff6b6b'}
  }else{
    const reader=new FileReader();
    reader.onload=()=>{textarea.value=reader.result;chip.textContent=`${file.name} · ${textarea.value.length.toLocaleString()} chars`};
    reader.readAsText(file);
  }
}

function _mine2StartProgress(){
  document.getElementById('mine2-progress').style.display='block';
  const poll=async()=>{
    try{
      const p=await (await fetch(`${API}/mining-progress`)).json();
      const pct=p.overall_pct||0;
      const labels={extracting:'Extracting grounded candidates',consolidating:'Consolidating across segments',edge_extraction:'Relating final candidates',done:'Done',error:'Error'};
      document.getElementById('mine2-progress-label').textContent=labels[p.phase]||p.phase||'Starting';
      document.getElementById('mine2-progress-detail').textContent=`${p.completed||0}/${p.total||0} · ${Math.round(p.elapsed_secs||0)}s`;
      document.getElementById('mine2-progress-fill').style.width=`${pct}%`;
    }catch(_error){}
  };
  poll(); _mine2ProgressPoll=setInterval(poll,1500);
}

function _mine2StopProgress(){
  if(_mine2ProgressPoll){clearInterval(_mine2ProgressPoll);_mine2ProgressPoll=null}
}

async function _mine2Run(){
  if(_mine2Running)return;
  const text=document.getElementById('mine2-text').value.trim();
  if(!text){alert('Paste or load a source first');return}
  _mine2Running=true;
  const button=document.getElementById('mine2-run'); button.disabled=true; button.textContent='Mining…';
  document.getElementById('mine2-status').textContent='Universal extraction is running…';
  _mine2StartProgress();
  try{
    const response=await fetch(`${API}/mine-v2`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
      text, source_label:document.getElementById('mine2-source').value||'universal',
      source_kind:document.getElementById('mine2-kind').value,
      min_confidence:Number(document.getElementById('mine2-conf').value||0.5),
      segment_chars:Number(document.getElementById('mine2-segment').value||0),
      target_collection:document.getElementById('mine2-collection').value||'default'
    })});
    if(!response.ok){const error=await response.json().catch(()=>({detail:response.statusText}));throw new Error(error.detail||'Mining failed')}
    _mine2Results=await response.json(); _mine2RenderResults();
  }catch(error){document.getElementById('mine2-status').innerHTML=`<span style="color:#ff6b6b">${_esc(error.message)}</span>`}
  finally{_mine2Running=false;button.disabled=false;button.textContent='⚡ Run Universal Mine';_mine2StopProgress()}
}

function _mine2RenderResults(){
  const result=_mine2Results||{}; const proposals=result.proposals||[]; const edges=result.edges||[];
  document.getElementById('mine2-status').innerHTML=`Detected <strong>${_esc(result.source_kind||'unknown')}</strong> · ${result.segments||0} segment(s) · ${_esc(result.extract_model||'')}`;
  const rows=proposals.map((p,index)=>{
    const label=p.canonical_phrase||p.title||'Untitled';
    const evidence=(p.source_refs&&p.source_refs[0])||{};
    const desc=p.description?`<div style="color:#00d4aa;margin-top:4px;font-style:italic">${_esc(p.description)}</div>`:'';
    return `<tr style="border-top:1px solid var(--border)"><td style="padding:9px"><input class="mine2-sel" data-idx="${index}" type="checkbox" checked></td><td style="padding:9px;color:${p.type==='anchor'?'#00d4aa':'#8aa4ff'}">${_esc(p.type)}</td><td style="padding:9px"><div style="color:var(--text);font-weight:600">${_esc(label)}</div>${desc}<div style="color:var(--text-dim);margin-top:4px">${_esc((p.canonical_text||evidence.quote||'').slice(0,240))}</div></td><td style="padding:9px">${Math.round((p.confidence||0)*100)}%</td></tr>`;
  }).join('');
  const edgeRows=edges.map(e=>`<div style="padding:7px 0;border-top:1px solid var(--border)"><span style="color:#e8a317">${_esc(e.type)}</span> ${_esc(e.from)} → ${_esc(e.to)}</div>`).join('');
  const warnings=(result.warnings||[]).map(w=>`<div>⚠ ${_esc(typeof w==='string'?w:(w.issue||JSON.stringify(w)))}</div>`).join('');
  document.getElementById('mine2-results').innerHTML=`
    <div class="dpanel" style="margin-bottom:14px"><div class="dpanel-title">Candidates (${proposals.length})</div>
      ${rows?`<table style="width:100%;border-collapse:collapse;font-size:10px"><thead><tr style="text-align:left;color:var(--text-dim)"><th></th><th>TYPE</th><th>CANDIDATE / EVIDENCE</th><th>CONF.</th></tr></thead><tbody>${rows}</tbody></table>`:'<div style="color:var(--text-dim);font-size:10px">No grounded candidates met the threshold.</div>'}
      ${rows?`<div style="display:flex;align-items:center;gap:10px;margin-top:12px"><button id="mine2-push" class="dbtn dbtn-green" onclick="_mine2PushSelected()">Push Selected to Drafts</button><span id="mine2-push-status" style="font-size:10px;color:var(--text-dim)">Review remains required before collection mutation.</span></div>`:''}
    </div>
    <div class="dpanel" style="margin-bottom:14px"><div class="dpanel-title">Relationships (${edges.length})</div><div style="font-size:10px">${edgeRows||'<span style="color:var(--text-dim)">No supported relationships.</span>'}</div></div>
    ${warnings?`<div class="dpanel"><div class="dpanel-title">Warnings</div><div style="font-size:10px;color:#ffc040">${warnings}</div></div>`:''}`;
}

async function _mine2PushSelected(){
  const status=document.getElementById('mine2-push-status');
  if(!await _ensureMiningSession(status))return;
  const indexes=[...document.querySelectorAll('.mine2-sel:checked')].map(el=>Number(el.dataset.idx));
  const proposals=indexes.map(index=>_mine2Results.proposals[index]).filter(Boolean);
  if(!proposals.length){alert('No candidates selected');return}
  const button=document.getElementById('mine2-push'); button.disabled=true; status.textContent=`Pushing ${proposals.length} proposal(s)…`;
  try{
    const response=await fetch(`${API}/sessions/${currentSessionId}/push-mined`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
      proposals, edges:_mine2Results.edges||[], target_collection:document.getElementById('mine2-collection').value||'default'
    })});
    if(!response.ok){const error=await response.json();throw new Error(error.detail||'Push failed')}
    const result=await response.json(); status.innerHTML=`<span style="color:#0AAC00">✓ Created ${result.created||0} draft(s)</span> · <a href="#" onclick="dashNav('verify');return false" style="color:var(--accent)">Verify & Promote</a>`; button.textContent='✓ Pushed';
  }catch(error){status.textContent=error.message;status.style.color='#ff6b6b';button.disabled=false}
}
