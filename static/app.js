'use strict';
const $ = id => document.getElementById(id);
let state = {students:[],sessions:[]}, token = '', stream = null, scanning = false, generation = 0;
let captures = [], objectURL = null;
const video = $('video'), overlay = $('overlay');
const escapeHTML = text => String(text ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function notice(text, error=false) { $('notice').textContent=text; $('notice').classList.toggle('error',error); }
async function api(path, body) {
  const options = body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json','X-App-Token':token},body:JSON.stringify(body)};
  const response = await fetch(path,options);
  const data = await response.json();
  if(!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
async function guarded(action) { try { await action(); } catch(e) { notice(e.message,true); } }
function activeSession() { return state.sessions.find(s=>!s.ended); }
async function refresh() {
  state = await api('/api/state'); token=state.token;
  $('stat-students').textContent=state.students.length;
  $('student-count').textContent=state.students.length;
  $('model-status').textContent=state.model_ready?'MobileFaceNet ready':'Model not loaded';
  $('model-status').classList.toggle('ready',state.model_ready);
  const active=activeSession();
  $('active-session').textContent=active?`Active: ${active.name}`:'No active session.';
  $('stat-session').textContent=active?active.name:'No active session';
  $('session-start').disabled=!!active; $('session-end').disabled=!active;
  $('student-list').innerHTML=state.students.length?state.students.map(s=>`<div class="student"><div><strong>${escapeHTML(s.name)}</strong><small>${escapeHTML(s.id)} · ${s.photos} photo${s.photos===1?'':'s'}</small></div><div class="student-actions"><label class="button compact ${active?'disabled':''}">Edit photos<input type="file" accept="image/jpeg,image/png" multiple hidden data-edit-photos="${escapeHTML(s.id)}" ${active?'disabled':''}></label><button class="danger" data-remove="${escapeHTML(s.id)}" ${active?'disabled':''}>Remove</button></div></div>`).join(''):'<p class="empty">No students enrolled.</p>';
  $('enroll-form').querySelector('button[type=submit]').disabled=!!active;
  const previous=$('session-select').value;
  $('session-select').innerHTML=state.sessions.length?state.sessions.map(s=>`<option value="${escapeHTML(s.id)}">${escapeHTML(s.name)} · ${s.ended?'ended':'active'}</option>`).join(''):'<option value="">No sessions yet</option>';
  if(state.sessions.some(s=>s.id===previous)) $('session-select').value=previous;
  if(state.missing_team_files?.length) {
    notice('Recognition files are missing. Add the required model files, then reload the page: '+state.missing_team_files.join(', '),true);
  } else if(state.model_error) {
    notice('The recognition model could not load: '+state.model_error,true);
  } else if(state.model_ready) {
    notice(state.students.length ? 'Model ready. Start a session when you are ready to take attendance.' : 'Model ready. Add students from the Students section to build the face gallery.');
  } else {
    notice('Loading recognition model…');
  }
  await loadReport();
  if(active) {
    const data=await api('/api/report?session_id='+encodeURIComponent(active.id));
    $('stat-present').textContent=data.rows.filter(r=>r.status==='present').length;
  } else $('stat-present').textContent='0';
}
async function loadReport() {
  const sid=$('session-select').value;
  $('download-csv').classList.toggle('disabled',!sid);
  if(!sid) { $('attendance-rows').innerHTML='<tr><td colspan="5">Start a session to create attendance records.</td></tr>'; return; }
  $('download-csv').href='/api/report?session_id='+encodeURIComponent(sid)+'&csv=1';
  const data=await api('/api/report?session_id='+encodeURIComponent(sid));
  $('attendance-rows').innerHTML=data.rows.map(r=>`<tr><td>${escapeHTML(r.student_id)}</td><td>${escapeHTML(r.name)}</td><td><span class="tag ${r.status==='present'?'':'unknown'}">${escapeHTML(r.status)}</span></td><td>${r.first_seen?escapeHTML(new Date(r.first_seen).toLocaleString()):'—'}</td><td>${r.distance===null?'—':(1-r.distance).toFixed(3)}</td></tr>`).join('');
}
document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>{
  document.querySelectorAll('.view').forEach(section=>section.hidden=section.id!=='view-'+button.dataset.view);
  document.querySelectorAll('.nav').forEach(b=>b.classList.toggle('active',b===button));
  $('page-title').textContent=button.textContent.slice(2).trim();
});
function frameData(source=video) {
  const width=source.videoWidth||source.naturalWidth, height=source.videoHeight||source.naturalHeight;
  if(!width||!height) throw new Error('Start the camera and wait for the video before capturing.');
  const scale=Math.min(1,960/width,960/height);
  const canvas=document.createElement('canvas');canvas.width=Math.round(width*scale);canvas.height=Math.round(height*scale);
  canvas.getContext('2d').drawImage(source,0,0,canvas.width,canvas.height);
  return {data:canvas.toDataURL('image/jpeg',.85),width:canvas.width,height:canvas.height};
}
function drawResults(data,frame) {
  $('stat-faces').textContent=data.faces.length;
  $('detections').innerHTML=data.faces.length?data.faces.map(face=>`<div class="detection"><strong>${escapeHTML(face.name)}</strong><span class="tag ${face.student_id?'':'unknown'}">${face.student_id?(face.confirmed?'Confirmed':'Known'):'Unknown'}</span></div>`).join(''):'<p class="empty">No faces detected.</p>';
  overlay.width=frame.width; overlay.height=frame.height;
  const stage=overlay.parentElement,scale=Math.min(stage.clientWidth/frame.width,stage.clientHeight/frame.height);
  overlay.style.width=frame.width*scale+'px';overlay.style.height=frame.height*scale+'px';
  overlay.style.left=(stage.clientWidth-frame.width*scale)/2+'px';overlay.style.top=(stage.clientHeight-frame.height*scale)/2+'px';
  const ctx=overlay.getContext('2d');ctx.lineWidth=3;ctx.font='16px system-ui';
  for(const face of data.faces){const [x1,y1,x2,y2]=face.bbox;ctx.strokeStyle=face.student_id?'#58e7b4':'#ffbb6a';ctx.strokeRect(x1,y1,x2-x1,y2-y1);const text=face.name;const w=ctx.measureText(text).width+12;ctx.fillStyle=ctx.strokeStyle;ctx.fillRect(x1,Math.max(0,y1-25),w,24);ctx.fillStyle='#102536';ctx.fillText(text,x1+6,Math.max(18,y1-7));}
}
function stop() {
  scanning=false;generation++;
  if(stream) stream.getTracks().forEach(track=>track.stop());stream=null;
  video.pause();video.srcObject=null;video.removeAttribute('src');video.load();
  if(objectURL) URL.revokeObjectURL(objectURL);objectURL=null;
  $('camera-status').textContent='Camera off';
  overlay.getContext('2d').clearRect(0,0,overlay.width,overlay.height);
  $('camera-placeholder').hidden=false;$('still').hidden=true;video.hidden=false;
}
async function scanLoop(id,source) {
  if(!scanning||id!==generation)return;
  try {
    const frame=frameData();
    const result=await api('/api/recognize',{image:frame.data,source});
    if(id!==generation)return;
    drawResults(result,frame);
    if(source==='live'&&activeSession()) await refresh();
  } catch(e) { if(id===generation){notice(e.message,true);stop();}return; }
  if(scanning&&id===generation)setTimeout(()=>scanLoop(id,source),650);
}
$('camera-start').onclick=()=>guarded(async()=>{
  stop();
  if(!navigator.mediaDevices?.getUserMedia)throw new Error('Camera access requires localhost or HTTPS. Open http://localhost:8000.');
  const device=$('camera-select').value;
  stream=await navigator.mediaDevices.getUserMedia({video:device?{deviceId:{exact:device},width:{ideal:960}}:{width:{ideal:960}},audio:false});
  video.srcObject=stream;await video.play();$('camera-placeholder').hidden=true;$('still').hidden=true;
  $('camera-status').textContent='Live · local camera';notice('Camera started. Recognition runs automatically; start a session to record attendance.');
  const devices=await navigator.mediaDevices.enumerateDevices();
  $('camera-select').innerHTML='<option value="">Default device camera</option>'+devices.filter(d=>d.kind==='videoinput').map((d,i)=>`<option value="${escapeHTML(d.deviceId)}">${escapeHTML(d.label||'Camera '+(i+1))}</option>`).join('');
  if(device)$('camera-select').value=device;
  scanning=true;const id=++generation;scanLoop(id,'live');
});
$('camera-stop').onclick=stop;
$('analyze-photo').onchange=()=>guarded(async()=>{
  const file=$('analyze-photo').files[0];if(!file)return;stop();
  const image=$('still');objectURL=URL.createObjectURL(file);image.src=objectURL;await image.decode();
  image.hidden=false;video.hidden=true;$('camera-placeholder').hidden=true;
  const frame=frameData(image);notice('Analyzing photo. Attendance will not change.');
  const result=await api('/api/recognize',{image:frame.data,source:'image'});drawResults(result,frame);notice('Photo analysis complete. Attendance was not recorded.');
});
$('session-start').onclick=()=>guarded(async()=>{const r=await api('/api/sessions/start',{name:$('session-name').value});await refresh();$('session-select').value=r.id;await loadReport();notice(r.message);});
$('session-end').onclick=()=>guarded(async()=>{const active=activeSession();if(active){const r=await api('/api/sessions/end',{session_id:active.id});await refresh();notice(r.message);}});
$('session-select').onchange=()=>guarded(loadReport);
$('student-list').onclick=e=>guarded(async()=>{const button=e.target.closest('[data-remove]');if(button&&confirm('Remove this student and their face templates?')){const r=await api('/api/students/remove',{student_id:button.dataset.remove});await refresh();notice(r.message);}});
$('student-list').onchange=e=>guarded(async()=>{const input=e.target.closest('[data-edit-photos]');if(!input||!input.files.length)return;const id=input.dataset.editPhotos;const images=[];for(const file of [...input.files])images.push(await fileData(file));notice('Updating photos…');const r=await api('/api/students/photos',{student_id:id,images});await refresh();notice(r.message);input.value='';});
$('capture-photo').onclick=()=>guarded(async()=>{if(!stream)throw new Error('Start the live camera first.');if(captures.length>=20)throw new Error('Maximum 20 photos.');captures.push(frameData().data);$('capture-count').textContent=captures.length+' camera photos captured';notice('Photo captured. Change pose slightly before taking the next photo.');});
async function fileData(file){const image=new Image(),url=URL.createObjectURL(file);try{image.src=url;await image.decode();return frameData(image).data;}finally{URL.revokeObjectURL(url);}}
$('enroll-form').onsubmit=e=>{e.preventDefault();guarded(async()=>{const button=e.target.querySelector('button[type=submit]');button.disabled=true;try{const files=[...$('enroll-files').files];if(files.length+captures.length>20)throw new Error('Maximum 20 photos per submission.');const images=[...captures];for(const file of files)images.push(await fileData(file));notice('Checking enrollment photos…');const r=await api('/api/students',{student_id:$('student-id').value,name:$('student-name').value,images});captures=[];$('capture-count').textContent='0 camera photos captured';e.target.reset();await refresh();notice(r.message);}finally{button.disabled=!!activeSession();}});};
window.addEventListener('beforeunload',stop);
guarded(async()=>{await refresh();if(!state.model_ready){await api('/api/model/load',{});await refresh();}});
