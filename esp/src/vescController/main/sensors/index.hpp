static const char *INDEX_HTML = R"HTML(
<!doctype html>
<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no" />
    <title>RoboCar Control</title>
    <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        html, body {
            width: 100%;
            height: 100%;
            overflow: hidden;
            background: #0d1b2e;
            touch-action: manipulation;
            -webkit-user-select: none;
            user-select: none;
        }

        body {
            font-family: system-ui, "Segoe UI", sans-serif;
            display: flex;
            flex-direction: column;
        }

        /* ── TOP BAR ── */
        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 6px 14px;
            background: rgba(0,0,0,.35);
            flex-shrink: 0;
            gap: 8px;
            flex-wrap: wrap;
        }

        .topbar-title {
            font-size: 14px;
            font-weight: 700;
            letter-spacing: .06em;
            text-transform: uppercase;
            color: #b0cfe8;
        }

        .hint { font-size: 11px; color: rgba(255,255,255,.38); }

        .topbar-right { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }

        .btn-sm {
            border: none;
            border-radius: 8px;
            padding: 6px 14px;
            font-family: inherit;
            font-size: 13px;
            font-weight: 700;
            cursor: pointer;
            color: #fff;
            touch-action: manipulation;
        }
        .btn-connect  { background: #1d8a50; }
        .btn-disc     { background: #3a4f62; }
        .btn-fs       { background: #2a4a6a; font-size: 18px; padding: 4px 12px; }

        /* ── MAIN LAYOUT ── */
        .layout {
            flex: 1;
            display: grid;
            grid-template-columns: 1fr 1px 1fr;
            min-height: 0;
        }

        .divider { background: rgba(255,255,255,.1); }

        .col {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 12px;
            gap: 20px;
        }

        .col-label {
            font-size: 10px;
            letter-spacing: .14em;
            text-transform: uppercase;
            color: rgba(255,255,255,.3);
            font-weight: 600;
        }

        .dpad-v { display: flex; flex-direction: column; align-items: center; gap: 10px; }
        .dpad-h { display: flex; flex-direction: row; align-items: center; gap: 20px; }

        .ctl {
            border: none;
            border-radius: 18px;
            color: #fff;
            background: #1475a0;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: transform 70ms, filter 70ms;
            touch-action: manipulation;
            -webkit-tap-highlight-color: transparent;
            font-size: 32px;
        }

        .ctl:active, .ctl.pressed {
            transform: scale(0.91);
            filter: brightness(.78);
        }

        .btn-fwd, .btn-bwd   { width: min(42vw, 220px); height: min(14vh, 96px); }
        .btn-left, .btn-right { width: min(14vw, 96px); height: min(28vh, 180px); }

        /* Bigger buttons in fullscreen */
        :fullscreen .btn-fwd,
        :fullscreen .btn-bwd   { width: min(42vw, 320px); height: min(17vh, 120px); }
        :fullscreen .btn-left,
        :fullscreen .btn-right { width: min(17vw, 120px); height: min(34vh, 220px); }
        :-webkit-full-screen .btn-fwd,
        :-webkit-full-screen .btn-bwd   { width: min(42vw, 320px); height: min(17vh, 120px); }
        :-webkit-full-screen .btn-left,
        :-webkit-full-screen .btn-right { width: min(17vw, 120px); height: min(34vh, 220px); }

        .stop-and-logs {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
        }

        .stop {
            border: none;
            border-radius: 50%;
            width: min(18vw, 110px);
            height: min(18vw, 110px);
            background: #b91c1c;
            color: #fff;
            font-family: inherit;
            font-size: 13px;
            font-weight: 800;
            letter-spacing: .1em;
            text-transform: uppercase;
            cursor: pointer;
            transition: transform 70ms, filter 70ms;
            touch-action: manipulation;
            line-height: 1.25;
        }
        .stop:active { transform: scale(0.91); filter: brightness(.82); }

        .log {
            width: min(38vw, 900px);
            max-height: 300px;
            overflow-y: auto;
            background: rgba(0,0,0,.45);
            border-radius: 10px;
            padding: 6px 10px;
            font-family: monospace;
            font-size: 10px;
            color: #7dd3e8;
            line-height: 1.5;
        }

        .log-toggle {
            font-size: 11px;
            color: rgba(255,255,255,.45);
            display: flex;
            align-items: center;
            gap: 5px;
            cursor: pointer;
        }
    </style>
</head>
<body>

    <div class="topbar">
        <div>
            <div class="topbar-title">RoboCar Drive Pad</div>
            <div class="hint">AP : http://192.168.4.1:3333</div>
        </div>
        <div class="topbar-right">
                        <label class="log-toggle">
                <input id="espLogs" type="checkbox" />
                Logs ESP
                </label>
            <button id="connect" class="btn-sm btn-connect">Connect</button>
            <button id="disconnect" class="btn-sm btn-disc">Disconnect</button>
            <button id="fsBtn" class="btn-sm btn-fs" title="Plein écran">⛶</button>
        </div>
    </div>




    <div class="layout">
        <div class="col">
            <div class="dpad-v">
                <button id="f" class="ctl btn-fwd">▲</button>
                <button id="b" class="ctl btn-bwd">▼</button>
            </div>
        </div>

        <div class="stop-and-logs divider" style="z-index: 1;">
                <button id="s" class="stop">STOP</button>
                <div id="log" class="log">Prêt</div>
        </div>
        <!-- <div class="divider"></div> -->

        <div class="col">
            <div class="dpad-h">
                <button id="l" class="ctl btn-left">◀</button>
                <button id="r" class="ctl btn-right">▶</button>
            </div>
        </div>

    </div>
    

    <script>
        /* ── FULLSCREEN ── */
        const fsBtn = document.getElementById('fsBtn');

        function isFs() {
            return !!(document.fullscreenElement || document.webkitFullscreenElement);
        }

        function updateFsBtn() {
            fsBtn.textContent = isFs() ? '✕' : '⛶';
            fsBtn.title = isFs() ? 'Quitter le plein écran' : 'Plein écran';
        }

        fsBtn.addEventListener('click', () => {
            if (isFs()) {
                (document.exitFullscreen || document.webkitExitFullscreen).call(document);
            } else {
                const el = document.documentElement;
                const req = el.requestFullscreen || el.webkitRequestFullscreen;
                if (req) req.call(el, { navigationUI: 'hide' }).catch(() => {});
            }
        });

        document.addEventListener('fullscreenchange', updateFsBtn);
        document.addEventListener('webkitfullscreenchange', updateFsBtn);

        /* ── ROBOCAR LOGIC (inchangée) ── */
        let connected = false;
        let logsSince = 0;
        let logsTimer = null;
        const logEl = document.getElementById('log');
        const espLogsToggle = document.getElementById('espLogs');

        function setConnectedUi(nextState) {
            connected = nextState;
        }

        function log(msg) {
            const t = new Date().toLocaleTimeString();
            logEl.innerHTML += '[' + t + '] ' + msg + '<br>';
            logEl.scrollTop = logEl.scrollHeight;
        }

        async function api(path) {
            const r = await fetch(path, { method: 'GET', cache: 'no-store' });
            if (!r.ok) throw new Error('HTTP ' + r.status);
            return r.text();
        }

        function stopLogPolling() {
            if (logsTimer !== null) { clearInterval(logsTimer); logsTimer = null; }
        }

        async function pollEspLogs() {
            if (!connected || !espLogsToggle.checked) return;
            try {
                const data = await api('/logs?since=' + encodeURIComponent(String(logsSince)));
                const lines = data.split('\n');
                for (const line of lines) {
                    if (!line) continue;
                    const sep = line.indexOf('|');
                    if (sep <= 0) continue;
                    const seq = Number(line.slice(0, sep));
                    const msg = line.slice(sep + 1);
                    if (Number.isFinite(seq) && seq > logsSince) logsSince = seq;
                    if (msg) log('ESP ' + msg);
                }
            } catch (e) {
                log('Log stream failed: ' + (e.message || e));
                stopLogPolling();
            }
        }

        function updateLogPolling() {
            stopLogPolling();
            if (connected && espLogsToggle.checked) {
                pollEspLogs();
                logsTimer = setInterval(pollEspLogs, 500);
            }
        }

        async function connect() {
            try {
                await api('/status');
                setConnectedUi(true);
                logsSince = 0;
                log('Connecté');
                updateLogPolling();
            } catch (e) {
                setConnectedUi(false);
                log('Échec connexion : ' + (e.message || e));
            }
        }

        function disconnect() {
            setConnectedUi(false);
            stopLogPolling();
            log('Déconnecté');
        }

        async function send(c) {
            if (!connected) { log('Non connecté'); return; }
            try {
                await api('/cmd?c=' + encodeURIComponent(c));
                log('→ ' + c);
            } catch (e) {
                log('Erreur : ' + (e.message || e));
            }
        }

        function bindHold(id, down, up) {
            const el = document.getElementById(id);
            let pressed = false;
            const p = (e) => { e.preventDefault(); if (pressed) return; pressed = true; el.classList.add('pressed'); send(down); };
            const r = (e) => { e.preventDefault(); if (!pressed) return; pressed = false; el.classList.remove('pressed'); send(up); };
            el.addEventListener('pointerdown', p);
            el.addEventListener('pointerup', r);
            el.addEventListener('pointercancel', r);
            el.addEventListener('touchstart', p, { passive: false });
            el.addEventListener('touchend', r, { passive: false });
            el.addEventListener('touchcancel', r, { passive: false });
            el.addEventListener('mousedown', p);
            el.addEventListener('mouseup', r);
            el.addEventListener('mouseleave', r);
        }

        document.getElementById('connect').addEventListener('click', connect);
        document.getElementById('disconnect').addEventListener('click', disconnect);
        espLogsToggle.addEventListener('change', updateLogPolling);
        document.getElementById('s').addEventListener('click', () => send('S'));
        bindHold('f', 'F', 'f');
        bindHold('l', 'L', 'l');
        bindHold('r', 'R', 'r');
        bindHold('b', 'B', 'b');
        log('Prêt');
    </script>
</body>
</html>
)HTML";