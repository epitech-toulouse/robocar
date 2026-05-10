static const char *INDEX_HTML = R"HTML(
<!doctype html>
<html lang="fr">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no, viewport-fit=cover" />
    <title>RoboCar Control</title>
    <style>
        *, *::before, *::after { box-sizing: border-box; }

        :root {
            color-scheme: dark;
            --bg: #101418;
            --panel: #171d22;
            --panel-strong: #1f2930;
            --line: #2c3841;
            --text: #edf5f7;
            --muted: #92a3ad;
            --blue: #2383c4;
            --blue-strong: #0d6fa9;
            --green: #18a058;
            --green-soft: #153526;
            --red: #d14343;
            --red-strong: #b52525;
            --amber: #e0a11b;
            --shadow: 0 18px 50px rgba(0, 0, 0, .28);
        }

        html, body {
            width: 100%;
            min-height: 100%;
            margin: 0;
            background: var(--bg);
            color: var(--text);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            touch-action: manipulation;
            -webkit-user-select: none;
            user-select: none;
        }

        body {
            min-height: 100dvh;
            overflow: auto;
            -webkit-overflow-scrolling: touch;
        }

        button, input {
            font: inherit;
        }

        input {
            -webkit-user-select: text;
            user-select: text;
            touch-action: auto;
        }

        button {
            border: 0;
            color: inherit;
            cursor: pointer;
            -webkit-tap-highlight-color: transparent;
            touch-action: manipulation;
        }

        .app {
            min-height: 100dvh;
            display: grid;
            grid-template-rows: auto 1fr;
            background:
                linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,0) 34%),
                radial-gradient(circle at top left, rgba(35,131,196,.16), transparent 30rem),
                var(--bg);
        }

        .topbar {
            display: grid;
            grid-template-columns: 1fr auto;
            align-items: center;
            gap: .75rem;
            padding: max(.65rem, env(safe-area-inset-top)) max(.8rem, env(safe-area-inset-right)) .65rem max(.8rem, env(safe-area-inset-left));
            border-bottom: 1px solid rgba(255,255,255,.08);
            background: rgba(16,20,24,.86);
            backdrop-filter: blur(18px);
        }

        .brand {
            min-width: 0;
            display: grid;
            gap: .15rem;
        }

        .brand-title {
            font-size: clamp(1rem, 2.2vw, 1.2rem);
            font-weight: 800;
            letter-spacing: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .brand-subtitle {
            color: var(--muted);
            font-size: .78rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .status-row {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: .45rem;
            flex-wrap: wrap;
        }

        .chip {
            min-height: 2rem;
            display: inline-flex;
            align-items: center;
            gap: .4rem;
            padding: .35rem .6rem;
            border: 1px solid var(--line);
            border-radius: 999px;
            background: rgba(255,255,255,.04);
            color: var(--muted);
            font-size: .78rem;
            font-weight: 700;
            line-height: 1;
        }

        .dot {
            width: .55rem;
            height: .55rem;
            border-radius: 999px;
            background: #687782;
            box-shadow: 0 0 0 .18rem rgba(104,119,130,.14);
        }

        .chip.ok {
            color: #d9f8e8;
            border-color: rgba(24,160,88,.55);
            background: var(--green-soft);
        }

        .chip.ok .dot {
            background: var(--green);
            box-shadow: 0 0 0 .18rem rgba(24,160,88,.18);
        }

        .chip.warn {
            color: #ffecc0;
            border-color: rgba(224,161,27,.55);
            background: rgba(224,161,27,.13);
        }

        .chip.warn .dot {
            background: var(--amber);
            box-shadow: 0 0 0 .18rem rgba(224,161,27,.18);
        }

        .btn {
            min-height: 2.25rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: .45rem;
            padding: .5rem .75rem;
            border-radius: .5rem;
            background: var(--panel-strong);
            border: 1px solid var(--line);
            font-weight: 800;
            font-size: .82rem;
            line-height: 1;
            transition: transform 80ms ease, filter 80ms ease, background 80ms ease, border-color 80ms ease;
        }

        .btn:active,
        .btn.pressed {
            transform: translateY(1px) scale(.98);
            filter: brightness(.9);
        }

        .btn-primary {
            background: var(--green);
            border-color: rgba(255,255,255,.08);
            color: #04140b;
        }

        .btn-danger {
            background: rgba(209,67,67,.12);
            border-color: rgba(209,67,67,.5);
            color: #ffdada;
        }

        .btn-icon {
            width: 2.25rem;
            padding: 0;
            font-size: 1.15rem;
        }

        .workspace {
            min-height: 0;
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(10rem, .55fr) minmax(0, 1fr);
            gap: clamp(.65rem, 1.5vw, 1.1rem);
            padding: clamp(.7rem, 1.6vw, 1.1rem);
        }

        .drive-zone,
        .center-zone {
            min-width: 0;
            min-height: 0;
            border: 1px solid rgba(255,255,255,.08);
            background: rgba(23,29,34,.84);
            box-shadow: var(--shadow);
        }

        .drive-zone {
            display: grid;
            align-content: center;
            justify-items: center;
            gap: clamp(.8rem, 2.2vh, 1.2rem);
            border-radius: .75rem;
            padding: clamp(.75rem, 2vw, 1.2rem);
        }

        .zone-label {
            align-self: end;
            color: var(--muted);
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .12em;
            text-transform: uppercase;
        }

        .pad-vertical {
            display: grid;
            gap: clamp(.75rem, 2vh, 1.1rem);
            width: min(100%, 22rem);
        }

        .pad-horizontal {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: clamp(.75rem, 2vw, 1.1rem);
            width: min(100%, 28rem);
        }

        .ctl {
            width: 100%;
            min-height: 7.5rem;
            border-radius: .85rem;
            background: linear-gradient(180deg, #2a93d1, var(--blue-strong));
            box-shadow: inset 0 1px 0 rgba(255,255,255,.2), 0 12px 28px rgba(0,0,0,.26);
            font-size: clamp(2.2rem, 8vw, 4.5rem);
            font-weight: 900;
            transition: transform 70ms ease, filter 70ms ease, box-shadow 70ms ease;
        }

        .ctl:active,
        .ctl.pressed {
            transform: scale(.96);
            filter: brightness(.86);
            box-shadow: inset 0 2px 8px rgba(0,0,0,.25), 0 7px 18px rgba(0,0,0,.22);
        }

        .ctl:disabled {
            cursor: not-allowed;
            filter: saturate(.35) brightness(.68);
            box-shadow: inset 0 1px 0 rgba(255,255,255,.08), 0 7px 18px rgba(0,0,0,.18);
        }

        .btn-left,
        .btn-right {
            min-height: 16rem;
        }

        .center-zone {
            display: grid;
            grid-template-rows: auto auto auto;
            gap: .75rem;
            border-radius: .75rem;
            padding: .8rem;
            overflow: auto;
            -webkit-overflow-scrolling: touch;
        }

        .stop {
            width: min(44vw, 10rem);
            aspect-ratio: 1;
            justify-self: center;
            border-radius: 999px;
            background: linear-gradient(180deg, var(--red), var(--red-strong));
            color: #fff;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.22), 0 16px 34px rgba(181,37,37,.25);
            font-size: clamp(1rem, 2.4vw, 1.35rem);
            font-weight: 950;
            letter-spacing: .06em;
            text-transform: uppercase;
            transition: transform 70ms ease, filter 70ms ease;
        }

        .stop:active {
            transform: scale(.96);
            filter: brightness(.88);
        }

        .actions {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: .5rem;
        }

        .actions .btn-primary {
            grid-column: 1 / -1;
        }

        .algo-panel {
            display: grid;
            gap: .55rem;
            padding: .7rem;
            border: 1px solid rgba(255,255,255,.08);
            border-radius: .65rem;
            background: rgba(255,255,255,.03);
        }

        .algo-copy {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: .5rem;
        }

        .algo-title {
            font-size: .82rem;
            font-weight: 800;
        }

        .algo-count {
            color: var(--muted);
            font-size: .72rem;
            font-weight: 900;
            letter-spacing: .1em;
            text-transform: uppercase;
        }

        .algo-hint {
            color: var(--muted);
            font-size: .74rem;
            line-height: 1.35;
        }

        .gps-panel {
            display: grid;
            gap: .55rem;
            padding: .7rem;
            border: 1px solid rgba(255,255,255,.08);
            border-radius: .65rem;
            background: rgba(255,255,255,.03);
        }

        .gps-copy {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: .5rem;
            flex-wrap: wrap;
        }

        .gps-title {
            font-size: .82rem;
            font-weight: 800;
        }

        .gps-current {
            color: var(--muted);
            font-size: .72rem;
            font-weight: 700;
        }

        .gps-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: .5rem;
        }

        .gps-field {
            display: grid;
            gap: .28rem;
            min-width: 0;
        }

        .gps-field span {
            color: var(--muted);
            font-size: .72rem;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: .08em;
        }

        .gps-field input {
            width: 100%;
            min-width: 0;
            min-height: 2.35rem;
            padding: .55rem .65rem;
            border: 1px solid rgba(255,255,255,.1);
            border-radius: .55rem;
            background: rgba(0,0,0,.2);
            color: var(--text);
        }

        .gps-field input:disabled {
            opacity: .65;
            cursor: not-allowed;
        }

        .gps-help {
            color: var(--muted);
            font-size: .74rem;
            line-height: 1.35;
        }

        .algo-list {
            display: grid;
            gap: .45rem;
        }

        .algo-row {
            min-height: 2.45rem;
            display: grid;
            grid-template-columns: auto minmax(0, 1fr) auto;
            align-items: center;
            gap: .65rem;
            padding: .55rem .6rem;
            border: 1px solid rgba(255,255,255,.08);
            border-radius: .55rem;
            background: rgba(0,0,0,.16);
            cursor: pointer;
        }

        .algo-row.is-disabled {
            opacity: .58;
            cursor: not-allowed;
        }

        .algo-row input {
            width: 1rem;
            height: 1rem;
            margin: 0;
            accent-color: var(--blue);
        }

        .algo-main {
            min-width: 0;
            display: grid;
            gap: .1rem;
        }

        .algo-name {
            font-size: .8rem;
            font-weight: 800;
        }

        .algo-meta {
            color: var(--muted);
            font-size: .71rem;
            line-height: 1.25;
        }

        .algo-status {
            min-width: 4.75rem;
            padding: .25rem .45rem;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,.08);
            background: rgba(255,255,255,.05);
            color: var(--muted);
            font-size: .68rem;
            font-weight: 800;
            text-align: center;
            text-transform: uppercase;
        }

        .algo-status.is-ready {
            color: #d9f8e8;
            border-color: rgba(24,160,88,.55);
            background: var(--green-soft);
        }

        .algo-status.is-waiting {
            color: #ffecc0;
            border-color: rgba(224,161,27,.55);
            background: rgba(224,161,27,.13);
        }

        .algo-status.is-disabled {
            color: #ffdada;
            border-color: rgba(209,67,67,.42);
            background: rgba(209,67,67,.12);
        }

        .log-panel {
            min-height: 0;
            display: grid;
            grid-template-rows: auto;
            gap: .5rem;
        }

        .log-tools {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: .5rem;
            color: var(--muted);
            font-size: .78rem;
            font-weight: 750;
        }

        .toggle {
            display: inline-flex;
            align-items: center;
            gap: .45rem;
            cursor: pointer;
        }

        .toggle input {
            width: 1rem;
            height: 1rem;
            accent-color: var(--blue);
        }

        .log {
            display: none;
            height: clamp(7rem, 24vh, 16rem);
            min-height: 5rem;
            max-height: 16rem;
            overflow-y: auto;
            border: 1px solid rgba(255,255,255,.08);
            border-radius: .55rem;
            background: rgba(0,0,0,.28);
            padding: .55rem .65rem;
            color: #a9e7f2;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
            font-size: .72rem;
            line-height: 1.45;
            overflow-wrap: anywhere;
        }

        .is-logs-open .log {
            display: block;
        }

        .is-logs-open .log-panel {
            grid-template-rows: auto minmax(0, auto);
        }

        .is-disconnected .ctl,
        .is-disconnected .stop,
        .is-disconnected #arm {
            filter: saturate(.45) brightness(.78);
        }

        .is-armed #arm {
            background: var(--green-soft);
            border-color: rgba(24,160,88,.6);
            color: #d9f8e8;
        }

        :fullscreen .workspace,
        :-webkit-full-screen .workspace {
            padding: clamp(.5rem, 1.2vw, 1rem);
        }

        :fullscreen .ctl,
        :-webkit-full-screen .ctl {
            min-height: 8.5rem;
        }

        :fullscreen .log,
        :-webkit-full-screen .log {
            height: clamp(6rem, 22vh, 14rem);
        }

        @media (orientation: landscape) and (max-height: 520px) {
            .topbar {
                grid-template-columns: minmax(8rem, 1fr) auto;
                padding-top: .45rem;
                padding-bottom: .45rem;
            }

            .brand-subtitle,
            .zone-label {
                display: none;
            }

            .workspace {
                grid-template-columns: 1fr minmax(8rem, .42fr) 1fr;
                gap: .55rem;
                padding: .55rem;
            }

            .drive-zone,
            .center-zone {
                padding: .55rem;
                border-radius: .6rem;
            }

            .ctl {
                min-height: 5.2rem;
                font-size: clamp(2rem, 8vh, 3.3rem);
            }

            .btn-left,
            .btn-right {
                min-height: 11rem;
            }

            .stop {
                width: min(9.5rem, 30vh);
            }

            .log-panel {
                display: none;
            }

            .center-zone {
                grid-template-rows: 1fr auto;
                align-items: center;
            }
        }

        @media (max-width: 760px) and (orientation: portrait) {
            body {
                overflow: auto;
            }

            .app {
                min-height: 100dvh;
            }

            .topbar {
                grid-template-columns: 1fr;
                align-items: start;
            }

            .status-row {
                justify-content: flex-start;
            }

            .workspace {
                grid-template-columns: 1fr;
                grid-template-rows: auto auto auto;
                padding-bottom: max(.8rem, env(safe-area-inset-bottom));
            }

            .drive-zone,
            .center-zone {
                min-height: auto;
            }

            .pad-vertical,
            .pad-horizontal {
                width: 100%;
            }

            .ctl {
                min-height: 5.6rem;
            }

            .btn-left,
            .btn-right {
                min-height: 6.6rem;
            }

            .center-zone {
                order: -1;
            }

            .stop {
                width: min(46vw, 8rem);
            }
        }

        @media (max-width: 430px) {
            .status-row {
                gap: .35rem;
            }

            .chip,
            .btn {
                font-size: .74rem;
                padding-left: .5rem;
                padding-right: .5rem;
            }

            .btn-icon {
                width: 2.15rem;
            }

            .actions {
                grid-template-columns: 1fr;
            }

            .actions .btn-primary {
                grid-column: auto;
            }

            .gps-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body class="is-disconnected">
    <main class="app">
        <header class="topbar">
            <div class="brand">
                <div class="brand-title">RoboCar Control</div>
                <div class="brand-subtitle">AP ROBOCAR_CTRL · http://192.168.4.1:3333</div>
            </div>
            <div class="status-row">
                <span id="connChip" class="chip"><span class="dot"></span><span id="connText">Hors ligne</span></span>
                <span id="vescChip" class="chip warn"><span class="dot"></span><span id="vescText">VESC off</span></span>
                <button id="connect" class="btn btn-primary">Connecter</button>
                <button id="disconnect" class="btn">Couper</button>
                <button id="fsBtn" class="btn btn-icon" title="Plein écran">⛶</button>
            </div>
        </header>

        <section class="workspace" aria-label="Commandes RoboCar">
            <section class="drive-zone" aria-label="Vitesse">
                <div class="zone-label">Vitesse</div>
                <div class="pad-vertical">
                    <button id="f" class="ctl btn-fwd" aria-label="Avancer">▲</button>
                    <button id="b" class="ctl btn-bwd" aria-label="Reculer">▼</button>
                </div>
            </section>

            <section class="center-zone" aria-label="Securite et logs">
                <button id="s" class="stop">Stop</button>
                <div class="actions">
                    <button id="arm" class="btn btn-primary">Activer VESC</button>
                    <button id="clearLog" class="btn">Nettoyer</button>
                    <button id="refreshStatus" class="btn">Statut</button>
                </div>
                <section class="algo-panel" aria-label="Selection des algorithmes">
                    <div class="algo-copy">
                        <span class="algo-title">Algorithmes</span>
                        <span id="algoCount" class="algo-count">3 actifs</span>
                    </div>
                    <div class="algo-list">
                        <label class="algo-row" for="algo-manual">
                            <input id="algo-manual" type="checkbox" data-algo="manual" checked />
                            <span class="algo-main">
                                <span class="algo-name">Manual</span>
                                <span id="algoMeta-manual" class="algo-meta">Poids 100</span>
                            </span>
                            <span id="algoStatus-manual" class="algo-status is-ready">pret</span>
                        </label>
                        <label class="algo-row" for="algo-close_obstacle">
                            <input id="algo-close_obstacle" type="checkbox" data-algo="close_obstacle" checked />
                            <span class="algo-main">
                                <span class="algo-name">Close obstacle</span>
                                <span id="algoMeta-close_obstacle" class="algo-meta">Poids 100</span>
                            </span>
                            <span id="algoStatus-close_obstacle" class="algo-status is-ready">pret</span>
                        </label>
                        <label class="algo-row" for="algo-lidar_corridor">
                            <input id="algo-lidar_corridor" type="checkbox" data-algo="lidar_corridor" checked />
                            <span class="algo-main">
                                <span class="algo-name">Corridor LiDAR</span>
                                <span id="algoMeta-lidar_corridor" class="algo-meta">Poids 5</span>
                            </span>
                            <span id="algoStatus-lidar_corridor" class="algo-status is-ready">pret</span>
                        </label>
                        <label class="algo-row" for="algo-gps">
                            <input id="algo-gps" type="checkbox" data-algo="gps" />
                            <span class="algo-main">
                                <span class="algo-name">GPS</span>
                                <span id="algoMeta-gps" class="algo-meta">Poids 1</span>
                            </span>
                            <span id="algoStatus-gps" class="algo-status is-waiting">veille</span>
                        </label>
                        <label class="algo-row" for="algo-camera">
                            <input id="algo-camera" type="checkbox" data-algo="camera" />
                            <span class="algo-main">
                                <span class="algo-name">Camera</span>
                                <span id="algoMeta-camera" class="algo-meta">Poids 5</span>
                            </span>
                            <span id="algoStatus-camera" class="algo-status">disponible</span>
                        </label>
                    </div>
                    <div id="algoHint" class="algo-hint">Actifs : Manual + Close obstacle + Corridor LiDAR</div>
                </section>
                <section class="gps-panel" aria-label="Destination GPS">
                    <div class="gps-copy">
                        <span class="gps-title">Destination GPS</span>
                        <span id="gpsGoalCurrent" class="gps-current">Actuelle : inconnue</span>
                    </div>
                    <div class="gps-grid">
                        <label class="gps-field" for="gpsGoalLat">
                            <span>Latitude</span>
                            <input id="gpsGoalLat" type="text" inputmode="decimal" placeholder="43.612139" autocomplete="off" spellcheck="false" />
                        </label>
                        <label class="gps-field" for="gpsGoalLon">
                            <span>Longitude</span>
                            <input id="gpsGoalLon" type="text" inputmode="decimal" placeholder="1.430194" autocomplete="off" spellcheck="false" />
                        </label>
                    </div>
                    <div class="gps-help">Format attendu : degres decimaux signes avec un point (.). Exemples : 43.612139 et 1.430194</div>
                    <button id="gpsGoalApply" class="btn">Appliquer destination GPS</button>
                </section>
                <div class="log-panel">
                    <div class="log-tools">
                        <span>Journal</span>
                        <label class="toggle">
                            <input id="espLogs" type="checkbox" />
                            Afficher logs
                        </label>
                    </div>
                    <div id="log" class="log">Pret</div>
                </div>
            </section>

            <section class="drive-zone" aria-label="Direction">
                <div class="zone-label">Direction</div>
                <div class="pad-horizontal">
                    <button id="l" class="ctl btn-left" aria-label="Gauche">◀</button>
                    <button id="r" class="ctl btn-right" aria-label="Droite">▶</button>
                </div>
            </section>
        </section>
    </main>

    <script>
        const body = document.body;
        const fsBtn = document.getElementById('fsBtn');
        const logEl = document.getElementById('log');
        const espLogsToggle = document.getElementById('espLogs');
        const connChip = document.getElementById('connChip');
        const connText = document.getElementById('connText');
        const vescChip = document.getElementById('vescChip');
        const vescText = document.getElementById('vescText');
        const armBtn = document.getElementById('arm');
        const algoCount = document.getElementById('algoCount');
        const algoHint = document.getElementById('algoHint');
        const gpsGoalCurrent = document.getElementById('gpsGoalCurrent');
        const gpsGoalLatInput = document.getElementById('gpsGoalLat');
        const gpsGoalLonInput = document.getElementById('gpsGoalLon');
        const gpsGoalApplyBtn = document.getElementById('gpsGoalApply');
        const algorithmCheckboxes = Array.from(document.querySelectorAll('[data-algo]'));
        const driveButtons = ['f', 'b', 'l', 'r'].map((id) => document.getElementById(id));
        const algorithmLabels = {
            manual: 'Manual',
            close_obstacle: 'Close obstacle',
            lidar_corridor: 'Corridor LiDAR',
            gps: 'GPS',
            camera: 'Camera',
        };

        let connected = false;
        let vescActive = false;
        let logsSince = 0;
        let logsTimer = null;
        let algorithmEntries = {};
        let gpsGoal = { lat: null, lon: null, enabled: true };
        let selectedAlgorithms = new Set(['camera']);
        const maxVisibleLogLines = 200;
        const visibleLogLines = [];

        function isFs() {
            return !!(document.fullscreenElement || document.webkitFullscreenElement);
        }

        function updateFsBtn() {
            fsBtn.textContent = isFs() ? '×' : '⛶';
            fsBtn.title = isFs() ? 'Quitter le plein écran' : 'Plein écran';
        }

        fsBtn.addEventListener('click', () => {
            if (isFs()) {
                const exit = document.exitFullscreen || document.webkitExitFullscreen;
                if (exit) exit.call(document);
            } else {
                const req = document.documentElement.requestFullscreen || document.documentElement.webkitRequestFullscreen;
                if (req) req.call(document.documentElement, { navigationUI: 'hide' }).catch(() => {});
            }
        });

        document.addEventListener('fullscreenchange', updateFsBtn);
        document.addEventListener('webkitfullscreenchange', updateFsBtn);

        function formatWeight(value) {
            const n = Number(value);
            if (!Number.isFinite(n)) return '?';
            if (Number.isInteger(n)) return String(n);
            return n.toFixed(2);
        }

        function applyAlgorithmPayload(data) {
            if (Array.isArray(data.selectedAlgorithms)) {
                selectedAlgorithms = new Set(data.selectedAlgorithms);
            }
            if (Array.isArray(data.algorithms)) {
                const nextEntries = {};
                for (const entry of data.algorithms) {
                    nextEntries[entry.id] = entry;
                }
                algorithmEntries = nextEntries;
            }
        }

        function applyGpsGoalPayload(data) {
            if (!data || !data.gpsGoal) return;
            const lat = Number(data.gpsGoal.lat);
            const lon = Number(data.gpsGoal.lon);
            if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
            gpsGoal = {
                lat,
                lon,
                enabled: data.gpsGoal.enabled !== false,
            };
        }

        function formatCoord(value) {
            const n = Number(value);
            if (!Number.isFinite(n)) return '?';
            return n.toFixed(6);
        }

        function syncGpsGoalInputs() {
            if (document.activeElement !== gpsGoalLatInput) {
                gpsGoalLatInput.value = Number.isFinite(gpsGoal.lat) ? String(gpsGoal.lat) : '';
            }
            if (document.activeElement !== gpsGoalLonInput) {
                gpsGoalLonInput.value = Number.isFinite(gpsGoal.lon) ? String(gpsGoal.lon) : '';
            }
        }

        function selectedAlgorithmLabels() {
            const labels = [];
            for (const id of selectedAlgorithms) {
                labels.push(algorithmLabels[id] || id);
            }
            return labels;
        }

        function renderState() {
            body.classList.toggle('is-disconnected', !connected);
            body.classList.toggle('is-armed', vescActive);
            body.classList.toggle('is-logs-open', espLogsToggle.checked);

            connChip.classList.toggle('ok', connected);
            connText.textContent = connected ? 'Connecté' : 'Hors ligne';

            vescChip.classList.toggle('ok', vescActive);
            vescChip.classList.toggle('warn', !vescActive);
            vescText.textContent = vescActive ? 'VESC actif' : 'VESC off';
            armBtn.textContent = vescActive ? 'VESC actif' : 'Activer VESC';

            const activeLabels = selectedAlgorithmLabels();
            algoCount.textContent = activeLabels.length === 0
                ? '0 actifs'
                : activeLabels.length + ' actifs';
            algoHint.textContent = activeLabels.length === 0
                ? 'Aucun algo actif : sortie nulle, la voiture s arrete.'
                : 'Actifs : ' + activeLabels.join(' + ');

            gpsGoalCurrent.textContent = (Number.isFinite(gpsGoal.lat) && Number.isFinite(gpsGoal.lon))
                ? ('Actuelle : ' + formatCoord(gpsGoal.lat) + ', ' + formatCoord(gpsGoal.lon))
                : 'Actuelle : inconnue';
            gpsGoalLatInput.disabled = !connected;
            gpsGoalLonInput.disabled = !connected;
            gpsGoalApplyBtn.disabled = !connected;

            for (const checkbox of algorithmCheckboxes) {
                const id = checkbox.dataset.algo;
                const entry = algorithmEntries[id] || {
                    id,
                    enabled: selectedAlgorithms.has(id),
                    available: false,
                    implemented: true,
                    weight: 0,
                };
                const row = checkbox.closest('.algo-row');
                const metaEl = document.getElementById('algoMeta-' + id);
                const statusEl = document.getElementById('algoStatus-' + id);
                const enabled = selectedAlgorithms.has(id);
                const implemented = !!entry.implemented;
                const available = !!entry.available;

                checkbox.checked = enabled;
                checkbox.disabled = !connected || !implemented;
                row.classList.toggle('is-disabled', !implemented);

                metaEl.textContent = 'Poids ' + formatWeight(entry.weight)
                    + (implemented ? (available ? ' · pret' : ' · veille') : ' · non implemente');

                statusEl.className = 'algo-status';
                if (!implemented) {
                    statusEl.classList.add('is-disabled');
                    statusEl.textContent = 'bientot';
                } else if (available) {
                    statusEl.classList.add('is-ready');
                    statusEl.textContent = 'pret';
                } else {
                    statusEl.classList.add('is-waiting');
                    statusEl.textContent = 'veille';
                }
            }

            const manualDriveEnabled = connected && selectedAlgorithms.has('manual');
            for (const button of driveButtons) {
                button.disabled = !manualDriveEnabled;
            }
        }

        function log(msg) {
            const t = new Date().toLocaleTimeString();
            visibleLogLines.push('[' + t + '] ' + msg);
            while (visibleLogLines.length > maxVisibleLogLines) {
                visibleLogLines.shift();
            }
            logEl.innerHTML = visibleLogLines.join('<br>') + '<br>';
            logEl.scrollTop = logEl.scrollHeight;
        }

        async function apiText(path) {
            const r = await fetch(path, { method: 'GET', cache: 'no-store' });
            if (!r.ok) throw new Error('HTTP ' + r.status);
            return r.text();
        }

        async function apiJson(path) {
            const r = await fetch(path, { method: 'GET', cache: 'no-store' });
            const text = await r.text();
            let data = null;
            try {
                data = text ? JSON.parse(text) : null;
            } catch (_) {
                data = null;
            }
            if (!r.ok) {
                throw new Error((data && data.error) ? data.error : ('HTTP ' + r.status));
            }
            return data;
        }

        async function readStatus() {
            const data = await apiJson('/status');
            connected = true;
            vescActive = !!data.vescActive;
            applyAlgorithmPayload(data);
            applyGpsGoalPayload(data);
            syncGpsGoalInputs();
            renderState();
            return data;
        }

        function stopLogPolling() {
            if (logsTimer !== null) {
                clearInterval(logsTimer);
                logsTimer = null;
            }
        }

        async function pollEspLogs() {
            if (!connected || !espLogsToggle.checked) return;
            try {
                const data = await apiText('/logs?since=' + encodeURIComponent(String(logsSince)));
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
                log('Logs stoppés : ' + (e.message || e));
                stopLogPolling();
            }
        }

        function updateLogPolling() {
            renderState();
            stopLogPolling();
            if (connected && espLogsToggle.checked) {
                pollEspLogs();
                logsTimer = setInterval(pollEspLogs, 500);
            }
        }

        async function connect() {
            try {
                await readStatus();
                logsSince = 0;
                log('Connecté');
                updateLogPolling();
            } catch (e) {
                connected = false;
                vescActive = false;
                renderState();
                log('Échec connexion : ' + (e.message || e));
            }
        }

        function disconnect() {
            connected = false;
            stopLogPolling();
            renderState();
            log('Déconnecté');
        }

        function getSelectedAlgorithmsFromUi() {
            return algorithmCheckboxes
                .filter((checkbox) => checkbox.checked && !checkbox.disabled)
                .map((checkbox) => checkbox.dataset.algo);
        }

        async function updateAlgorithms(nextSelected) {
            if (!connected) {
                log('Non connecté');
                renderState();
                return false;
            }
            try {
                const data = await apiJson('/algorithms?selected=' + encodeURIComponent(nextSelected.join(',')));
                applyAlgorithmPayload(data);
                renderState();
                log('Algorithmes : ' + (nextSelected.length ? nextSelected.join(', ') : 'aucun'));
                return true;
            } catch (e) {
                await readStatus().catch(() => {});
                renderState();
                log('Erreur algorithmes : ' + (e.message || e));
                return false;
            }
        }

        async function send(c, label) {
            if (!connected) {
                log('Non connecté');
                return false;
            }
            try {
                await apiJson('/cmd?c=' + encodeURIComponent(c));
                if (c === 'A') vescActive = true;
                if (c === 'S') vescActive = false;
                renderState();
                log(label || ('→ ' + c));
                return true;
            } catch (e) {
                log('Erreur : ' + (e.message || e));
                return false;
            }
        }

        async function updateGpsGoal() {
            if (!connected) {
                log('Non connecté');
                return false;
            }

            const lat = gpsGoalLatInput.value.trim();
            const lon = gpsGoalLonInput.value.trim();
            if (!lat || !lon) {
                log('Coordonnées GPS manquantes');
                return false;
            }

            try {
                const data = await apiJson('/gps-goal?lat=' + encodeURIComponent(lat) + '&lon=' + encodeURIComponent(lon));
                applyGpsGoalPayload(data);
                syncGpsGoalInputs();
                renderState();
                log('Destination GPS : ' + formatCoord(gpsGoal.lat) + ', ' + formatCoord(gpsGoal.lon));
                return true;
            } catch (e) {
                log('Erreur destination GPS : ' + (e.message || e));
                return false;
            }
        }

        function bindHold(id, down, up) {
            const el = document.getElementById(id);
            let pressed = false;
            const press = (e) => {
                e.preventDefault();
                if (pressed) return;
                pressed = true;
                el.classList.add('pressed');
                send(down);
            };
            const release = (e) => {
                e.preventDefault();
                if (!pressed) return;
                pressed = false;
                el.classList.remove('pressed');
                send(up);
            };
            el.addEventListener('pointerdown', press);
            el.addEventListener('pointerup', release);
            el.addEventListener('pointercancel', release);
            el.addEventListener('pointerleave', release);
            el.addEventListener('touchstart', press, { passive: false });
            el.addEventListener('touchend', release, { passive: false });
            el.addEventListener('touchcancel', release, { passive: false });
            el.addEventListener('mousedown', press);
            el.addEventListener('mouseup', release);
            el.addEventListener('mouseleave', release);
        }

        document.getElementById('connect').addEventListener('click', connect);
        document.getElementById('disconnect').addEventListener('click', disconnect);
        document.getElementById('clearLog').addEventListener('click', () => {
            visibleLogLines.length = 0;
            logEl.innerHTML = '';
            log('Pret');
        });
        document.getElementById('refreshStatus').addEventListener('click', () => readStatus().then(() => log('Statut actualisé')).catch((e) => log('Statut indisponible : ' + (e.message || e))));
        armBtn.addEventListener('click', () => send('A', 'VESC activé'));
        gpsGoalApplyBtn.addEventListener('click', updateGpsGoal);
        algorithmCheckboxes.forEach((checkbox) => {
            checkbox.addEventListener('change', () => {
                const nextSelected = getSelectedAlgorithmsFromUi();
                selectedAlgorithms = new Set(nextSelected);
                renderState();
                updateAlgorithms(nextSelected);
            });
        });
        espLogsToggle.addEventListener('change', updateLogPolling);
        document.getElementById('s').addEventListener('click', () => send('S', 'STOP envoyé'));
        bindHold('f', 'F', 'f');
        bindHold('l', 'L', 'l');
        bindHold('r', 'R', 'r');
        bindHold('b', 'B', 'b');
        renderState();
    </script>
</body>
</html>
)HTML";
