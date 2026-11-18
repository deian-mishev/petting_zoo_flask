const vid = document.getElementById('video');
const framePlayer = new FramePlayer(vid);
const pressedKeys = new Set();
const title = document.getElementById('title');
const envSelect = document.getElementById('envSelect');
const playerConfigDiv = document.getElementById("player-config");
const startBtn = document.getElementById('startBtn');
const screen_container = document.getElementById('screen-container');
const controls = document.getElementById('controls');
const joystickZone = document.getElementById('joystick-zone');
const actionButton = document.getElementById('action-button');

let isConnected = false;
let socket = null;
let animationFrameId = null;
let lastSentKeys = '';
let lastSendTime = 0;
let joystick = null;
let joystickDirection = null;
let actionPointerId = null;
const SEND_INTERVAL_MS = 50;
// Reuse each game's existing numpad/action mappings for laptop keyboards.
const KEY_ALIASES = new Map([
    ['q', '7'], ['w', '8'], ['e', '9'],
    ['a', '4'], ['d', '6'],
    ['z', '1'], ['x', '2'], ['c', '3'],
    [' ', 's'],
    ['arrowup', '8'], ['arrowright', '6'],
    ['arrowdown', '2'], ['arrowleft', '4'],
]);

// Prevent native dragging, selection, and long-press menus on game controls.
const gameWrapper = document.getElementById('game-wrapper');
for (const eventName of ['dragstart', 'selectstart', 'contextmenu']) {
    gameWrapper.addEventListener(eventName, (event) => event.preventDefault());
}

// Keep the configured display ratio and only scale down to fit the viewport.
const videoWidth = Number(vid.getAttribute('width'));
const videoHeight = Number(vid.getAttribute('height'));

function fitGameVideo() {
    if (!isConnected || videoWidth <= 0 || videoHeight <= 0) return;
    const controlsStyle = getComputedStyle(controls);
    const controlsHeight = controls.getBoundingClientRect().height
        + parseFloat(controlsStyle.marginTop) + parseFloat(controlsStyle.marginBottom);
    const availableHeight = Math.max(0, gameWrapper.clientHeight - controlsHeight);
    const scale = Math.min(1, gameWrapper.clientWidth / videoWidth,
        availableHeight / videoHeight);
    vid.style.width = `${videoWidth * scale}px`;
    vid.style.height = `${videoHeight * scale}px`;
}

const gameSizeObserver = new ResizeObserver(fitGameVideo);
gameSizeObserver.observe(gameWrapper);
gameSizeObserver.observe(controls);

let environments = [];
let aiPlayers = [];
let envAgentsMap = {};

function handleKeydown(e) {
    if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' ',
        'PageUp', 'PageDown', 'Home', 'End'].includes(e.key)) {
        e.preventDefault();
    }
    const sizeBefore = pressedKeys.size;
    pressedKeys.add(e.key.toLowerCase());
    if (pressedKeys.size !== sizeBefore) lastSendTime = 0;
}

function handleKeyup(e) {
    const removed = pressedKeys.delete(e.key.toLowerCase());
    if (removed) lastSendTime = 0;
}

function setJoystickDirection(direction) {
    if (joystickDirection !== direction) {
        joystickDirection = direction;
        lastSendTime = 0;
    }
}

function startJoystick() {
    if (!window.nipplejs) {
        joystickZone.textContent = 'Joystick unavailable';
        return;
    }

    joystick = nipplejs.create({
        zone: joystickZone,
        mode: 'static',
        position: { left: '50%', top: '50%' },
        color: '#c8e6a0',
        size: 130,
    });

    joystick.on('move', (event) => {
        const data = event.data;
        if (data.distance < 12) {
            setJoystickDirection(null);
            return;
        }
        const directions = ['6', '9', '8', '7', '4', '1', '2', '3'];
        const angle = Math.atan2(data.vector.y, data.vector.x);
        const sector = (Math.round(angle / (Math.PI / 4)) + 8) % 8;
        setJoystickDirection(directions[sector]);
    });
    joystick.on('end', () => setJoystickDirection(null));
}

actionButton.addEventListener('pointerdown', (event) => {
    if (!isConnected || actionPointerId !== null) return;
    event.preventDefault();
    actionPointerId = event.pointerId;
    actionButton.setPointerCapture(event.pointerId);
    lastSendTime = 0;
});

function releaseAction(event) {
    if (event.pointerId === actionPointerId) {
        actionPointerId = null;
        lastSendTime = 0;
        sendInputIfChanged();
    }
}

actionButton.addEventListener('pointerup', releaseAction);
actionButton.addEventListener('pointercancel', releaseAction);
actionButton.addEventListener('lostpointercapture', releaseAction);

function clearInputs() {
    pressedKeys.clear();
    joystickDirection = null;
    actionPointerId = null;
    lastSendTime = 0;
}

function clearInputsAndSend() {
    clearInputs();
    sendInputIfChanged();
}

window.addEventListener('blur', clearInputsAndSend);
document.addEventListener('visibilitychange', () => {
    if (document.hidden) clearInputsAndSend();
});

function checkEnableStart() {
    const selects = playerConfigDiv.querySelectorAll("select");
    let humanCount = 0;
    selects.forEach(sel => {
        if (sel.value === "human") humanCount++;
    });
    startBtn.disabled = humanCount > 1 || envSelect.value === "";
}

function sendInputIfChanged() {
    if (!isConnected || !socket?.connected) return;

    const keysArray = [...new Set([
        ...(actionPointerId !== null ? ['s'] : []),
        ...(joystickDirection ? [joystickDirection] : []),
        ...[...pressedKeys].map(key => KEY_ALIASES.get(key) ?? key),
    ])];
    const keyState = keysArray.join(',');
    if (keyState !== lastSentKeys) {
        lastSentKeys = keyState;
        socket.emit('input', keysArray);
    }
}

function gameLoop(timestamp) {
    if (!isConnected) return;
    if (timestamp - lastSendTime >= SEND_INTERVAL_MS) {
        sendInputIfChanged();
        lastSendTime = timestamp;
    }

    animationFrameId = requestAnimationFrame(gameLoop);
}

function cleanup() {
    if (!isConnected && !socket) return;
    const previousSocket = socket;
    socket = null;
    startBtn.disabled = false;
    isConnected = false;
    document.documentElement.classList.remove('playing');

    document.removeEventListener("keydown", handleKeydown);
    document.removeEventListener("keyup", handleKeyup);
    cancelAnimationFrame(animationFrameId);
    if (joystick) {
        joystick.destroy();
        joystick = null;
    }
    clearInputs();
    lastSentKeys = '';
    framePlayer.reset();

    previousSocket?.disconnect();
    title.textContent = "Episode ended, play again ..";
    vid.style.display = 'none';
    controls.style.display = 'none';
    screen_container.style.display = 'flex';

    checkEnableStart();
}

fetch('/preconnect')
    .then(response => response.json())
    .then(data => {
        environments = data.environments;
        aiPlayers = data.ai_players;

        environments.forEach(env => {
            const option = document.createElement("option");
            option.value = env.name;
            option.textContent = env.name;
            envSelect.appendChild(option);
            envAgentsMap[env.name] = env.agents;
        });
        if (environments.length > 0) {
            envSelect.value = environments[0].name;
            renderPlayerConfig(environments[0].name);
        }

        checkEnableStart();
    })
    .catch(err => {
        console.error("AI Service is down:", err);
    });

envSelect.addEventListener("change", () => renderPlayerConfig(envSelect.value));

function renderPlayerConfig(envName) {
    playerConfigDiv.innerHTML = "";
    const agents = envAgentsMap[envName] || [];

    agents.forEach(agent => {
        const label = document.createElement("label");
        label.textContent = `${agent}`;

        const select = document.createElement("select");
        select.dataset.agent = agent;

        aiPlayers.forEach(ai => {
            const opt = document.createElement("option");
            opt.value = ai;
            opt.textContent = `${ai}`;
            select.appendChild(opt);
        });

        select.addEventListener('change', checkEnableStart);

        playerConfigDiv.appendChild(label);
        playerConfigDiv.appendChild(select);
    });

    checkEnableStart();
}

startBtn.addEventListener("click", () => {
    if (isConnected) return;
    startBtn.disabled = true;
    const selects = playerConfigDiv.querySelectorAll("select");
    const environment = envSelect.value;
    if (!environment || selects.length === 0) {
        checkEnableStart();
        return;
    }

    const players = {};
    selects.forEach(sel => {
        players[sel.dataset.agent] = sel.value;
    });

    socket = io({
        query: {
            env: environment,
            players: JSON.stringify(players),
            frame_ack: '1',
        },
        transports: ['polling', 'websocket']
    });

    socket.on("connect",
        () => {
            title.textContent = "Agent Playground";
            screen_container.style.display = 'none';
            vid.style.display = 'block';
            controls.style.display = 'block';
            isConnected = true;
            const engine = socket.io.engine;
            console.info('Game transport:', engine.transport.name);
            engine.once('upgrade', transport => console.info('Game transport upgraded:', transport.name));
            window.getSelection()?.removeAllRanges();
            document.documentElement.classList.add('playing');
            window.scrollTo(0, 0);
            startJoystick();
            fitGameVideo();
            checkEnableStart();

            document.addEventListener("keydown", handleKeydown);
            document.addEventListener("keyup", handleKeyup);
            animationFrameId = requestAnimationFrame(gameLoop);
        }
    );

    socket.on("connect_error", (error) => {
        console.error("Socket connection error:", error);
        cleanup();
        title.textContent = "Game could not start. Please try again later.";
    });

    const frameSocket = socket;
    socket.on("frame", (frameBytes, acknowledge) => {
        if (socket !== frameSocket || !isConnected) {
            if (typeof acknowledge === 'function') acknowledge();
            return;
        }
        framePlayer.receive(frameBytes, acknowledge);
    });

    socket.on("episode_end", cleanup);
    socket.on("disconnect", cleanup);
});

window.addEventListener("beforeunload", cleanup);
