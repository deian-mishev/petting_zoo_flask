const vid = document.getElementById('video');
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

let environments = [];
let aiPlayers = [];
let envAgentsMap = {};

function handleKeydown(e) {
    const sizeBefore = pressedKeys.size;
    pressedKeys.add(e.key);
    if (pressedKeys.size !== sizeBefore) lastSendTime = 0;
}

function handleKeyup(e) {
    const removed = pressedKeys.delete(e.key);
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
        ...pressedKeys,
    ])];
    const keyState = keysArray.join(',');
    if (keyState !== lastSentKeys) {
        lastSentKeys = keyState;
        socket.emit('input', keysArray);
    }
}

function gameLoop(timestamp) {
    if (timestamp - lastSendTime >= SEND_INTERVAL_MS) {
        sendInputIfChanged();
        lastSendTime = timestamp;
    }

    animationFrameId = requestAnimationFrame(gameLoop);
}

function cleanup() {
    if (!isConnected) return;
    startBtn.disabled = false;
    isConnected = false;

    document.removeEventListener("keydown", handleKeydown);
    document.removeEventListener("keyup", handleKeyup);
    cancelAnimationFrame(animationFrameId);
    if (joystick) {
        joystick.destroy();
        joystick = null;
    }
    clearInputs();
    lastSentKeys = '';

    if (socket?.connected) {
        socket.disconnect();
    }

    socket = null;
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
    if (!environment || selects.length === 0) return;

    const players = {};
    selects.forEach(sel => {
        players[sel.dataset.agent] = sel.value;
    });

    socket = io({
        query: {
            env: environment,
            players: JSON.stringify(players)
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
            startJoystick();
            checkEnableStart();

            document.addEventListener("keydown", handleKeydown);
            document.addEventListener("keyup", handleKeyup);
            requestAnimationFrame(gameLoop);
        },
        (error) => {
            console.error("Socket connection error:", error);
            cleanup();
        }
    );

    socket.on("frame", (frameBytes) => {
        const blob = new Blob([frameBytes], { type: 'image/jpeg' });
        const url = URL.createObjectURL(blob);
        const prev = vid.src;
        vid.src = url;
        if (prev) URL.revokeObjectURL(prev);
    });

    socket.on("episode_end", cleanup);
    socket.on("disconnect", cleanup);
});

window.addEventListener("beforeunload", cleanup);
