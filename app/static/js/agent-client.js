const vid = document.getElementById('video');
const pressedKeys = new Set();
const title = document.getElementById('title');
const envSelect = document.getElementById('envSelect');
const playerConfigDiv = document.getElementById("player-config");
const startBtn = document.getElementById('startBtn');
const screen_container = document.getElementById('screen-container');

let isConnected = false;
let socket = null;
let animationFrameId = null;
let lastSentKeys = '';
let lastSendTime = 0;
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

function checkEnableStart() {
    const selects = playerConfigDiv.querySelectorAll("select");
    let humanCount = 0;
    selects.forEach(sel => {
        if (sel.value === "human") humanCount++;
    });
    startBtn.disabled = humanCount > 1 || envSelect.value === "";
}

function gameLoop(timestamp) {
    if (timestamp - lastSendTime >= SEND_INTERVAL_MS) {
        if (pressedKeys.size > 0) {
            const keysArray = Array.from(pressedKeys);
            keysArray.sort((a, b) => {
                if (a === 's') return -1;
                if (b === 's') return 1;
                return 0;
            });
            const keyState = keysArray.join(',');

            if (keyState !== lastSentKeys) {
                lastSentKeys = keyState;
                socket.emit('input', keysArray);
            }
        } else if (lastSentKeys !== '') {
            lastSentKeys = '';
            socket.emit('input', []);
        }
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

    if (socket?.connected) {
        socket.disconnect();
    }

    socket = null;
    title.textContent = "Episode ended, play again ..";
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
        transports: ['websocket']
    });

    socket.on("connect",
        () => {
            title.textContent = "Agent Playground";
            screen_container.style.display = 'none';
            isConnected = true;
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

    socket.on("frame", (base64Frame) => {
        vid.src = 'data:image/png;base64,' + base64Frame;
    });

    socket.on("episode_end", cleanup);
    socket.on("disconnect", cleanup);
});

window.addEventListener("beforeunload", cleanup);
