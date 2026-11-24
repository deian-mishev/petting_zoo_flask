// Run with: node --test tests/test_streaming_client.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

function browser() {
    const decoded = [];
    const revoked = [];
    const emitted = [];
    const elements = new Map();
    const windowHandlers = {};
    const documentHandlers = {};
    let nextUrl = 0;
    function element(id) {
        if (!elements.has(id)) elements.set(id, {
            src: '', style: {}, handlers: {},
            addEventListener(name, fn) { this.handlers[name] = fn; },
            getAttribute() { return '480'; },
            removeAttribute(name) { this[name] = ''; },
            setPointerCapture() {},
            querySelectorAll() { return []; },
        });
        return elements.get(id);
    }
    const context = vm.createContext({
        Blob,
        URL: {
            createObjectURL() { return `blob:${++nextUrl}`; },
            revokeObjectURL(url) { revoked.push(url); },
        },
        Image: class {
            decode() {
                return new Promise((resolve, reject) => decoded.push({ resolve, reject, image: this }));
            }
        },
        document: {
            hidden: false,
            getElementById: element,
            addEventListener(name, fn) { documentHandlers[name] = fn; },
            removeEventListener() {},
            documentElement: { classList: { remove() {} } },
        },
        window: { addEventListener(name, fn) { windowHandlers[name] = fn; } },
        ResizeObserver: class { observe() {} },
        fetch: () => new Promise(() => {}),
        setInterval() { return 1; },
        requestAnimationFrame() { return 1; }, cancelAnimationFrame() {},
        capture: (event, keys) => emitted.push({ event, keys: Array.from(keys) }),
    });
    for (const name of ['frame-player.js', 'agent-client.js']) {
        vm.runInContext(fs.readFileSync(path.join(__dirname, '../app/static/js', name), 'utf8'), context);
    }
    const run = source => vm.runInContext(source, context);
    run('isConnected = true; socket = { connected: true, emit: capture, disconnect() {} };');
    return { context, run, decoded, revoked, emitted, elements, windowHandlers, documentHandlers };
}

test('decoded replacements retain the last frame; late older frames cannot overwrite newer ones', async () => {
    const b = browser();
    const first = b.run('framePlayer.receive([1])');
    assert.equal(b.elements.get('video').src, '');
    b.decoded[0].resolve();
    await first;
    assert.equal(b.elements.get('video').src, 'blob:1');
    const second = b.run('framePlayer.receive([2])');
    const third = b.run('framePlayer.receive([3])');
    assert.equal(b.elements.get('video').src, 'blob:1');
    assert.deepEqual(b.revoked, []);
    b.decoded[2].resolve();
    await third;
    b.decoded[1].resolve();
    await second;
    assert.equal(b.elements.get('video').src, 'blob:3');
    assert.deepEqual(b.revoked, ['blob:1', 'blob:2']);
});

test('decode failures acknowledge and keep the last good image', async () => {
    const b = browser();
    let acks = 0;
    b.context.ack = () => acks++;
    const good = b.run('framePlayer.receive([1], ack)');
    b.decoded[0].resolve();
    await good;
    const bad = b.run('framePlayer.receive([2], ack)');
    b.decoded[1].reject(new Error('corrupt image'));
    await bad;
    assert.equal(acks, 2);
    assert.equal(b.elements.get('video').src, 'blob:1');
    assert.deepEqual(b.revoked, ['blob:2']);
});

test('disconnect cleans pending URLs and an old decode cannot paint over a new session', async () => {
    const b = browser();
    let acks = 0;
    b.context.ack = () => acks++;
    const old = b.run('framePlayer.receive([1], ack)');
    b.run('cleanup()');
    assert.equal(acks, 1);
    const fresh = b.run('framePlayer.receive([2], ack)');
    b.decoded[1].resolve();
    await fresh;
    b.decoded[0].resolve();
    await old;
    assert.equal(acks, 2);
    assert.equal(b.elements.get('video').src, 'blob:2');
    assert.deepEqual(b.revoked, ['blob:1']);
});

test('hidden pages release frame credit without allocating images', async () => {
    const b = browser();
    let acks = 0;
    b.context.ack = () => acks++;
    b.context.document.hidden = true;
    await b.run('framePlayer.receive([1], ack)');
    assert.equal(acks, 1);
    assert.equal(b.decoded.length, 0);
});

test('keyboard aliases, held/released keys and blur work while image decoding is stalled', () => {
    const b = browser();
    b.run('framePlayer.receive([1])'); // Deliberately never decode this frame.
    for (const [key, expected] of Object.entries({q:'7', w:'8', e:'9', a:'4', d:'6', z:'1', x:'2', c:'3', s:'s', ' ':'s', '0':'0', '8':'8'})) {
        b.context.key = key;
        b.run('handleKeydown({key, preventDefault() {}}); gameLoop(100);');
        assert.deepEqual(b.emitted.at(-1), { event: 'input', keys: [expected] });
        b.run('handleKeyup({key}); gameLoop(200);');
        assert.deepEqual(b.emitted.at(-1).keys, []);
    }
    b.run("handleKeydown({key: 'Q'}); gameLoop(300); handleKeyup({key: 'q'}); gameLoop(400);");
    assert.deepEqual(b.emitted.at(-1).keys, []);
    b.run("handleKeydown({key: ' ', preventDefault() {}}); handleKeydown({key: 's'}); gameLoop(500);");
    assert.deepEqual(b.emitted.at(-1).keys, ['s']);
    b.run("handleKeyup({key: ' '}); gameLoop(600);");
    assert.deepEqual(b.emitted.at(-1).keys, ['s']);
    b.windowHandlers.blur();
    assert.deepEqual(b.emitted.at(-1).keys, []);
});

test('touch action release, joystick and visibility clearing do not wait for frames', () => {
    const b = browser();
    b.run('framePlayer.receive([1])');
    b.run("setJoystickDirection('4'); gameLoop(100);");
    assert.deepEqual(b.emitted.at(-1).keys, ['4']);
    const handlers = b.elements.get('action-button').handlers;
    handlers.pointerdown({ pointerId: 1, preventDefault() {} });
    b.run('gameLoop(200)');
    assert.deepEqual(b.emitted.at(-1).keys, ['s', '4']);
    handlers.pointercancel({ pointerId: 1 });
    assert.deepEqual(b.emitted.at(-1).keys, ['4']);
    b.context.document.hidden = true;
    b.documentHandlers.visibilitychange();
    assert.deepEqual(b.emitted.at(-1).keys, []);
});

test('keyboard directions and fire are sent together and released independently', () => {
    const b = browser();
    b.run("handleKeydown({key: 'ArrowUp', preventDefault() {}}); handleKeydown({key: 'd'}); handleKeydown({key: ' ', preventDefault() {}}); gameLoop(100);");
    assert.deepEqual(b.emitted.at(-1).keys, ['8', '6', 's']);
    b.run("handleKeyup({key: ' '}); gameLoop(200);");
    assert.deepEqual(b.emitted.at(-1).keys, ['8', '6']);
    b.run("handleKeyup({key: 'ArrowUp'}); gameLoop(300);");
    assert.deepEqual(b.emitted.at(-1).keys, ['6']);
});
