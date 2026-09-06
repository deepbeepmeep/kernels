const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function productionFunction(name) {
  const source = fs.readFileSync(path.join(__dirname, '..', 'shared', 'gradio', 'assistant_chat.py'), 'utf8');
  const marker = `${name} = function`;
  const start = source.indexOf(marker);
  assert.notEqual(start, -1, `missing ${name}`);
  const brace = source.indexOf('{', start);
  let depth = 0;
  let quote = '';
  let escaped = false;
  for (let index = brace; index < source.length; index += 1) {
    const char = source[index];
    if (quote) {
      if (escaped) escaped = false;
      else if (char === '\\') escaped = true;
      else if (char === quote) quote = '';
      continue;
    }
    if (char === '"' || char === "'" || char === '`') {
      quote = char;
      continue;
    }
    if (char === '{') depth += 1;
    else if (char === '}') {
      depth -= 1;
      if (depth === 0) return source.slice(start + name.length + 3, index + 1);
    }
  }
  throw new Error(`unterminated ${name}`);
}

function streamingMarkdownHarness() {
  class FakeNode {
    constructor(tagName = '', text = '') {
      this.nodeType = tagName ? 1 : 3;
      this.tagName = tagName ? tagName.toUpperCase() : '';
      this.children = [];
      this.parentNode = null;
      this.className = '';
      this._text = text;
    }
    appendChild(child) {
      child.parentNode = this;
      this.children.push(child);
      return child;
    }
    replaceChildren(...children) {
      for (const child of this.children) child.parentNode = null;
      this.children = [];
      for (const child of children) this.appendChild(child);
    }
    remove() {
      if (!this.parentNode) return;
      this.parentNode.children = this.parentNode.children.filter((child) => child !== this);
      this.parentNode = null;
    }
    get lastElementChild() {
      return [...this.children].reverse().find((child) => child.nodeType === 1) || null;
    }
    get textContent() {
      return this.nodeType === 3 ? this._text : this.children.map((child) => child.textContent).join('');
    }
    set textContent(value) {
      if (this.nodeType === 3) this._text = String(value);
      else this.replaceChildren(new FakeNode('', String(value)));
    }
  }
  const document = {
    baseURI: 'http://127.0.0.1:7865/',
    createElement(tagName) { return new FakeNode(tagName); },
    createTextNode(text) { return new FakeNode('', String(text)); },
  };
  const WAC = {};
  const context = { WAC, document, URL };
  vm.createContext(context);
  for (const name of ['safeStreamingMarkdownUrl', 'streamingMarkdownDelimiterFlags', 'findStreamingMarkdownCloser', 'appendStreamingInlineMarkdown', 'resetStreamingMarkdown', 'appendStreamingListItem', 'renderStreamingMarkdownLine', 'renderStreamingMarkdown']) {
    vm.runInContext(`WAC.${name} = ${productionFunction(`WAC.${name}`)};`, context);
  }
  return { WAC, root: new FakeNode('div') };
}

function consumer() {
  const calls = [];
  let parseCount = 0;
  const WAC = {
    lastPayloadId: '', lastPayloadText: '', serverInstanceId: '', chatSessionId: '', chatRevision: -1, chatSequence: -1,
    pendingSteeringId: '', state: { status: null }, syncRequired: false,
    reset() { this.chatRevision = -1; this.chatSequence = -1; },
    markSyncRequired(event) { this.syncRequired = true; calls.push(['gap', event.type]); },
    mergeStaleSync() { calls.push(['stale-sync']); }, acknowledgeOptimisticSubmits() {}, setStatus() {}, setStats() {},
    upsertMessage(value, preserve, messageIndex) { calls.push(['message', value.id, preserve, messageIndex]); }, removeMessage(value) { calls.push(['remove-message', value]); },
    upsertBlock(value) { calls.push(['upsert', value.block_id]); }, appendBlockText(value) { calls.push(['append', value.text]); },
    replaceBlockText(value) { calls.push(['replace', value.text]); }, finalizeBlock(value) { calls.push(['finalize', value.block_id]); },
    removeBlock(value) { calls.push(['remove-block', value.block_id]); },
    sync() { calls.push(['sync']); },
  };
  const context = { WAC, calls, console, JSON: { parse(value) { parseCount += 1; return JSON.parse(value); }, stringify: JSON.stringify } };
  vm.createContext(context);
  vm.runInContext(`WAC.consumePayload = ${productionFunction('WAC.consumePayload')};`, context);
  context.parseCount = () => parseCount;
  return context;
}

function envelope(id, event) {
  return JSON.stringify({ event_id: id, instance_id: 'server', event: { chat_session_id: 'session', revision: event.revision ?? 1, ...event } });
}

test('production consumer recovers a missing text delta before the thought finishes', () => {
  const context = consumer();
  const { WAC, calls } = context;
  let requested = 0;
  WAC.requestCanonicalSync = () => { requested += 1; return true; };
  context.window = { dispatchEvent() {} };
  context.CustomEvent = function () {};
  vm.runInContext(`WAC.markSyncRequired = ${productionFunction('WAC.markSyncRequired')};`, context);
  WAC.consumePayload(envelope('sync-1', { type: 'sync', sequence: 1, sequence_start: 1, messages: [] }));
  WAC.consumePayload(envelope('block-2', { type: 'upsert_block', sequence: 2, sequence_start: 2, block_id: 'b', message_id: 'm' }));
  WAC.consumePayload(envelope('append-3', { type: 'append_block_text', sequence: 3, sequence_start: 3, block_id: 'b', message_id: 'm', text: 'x' }));
  WAC.consumePayload(envelope('append-3-duplicate-id', { type: 'append_block_text', sequence: 3, sequence_start: 3, block_id: 'b', message_id: 'm', text: 'x' }));
  assert.equal(requested, 0);
  // Lose sequence 4, then deliver twenty seconds' worth of 250 ms updates.
  for (let sequence = 5; sequence <= 84; sequence += 1) {
    WAC.consumePayload(envelope(`append-${sequence}`, { type: 'append_block_text', sequence, sequence_start: sequence, block_id: 'b', message_id: 'm', text: 'gap' }));
  }
  assert.equal(requested, 1);
  assert.equal(WAC.syncRecoveryPending, true);
  assert.deepEqual(calls.filter(([kind]) => kind === 'append'), [['append', 'x']]);
  WAC.consumePayload(envelope('recovery-85', { type: 'sync', sequence: 85, sequence_start: 85, revision: 85, messages: [] }));
  WAC.consumePayload(envelope('append-86', { type: 'append_block_text', sequence: 86, sequence_start: 86, revision: 86, block_id: 'b', message_id: 'm', text: 'resumed' }));
  assert.deepEqual(calls.filter(([kind]) => kind === 'append'), [['append', 'x'], ['append', 'resumed']]);
  assert.deepEqual(calls.filter(([kind]) => kind === 'finalize'), []);
  assert.equal(WAC.chatSequence, 86);
  assert.equal(WAC.syncRequired, false);
  assert.equal(WAC.syncRecoveryPending, false);
  assert.equal(requested, 1);
});

test('production consumer requests recovery for a replacement after a sequence gap', () => {
  const { WAC, calls } = consumer();
  WAC.consumePayload(envelope('sync-1', { type: 'sync', sequence: 1, sequence_start: 1, messages: [] }));
  WAC.consumePayload(envelope('replace-3', { type: 'replace_block_text', sequence: 3, sequence_start: 3, block_id: 'b', message_id: 'm', text: 'rewritten' }));
  assert.deepEqual(calls.filter(([kind]) => kind === 'gap'), [['gap', 'replace_block_text']]);
  assert.deepEqual(calls.filter(([kind]) => kind === 'replace'), []);
  assert.equal(WAC.chatSequence, 1);
});

test('production consumer still requests a canonical sync for a missing destructive event', () => {
  const { WAC, calls } = consumer();
  WAC.consumePayload(envelope('sync-1', { type: 'sync', sequence: 1, sequence_start: 1, messages: [] }));
  WAC.consumePayload(envelope('remove-3', { type: 'remove_message', sequence: 3, sequence_start: 3, revision: 3, message_id: 'm' }));
  assert.deepEqual(calls.filter(([kind]) => kind === 'gap'), [['gap', 'remove_message']]);
  assert.equal(WAC.chatSequence, 1);
  assert.equal(WAC.syncRequired, true);
});

test('production consumer dispatches replacement, finalization, and removal in order', () => {
  const { WAC, calls } = consumer();
  WAC.consumePayload(envelope('sync', { type: 'sync', sequence: 10, sequence_start: 10, revision: 10, messages: [] }));
  WAC.consumePayload(envelope('replace', { type: 'replace_block_text', sequence: 11, sequence_start: 11, revision: 11, block_id: 'b', message_id: 'm', text: 'new' }));
  WAC.consumePayload(envelope('final', { type: 'finalize_block', sequence: 12, sequence_start: 12, revision: 12, block_id: 'b', message_id: 'm' }));
  WAC.consumePayload(envelope('remove', { type: 'remove_block', sequence: 13, sequence_start: 13, revision: 13, block_id: 'b', message_id: 'm' }));
  assert.deepEqual(calls.slice(-3), [['replace', 'new'], ['finalize', 'b'], ['remove-block', 'b']]);
});

test('message events forward their canonical transcript position', () => {
  const { WAC, calls } = consumer();
  WAC.consumePayload(envelope('message-position', { type: 'upsert_message', sequence: 1, sequence_start: 1, revision: 1, message_index: 2, message: { id: 'assistant_1' } }));
  assert.deepEqual(calls, [['message', 'assistant_1', false, 2]]);
});

test('production consumer accepts a coalesced contiguous sequence range', () => {
  const { WAC, calls } = consumer();
  WAC.consumePayload(envelope('sync-range', { type: 'sync', sequence: 1, sequence_start: 1, revision: 1, messages: [] }));
  WAC.consumePayload(envelope('append-range', { type: 'append_block_text', sequence: 5, sequence_start: 2, revision: 5, block_id: 'b', message_id: 'm', text: 'abcd' }));
  assert.deepEqual(calls.filter(([kind]) => kind === 'gap'), []);
  assert.deepEqual(calls.filter(([kind]) => kind === 'append'), [['append', 'abcd']]);
  assert.equal(WAC.chatSequence, 5);
});

test('duplicate polling skips JSON parsing before dispatch', () => {
  const { WAC, calls, parseCount } = consumer();
  const payload = envelope('same-payload', { type: 'sync', sequence: 1, sequence_start: 1, messages: [] });
  WAC.consumePayload(payload);
  WAC.consumePayload(payload);
  assert.equal(parseCount(), 1);
  assert.deepEqual(calls, [['sync']]);
});

test('replay batches suppress intermediate UI work and restore the transcript once', () => {
  const { WAC, calls } = consumer();
  const transcript = { style: { visibility: '' } };
  const scroll = { scrollTop: 0, scrollHeight: 480 };
  const shellClasses = new Set();
  const shellAttributes = new Map();
  const shell = {
    classList: { add(value) { shellClasses.add(value); }, remove(value) { shellClasses.delete(value); } },
    setAttribute(key, value) { shellAttributes.set(key, value); },
    removeAttribute(key) { shellAttributes.delete(key); },
  };
  let disclosureRefreshes = 0;
  let emptyRefreshes = 0;
  WAC.replayDepth = 0;
  WAC.ensureShell = () => true;
  WAC.shell = () => shell;
  WAC.transcript = () => transcript;
  WAC.applyDisclosureState = () => { disclosureRefreshes += 1; };
  WAC.showEmptyIfNeeded = () => { emptyRefreshes += 1; };
  WAC.scroll = () => scroll;
  WAC.syncJumpToBottom = () => {};
  const batch = {
    event_id: 'replay-batch',
    instance_id: 'server',
    replay: true,
    batch: [
      JSON.parse(envelope('replay-reset', { type: 'reset', sequence: 1, revision: 1 })),
      JSON.parse(envelope('replay-message', { type: 'upsert_message', sequence: 2, sequence_start: 2, revision: 1, message_index: 0, message: { id: 'user_1' } })),
    ],
  };

  WAC.consumePayload(JSON.stringify(batch));

  assert.deepEqual(calls, [['message', 'user_1', false, 0]]);
  assert.equal(WAC.replayDepth, 0);
  assert.equal(transcript.style.visibility, '');
  assert.equal(disclosureRefreshes, 1);
  assert.equal(emptyRefreshes, 1);
  assert.equal(scroll.scrollTop, 480);
  assert.equal(shellClasses.has('is-replaying'), false);
  assert.equal(shellAttributes.has('aria-busy'), false);
});

test('streaming markdown appends plain text cheaply and renders a delimiter when it closes', () => {
  const { WAC, root } = streamingMarkdownHarness();
  const renderInline = WAC.appendStreamingInlineMarkdown;
  let parseCount = 0;
  WAC.appendStreamingInlineMarkdown = (...args) => {
    parseCount += 1;
    return renderInline(...args);
  };
  WAC.renderStreamingMarkdown(root, 'Hello ');
  assert.equal(parseCount, 1);
  const initialTail = root.__wangpStreamingMarkdown.tail;
  WAC.renderStreamingMarkdown(root, 'Hello world');
  assert.equal(root.__wangpStreamingMarkdown.tail, initialTail);
  assert.equal(parseCount, 1);
  assert.equal(root.textContent, 'Hello world');

  WAC.renderStreamingMarkdown(root, 'Hello world **bo');
  const openTail = root.__wangpStreamingMarkdown.tail;
  const openParseCount = parseCount;
  WAC.renderStreamingMarkdown(root, 'Hello world **bold');
  assert.equal(root.__wangpStreamingMarkdown.tail, openTail);
  assert.equal(parseCount, openParseCount);
  assert.equal(root.textContent, 'Hello world **bold');

  WAC.renderStreamingMarkdown(root, 'Hello world **bold**');
  const tags = [];
  const visit = (node) => {
    if (node.tagName) tags.push(node.tagName);
    for (const child of node.children) visit(child);
  };
  visit(root);
  assert.ok(tags.includes('STRONG'));
  assert.equal(root.textContent, 'Hello world bold');

  const linkRoot = streamingMarkdownHarness();
  linkRoot.WAC.renderStreamingMarkdown(linkRoot.root, '[OpenAI](https://ope');
  assert.equal(linkRoot.root.textContent, '[OpenAI](https://ope');
  linkRoot.WAC.renderStreamingMarkdown(linkRoot.root, '[OpenAI](https://openai.com)');
  const linkTags = [];
  const visitLink = (node) => {
    if (node.tagName) linkTags.push(node.tagName);
    for (const child of node.children) visitLink(child);
  };
  visitLink(linkRoot.root);
  assert.ok(linkTags.includes('A'));
  assert.equal(linkRoot.root.textContent, 'OpenAI');
});

test('streaming markdown does not turn list markers or identifier underscores into italics', () => {
  const { WAC, root } = streamingMarkdownHarness();
  WAC.renderStreamingMarkdown(root, '* list item with another * marker');
  WAC.renderStreamingMarkdown(root, '* list item with another * marker and snake_case_name');
  const tags = [];
  const visit = (node) => {
    if (node.tagName) tags.push(node.tagName);
    for (const child of node.children) visit(child);
  };
  visit(root);
  assert.ok(!tags.includes('EM'));
  assert.equal(root.textContent, 'list item with another * marker and snake_case_name');

  WAC.renderStreamingMarkdown(root, '* list item with another * marker and snake_case_name\nThis is _actually italic_.');
  const updatedTags = [];
  const visitUpdated = (node) => {
    if (node.tagName) updatedTags.push(node.tagName);
    for (const child of node.children) visitUpdated(child);
  };
  visitUpdated(root);
  assert.equal(updatedTags.filter((tag) => tag === 'EM').length, 1);
});

test('streaming list indentation matches finalized Python Markdown rules', () => {
  const tags = (root) => {
    const result = [];
    const visit = (node) => {
      if (node.tagName) result.push(node.tagName);
      for (const child of node.children) visit(child);
    };
    visit(root);
    return result;
  };

  const inline = streamingMarkdownHarness();
  inline.WAC.renderStreamingMarkdown(inline.root, 'Request ledger\n1. First\n2. Second\n');
  assert.ok(!tags(inline.root).includes('OL'));

  const parenthesized = streamingMarkdownHarness();
  parenthesized.WAC.renderStreamingMarkdown(parenthesized.root, '1) First\n2) Second\n');
  assert.ok(!tags(parenthesized.root).includes('OL'));

  const list = streamingMarkdownHarness();
  list.WAC.renderStreamingMarkdown(list.root, 'Request ledger\n\n1. First\n2. Second\n');
  assert.equal(tags(list.root).filter((tag) => tag === 'OL').length, 1);
  assert.equal(tags(list.root).filter((tag) => tag === 'LI').length, 2);
});

test('a list item is indented as its marker arrives, before a newline', () => {
  for (const marker of ['1.', '*', '-']) {
    const { WAC, root } = streamingMarkdownHarness();
    let text = '';
    for (const fragment of [marker, ' ', 'Current', ' paragraph', ' with **bold**']) {
      text += fragment;
      WAC.renderStreamingMarkdown(root, text);
      if (text === marker) continue;
      const lists = root.children.filter((node) => ['OL', 'UL'].includes(node.tagName));
      assert.equal(lists.length, 1);
      assert.equal(lists[0].children.length, 1);
      assert.equal(lists[0].lastElementChild.tagName, 'LI');
      assert.equal(lists[0].textContent, text.slice(marker.length + 1).replaceAll('**', ''));
    }
    WAC.renderStreamingMarkdown(root, text + '\n');
    assert.equal(root.children.filter((node) => ['OL', 'UL'].includes(node.tagName)).length, 1);
  }
});

test('blank lines between numbered items keep a single continuous list', () => {
  const { WAC, root } = streamingMarkdownHarness();
  let text = '1. First';
  WAC.renderStreamingMarkdown(root, text);
  for (const fragment of ['\n\n', '2. Second', '\n\n', '3. Third', '\n']) {
    text += fragment;
    WAC.renderStreamingMarkdown(root, text);
  }
  const lists = root.children.filter((node) => node.tagName === 'OL');
  assert.equal(lists.length, 1);
  assert.equal(lists[0].children.length, 3);
  assert.deepEqual(lists[0].children.map((item) => item.textContent), ['First', 'Second', 'Third']);
});

test('a separate numbered list preserves its starting number', () => {
  const { WAC, root } = streamingMarkdownHarness();
  WAC.renderStreamingMarkdown(root, '4. Fourth\n\n5. Fifth\n\nA paragraph.\n\n9. Ninth');
  const lists = root.children.filter((node) => node.tagName === 'OL');
  assert.equal(lists.length, 2);
  assert.equal(lists[0].start, 4);
  assert.equal(lists[1].start, 9);
});

test('event source value writes publish immediately without polling', () => {
  const calls = [];
  const WAC = { consumePayload(value) { calls.push(value); } };
  const context = { WAC, calls };
  vm.createContext(context);
  vm.runInContext(`
    WAC.observeEventSourceValue = ${productionFunction('WAC.observeEventSourceValue')};
    const prototype = {};
    Object.defineProperty(prototype, 'value', {
      configurable: true,
      get() { return this._value || ''; },
      set(value) { this._value = String(value); },
    });
    const node = Object.create(prototype);
    WAC.observeEventSourceValue(node);
    node.value = ' first event ';
    node.value = 'second event';
  `, context);
  assert.deepEqual(calls, ['first event', 'second event']);
});

test('canonical split requests keep following the submitted request to the last card', () => {
  const calls = [];
  const WAC = {
    followSubmissionId: 'optimistic_batch',
    blockState: { old: true },
    ensureShell() {}, captureDisclosureState() {}, captureAutoscrollState() { return { atBottom: false, top: 120 }; },
    syncAcknowledgesFollowedSubmission: null,
    replaceState() {}, reconcileOptimisticSubmits() {},
    hydrate(state) { calls.push(['hydrate', state]); },
    scrollToBottomAfterLayout() { calls.push(['bottom']); },
    transcript() { return null; },
  };
  const context = { WAC, calls, Array, String };
  vm.createContext(context);
  vm.runInContext(`WAC.syncAcknowledgesFollowedSubmission = ${productionFunction('WAC.syncAcknowledgesFollowedSubmission')};`, context);
  vm.runInContext(`WAC.sync = ${productionFunction('WAC.sync')};`, context);
  WAC.sync([
    { id: 'user_1', client_submission_id: 'optimistic_batch' },
    { id: 'user_2', client_submission_id: 'optimistic_batch' },
    { id: 'user_3', client_submission_id: 'optimistic_batch' },
  ], null, null, ['optimistic_batch']);
  assert.equal(calls.length, 2);
  assert.equal(calls[0][0], 'hydrate');
  assert.equal(calls[0][1].atBottom, true);
  assert.equal(calls[0][1].top, 0);
  assert.equal(calls[1][0], 'bottom');
  assert.equal(WAC.followSubmissionId, '');
});

test('production gap recovery requests one canonical sync until recovery arrives', () => {
  let requested = 0;
  const WAC = { syncRequired: false, syncRecoveryPending: false, chatSessionId: 'session', chatSequence: 7, requestCanonicalSync() { requested += 1; return true; } };
  const context = { WAC, window: { dispatchEvent() {} }, CustomEvent: function () {} };
  vm.createContext(context);
  vm.runInContext(`WAC.markSyncRequired = ${productionFunction('WAC.markSyncRequired')};`, context);
  WAC.markSyncRequired({ type: 'append_block_text' });
  WAC.markSyncRequired({ type: 'append_block_text' });
  assert.equal(requested, 1);
  assert.equal(WAC.syncRecoveryPending, true);
});

test('optimistic busy requests expose queued state and actions immediately', () => {
  const WAC = {
    normalizeText(value) { return String(value || '').trim(); },
    escapeHtml(value) { return String(value || '').replaceAll('&', '&amp;').replaceAll("'", '&#39;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'); },
    timeLabel() { return '10:15'; },
  };
  const context = { WAC };
  vm.createContext(context);
  vm.runInContext(`WAC.buildOptimisticUserMessage = ${productionFunction('WAC.buildOptimisticUserMessage')};`, context);
  const queued = WAC.buildOptimisticUserMessage('optimistic_test', 'Queue me', 0, 'Queued');
  const normal = WAC.buildOptimisticUserMessage('optimistic_normal', 'Run me', 0, '');
  assert.match(queued.html, /chat__badge'>Queued/);
  assert.match(queued.html, /data-message-action='steer'/);
  assert.match(queued.html, /data-message-action='edit'/);
  assert.match(queued.html, /data-message-action='remove'/);
  assert.equal(queued.client_submission_id, 'optimistic_test');
  assert.doesNotMatch(normal.html, /data-message-action=/);
});

test('optimistic Ctrl Enter steering is inserted ahead of the queued tail', () => {
  const calls = [];
  const WAC = {
    state: {
      order: ['active', 'assistant', 'queued-1', 'queued-2'],
      messages: {
        active: { id: 'active', role: 'user', queued: false },
        assistant: { id: 'assistant', role: 'assistant', queued: false },
        'queued-1': { id: 'queued-1', role: 'user', queued: true },
        'queued-2': { id: 'queued-2', role: 'user', queued: true },
      },
    },
    optimisticSubmits: [], optimisticMaxAgeMs: 30000, followSubmissionId: '',
    normalizeText(value) { return String(value || '').trim(); },
    newSubmissionId() { return 'optimistic-steer'; },
    buildOptimisticUserMessage(id, content, _now, badge) { return { id, content, role: 'user', badge, queued: true }; },
    upsertMessage(message, _preserve, index) { calls.push([message.id, index]); },
    scrollToBottomAfterLayout() {}, dropOptimisticSubmit() {}, removeMessage() {},
  };
  const context = { WAC, window: { setTimeout() {} }, Date };
  vm.createContext(context);
  for (const name of ['queuedTailInsertIndex', 'pushOptimisticUserMessage']) vm.runInContext(`WAC.${name} = ${productionFunction(`WAC.${name}`)};`, context);
  WAC.pushOptimisticUserMessage('urgent steering', 'Steered');
  assert.deepEqual(calls, [['optimistic-steer', 2]]);
});
