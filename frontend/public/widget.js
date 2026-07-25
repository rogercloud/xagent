(function () {
  // Config
  var scriptTag = document.currentScript;
  var host = new URL(scriptTag.src).origin;

  var SESSION_TIMEOUT_MS = 5000;          // constants appendix #16
  var SESSION_RETRY_DELAYS = [1000, 2000, 4000];  // four attempts, three waits
  var SESSION_MAX_RATE_LIMIT_RETRIES = 2;
  var SESSION_REFRESH_THRESHOLD_MS = 60000;

  function sessionDelay(ms) {
    return new Promise(function (resolve) { setTimeout(resolve, ms); });
  }

  function retryAfterMs(result, fallback) {
    var header = result && result.retryAfter;
    var seconds = header ? parseInt(header, 10) : NaN;
    return isNaN(seconds) ? fallback : seconds * 1000;
  }

  // Retries only the transport classes: rejected fetches (network, CORS, CSP,
  // abort) and any 5xx, coded or not. 429 has its own small budget.
  function withRetry(makeRequest) {
    var attempt = 0;
    var rateLimited = 0;

    function retryOrGiveUp(result, error) {
      if (attempt >= SESSION_RETRY_DELAYS.length) {
        if (error) throw error;
        return result;
      }
      var wait = SESSION_RETRY_DELAYS[attempt];
      attempt += 1;
      return sessionDelay(wait).then(run);
    }

    function run() {
      return makeRequest().then(function (result) {
        if (result.status === 429) {
          if (rateLimited >= SESSION_MAX_RATE_LIMIT_RETRIES) return result;
          rateLimited += 1;
          return sessionDelay(retryAfterMs(result, 1000)).then(run);
        }
        if (result.status >= 500) return retryOrGiveUp(result, null);
        return result;
      }, function (error) {
        return retryOrGiveUp(null, error);
      });
    }

    return run();
  }

  // Single mode branch point: a delegated context grant switches the whole
  // identity channel. Either factory returns null after reporting a
  // fail-closed integration error, in which case nothing is rendered at all.
  var mode = scriptTag.hasAttribute('data-encrypted-context')
    ? createSessionMode(scriptTag, host)
    : createGuestMode(scriptTag, host);
  if (!mode) {
    return;
  }

  // Visual Configurations
  var buttonSize = scriptTag.getAttribute('data-button-size') || '60px';
  var buttonColor = scriptTag.getAttribute('data-button-color') || '#000';
  var iconColor = scriptTag.getAttribute('data-icon-color') || '#fff';
  var panelBgColor = scriptTag.getAttribute('data-panel-bg-color') || '#fff';

  // Styles
  var style = document.createElement('style');
  style.innerHTML = `
    .xagent-widget-container {
      position: fixed;
      bottom: 20px;
      right: 20px;
      z-index: 999999;
      font-family: system-ui, -apple-system, sans-serif;
    }

    .xagent-widget-fab {
      width: ${buttonSize};
      height: ${buttonSize};
      border-radius: 50%;
      background-color: ${buttonColor};
      color: ${iconColor};
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
      transition: transform 0.2s ease, opacity 0.2s ease;
      border: none;
      outline: none;
      padding: 0;
    }

    .xagent-widget-fab:hover {
      transform: scale(1.05);
      opacity: 0.9;
    }

    .xagent-widget-fab svg {
      width: calc(${buttonSize} * 0.53);
      height: calc(${buttonSize} * 0.53);
      fill: currentColor;
    }

    .xagent-widget-panel {
      position: absolute;
      bottom: calc(${buttonSize} + 20px);
      right: 0;
      width: 380px;
      height: 600px;
      max-height: calc(100vh - 100px);
      background: ${panelBgColor};
      border-radius: 12px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.16);
      overflow: hidden;
      opacity: 0;
      visibility: hidden;
      transform: translateY(20px);
      transition: opacity 0.3s ease, transform 0.3s ease, visibility 0.3s;
      border: 1px solid rgba(0,0,0,0.1);
    }

    .xagent-widget-panel.open {
      opacity: 1;
      visibility: visible;
      transform: translateY(0);
    }

    .xagent-widget-iframe {
      width: 100%;
      height: 100%;
      border: none;
      background: transparent;
    }

    @media (max-width: 480px) {
      .xagent-widget-panel {
        width: calc(100vw - 40px);
        height: calc(100vh - 120px);
      }
    }
  `;
  document.head.appendChild(style);

  // Container
  var container = document.createElement('div');
  container.className = 'xagent-widget-container';

  // Panel
  var panel = document.createElement('div');
  panel.className = 'xagent-widget-panel';

  // Iframe
  var iframe = document.createElement('iframe');
  iframe.className = 'xagent-widget-iframe';
  panel.appendChild(iframe);

  // FAB
  var fab = document.createElement('button');
  fab.className = 'xagent-widget-fab';
  // Chat icon SVG
  var chatIcon = '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/></svg>';
  // Close icon SVG
  var closeIcon = '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>';

  fab.innerHTML = chatIcon;

  var isOpen = false;
  fab.onclick = function () {
    isOpen = !isOpen;
    if (isOpen) {
      panel.classList.add('open');
      fab.innerHTML = closeIcon;
    } else {
      panel.classList.remove('open');
      fab.innerHTML = chatIcon;
    }
  };

  container.appendChild(panel);
  container.appendChild(fab);
  document.body.appendChild(container);

  mode.attach(iframe);

  function createGuestMode(scriptTag, host) {
    var token = scriptTag.getAttribute('data-token') || 'default';
    var widgetKey = scriptTag.getAttribute('data-widget-key');

    if (!widgetKey && token === 'default') {
      console.error('Xagent Widget: Missing data-widget-key attribute. Re-copy the embed snippet from the agent widget settings.');
      return null;
    }

    return {
      attach: function (iframe) {
        // Generate guest_id if not exists
        var guestId = localStorage.getItem('xagent_guest_id');
        if (!guestId) {
          guestId = 'guest_' + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
          localStorage.setItem('xagent_guest_id', guestId);
        }

        function loadIframe(ticket, agentId) {
          // The widget key is deliberately NOT placed in the iframe URL: the ticket
          // is sufficient to authenticate, and keeping the key out of the frame
          // means the embedded widget has no credential to fall back on.
          var url = host + '/widget/chat/' + token + '?guest_id=' + guestId;
          if (agentId) {
            url += '&agent_id=' + encodeURIComponent(agentId);
          }
          if (ticket) {
            url += '&embed_ticket=' + encodeURIComponent(ticket);
          }
          iframe.src = url;
        }

        // Request a short-lived embed ticket from the top-level page. This fetch
        // carries the embedding page's real, browser-enforced Origin header, which
        // the backend validates against allowed_domains before signing the ticket.
        // Fetches inside the iframe carry the iframe's own origin instead, so the
        // ticket is how the validated embedding origin reaches the auth call.
        if (widgetKey) {
          fetch(host + '/api/widget/embed-ticket', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ widget_key: widgetKey })
          })
            .then(function (res) {
              if (!res.ok) {
                // Fail closed: no ticket means auth would fail, and loading the
                // iframe anyway would let a non-allowlisted embed slip through the
                // direct-visit path. Surface an actionable error instead.
                console.error('Xagent Widget: embed authorization failed (HTTP ' + res.status + '). Check that this page is in the agent\'s allowed domains and that the embed snippet is current.');
                return null;
              }
              return res.json();
            })
            .then(function (data) {
              if (!data || !data.ticket) {
                return;
              }
              loadIframe(data.ticket, data.agent_id);
            })
            .catch(function (err) {
              console.error('Xagent Widget: embed authorization request failed (' + err + ').');
            });
        } else {
          // Deprecated data-token channel (dead server-side); loaded without a ticket.
          loadIframe(null, null);
        }
      }
    };
  }

  function xagentSessionRegistry() {
    if (!window.__xagentWidgetGrants) {
      window.__xagentWidgetGrants = {};
    }
    return window.__xagentWidgetGrants;
  }

  // 32-bit FNV-1a: a fast, dependency-free, synchronous string digest. This is
  // NOT a security boundary (no crypto guarantees) -- it only exists so the
  // dedupe registry never stores the grant plaintext itself, so that a
  // third-party script scraping window.__xagentWidgetGrants can't recover the
  // grant the way it could if the raw string were used as the key.
  function hashGrant(text) {
    var hash = 0x811c9dc5;
    for (var i = 0; i < text.length; i++) {
      hash ^= text.charCodeAt(i);
      hash = (hash * 0x01000193) >>> 0;
    }
    return 'g' + hash.toString(16);
  }

  function logSession(level, text, code, status) {
    var suffix = status ? ' (HTTP ' + status + ')' : '';
    console[level]('Xagent Widget: ' + text + ' [' + code + ']' + suffix + '.');
  }

  function createSessionMode(scriptTag, host) {
    var grant = (scriptTag.getAttribute('data-encrypted-context') || '').trim();

    // Fail closed before any network call, and render nothing at all: these are
    // integration mistakes on the embedding page, mirroring the missing
    // data-widget-key precedent. Never fall back to guest mode.
    if (!grant) {
      logSession('error', 'chat unavailable, the grant attribute is empty', 'grant_malformed');
      return null;
    }
    if (scriptTag.hasAttribute('data-widget-key') || scriptTag.hasAttribute('data-token')) {
      logSession('error', 'chat unavailable, remove the legacy widget attribute', 'attribute_conflict');
      return null;
    }

    var registry = xagentSessionRegistry();
    var registryKey = hashGrant(grant);
    if (registry[registryKey]) {
      logSession('warn', 'this grant is already running on the page, ignoring the duplicate embed', 'duplicate_init');
      return null;
    }
    registry[registryKey] = true;

    // The grant lives in the closure from here on; keeping it in the DOM would
    // let any third-party script on the page scrape it off the script tag.
    scriptTag.removeAttribute('data-encrypted-context');

    return createSessionController(grant, host);
  }

  function createSessionController(grant, host) {
    var state = {
      grant: grant,
      iframe: null,
      ready: false,
      session: null,
      reconnectToken: null,
      terminalCode: null,
      settled: false,
      inflight: { exchange: null, reconnect: null }
    };

    function attach(iframe) {
      state.iframe = iframe;
      iframe.src = host + '/widget/chat/session';
      window.addEventListener('message', onMessage);
      runLoadFlow();
    }

    function runLoadFlow() {
      return exchange();
    }

    function onMessage(event) {
      if (!state.iframe || event.source !== state.iframe.contentWindow) return;
      if (event.origin !== host) return;
      var data = event.data;
      if (!data || data.xagent !== true || data.v !== 1) return;

      if (data.type === 'ready') {
        state.ready = true;
        flush();
      } else if (data.type === 'reconnect_request') {
        onReconnectRequest();
      }
    }

    function send(message) {
      message.xagent = true;
      message.v = 1;
      state.iframe.contentWindow.postMessage(message, host);
    }

    // Level-triggered: every ready re-sends whatever the current state is, and
    // only the latest state is ever sent. Replaying an older session_update
    // would hand the iframe an already-rotated token.
    function flush() {
      if (!state.ready) return;
      if (state.terminalCode) {
        send({ type: 'session_terminal', code: state.terminalCode });
        return;
      }
      if (!state.session) return;
      send({
        type: 'session_update',
        session_token: state.session.session_token,
        session_token_expires_at: state.session.session_token_expires_at,
        absolute_expires_at: state.session.absolute_expires_at,
        agent: state.session.agent
      });
    }

    function applySession(data) {
      state.session = {
        session_token: data.session_token,
        session_token_expires_at: data.session_token_expires_at,
        absolute_expires_at: data.session && data.session.absolute_expires_at,
        agent: data.session && data.session.agent
      };
      state.reconnectToken = data.reconnect_token;
    }

    function goTerminal(code, status) {
      if (state.terminalCode) return;
      state.terminalCode = code;
      logSession('error', 'chat unavailable', code, status);
      flush();
    }

    var STALE_GRANT_CODES = { grant_expired: true, grant_already_used: true };

    function handleResult(result, phase) {
      if (result.ok && result.data && result.data.session_token) {
        if (phase === 'exchange') {
          if (state.settled) return;   // first response wins
          state.settled = true;
        }
        applySession(result.data);
        flush();
        return;
      }

      var code = result.data && result.data.error && result.data.error.code;
      if (!code) {
        goTerminal(result.status >= 500 ? 'network_unavailable' : 'unexpected_error', result.status);
        return;
      }
      if (phase === 'exchange' && STALE_GRANT_CODES[code] && state.reconnectToken) {
        reconnect();
        return;
      }
      goTerminal(code, result.status);
    }

    function postJson(url, body) {
      var controller = new AbortController();
      var timer = setTimeout(function () { controller.abort(); }, SESSION_TIMEOUT_MS);
      return fetch(url, {
        method: 'POST',
        credentials: 'omit',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal
      }).then(function (res) {
        clearTimeout(timer);
        // .clone() before reading: mocks (and retried real fetches) can hand back
        // the same Response instance across attempts, and a body can only be
        // read once from any given instance.
        return res.clone().json().catch(function () { return null; }).then(function (data) {
          return {
            ok: res.ok,
            status: res.status,
            data: data,
            retryAfter: res.headers.get('Retry-After')
          };
        });
      }, function (err) {
        clearTimeout(timer);
        throw err;
      });
    }

    function exchange() {
      return withRetry(function () {
        return postJson(host + '/v1/external/chat/sessions', { encrypted_context: state.grant });
      }).then(function (result) {
        handleResult(result, 'exchange');
      }, function () {
        goTerminal('network_unavailable');
      });
    }

    function onReconnectRequest() {
      reconnect();
    }

    function reconnect() {
      var body = { reconnect_token: state.reconnectToken };
      if (state.grant) body.encrypted_context = state.grant;
      return withRetry(function () {
        return postJson(host + '/v1/external/chat/sessions/reconnect', body);
      }).then(function (result) {
        handleResult(result, 'reconnect');
      }, function () {
        goTerminal('network_unavailable');
      });
    }

    return { attach: attach };
  }
})();
