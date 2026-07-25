(function () {
  // Config
  var scriptTag = document.currentScript;
  var host = new URL(scriptTag.src).origin;

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
    if (registry[grant]) {
      logSession('warn', 'this grant is already running on the page, ignoring the duplicate embed', 'duplicate_init');
      return null;
    }
    registry[grant] = true;

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
      runLoadFlow();
    }

    function runLoadFlow() {
      return exchange();
    }

    function exchange() {
      return fetch(host + '/v1/external/chat/sessions', {
        method: 'POST',
        credentials: 'omit',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ encrypted_context: state.grant })
      });
    }

    return { attach: attach };
  }
})();
