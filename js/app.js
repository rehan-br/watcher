/* Watcher — reverse Wordle. The user picks a 5-letter word; the LLM tries
 * to guess it. The page renders the model's reasoning in a notebook panel.
 *
 * Architecture: a single-page state machine with two phases.
 *   Phase 1 (entry)  — user picks the secret word.
 *   Phase 2 (game)   — model guesses, user confirms tile colours per turn.
 *
 * The secret word never leaves the browser. The /api/guess endpoint only
 * receives the model's prior guesses + the colours the user confirmed.
 *
 * Anti-cheat (collaborative with /api/guess):
 *   - All state lives inside this IIFE closure; it is not exposed on
 *     window or any global. Casual inspection via dev tools is harder.
 *   - Every API response carries an HMAC-signed `token` for that turn.
 *     We echo all prior tokens on every subsequent request. The server
 *     re-validates the chain — meaning a user CANNOT inject a fake turn
 *     ("Watcher solved my word in 1 guess") without also forging the
 *     server's HMAC secret. Tampering with prior colors also breaks the
 *     chain because tokens bind word + prior color snapshot.
 *   - Tile colours for the *current* row are auto-computed locally from
 *     the secret. A user could in theory override them via dev tools,
 *     but this only lies to the AI in the user's own session — it can't
 *     produce a falsified "Watcher won" screenshot, because the server
 *     still drives every guess word and signs every turn.
 */
(function () {
    'use strict';

    // -----------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------
    const MAX_GUESSES = 6;
    const WORD_LEN = 5;
    const TILE_STATES = ['absent', 'present', 'correct'];

    // -----------------------------------------------------------------
    // LLM — POSTs to the Pages Function. The function is the single
    // source of truth for guesses; every response carries an HMAC token
    // that we echo back on subsequent turns. Tampering with prior turns
    // (or fabricating a "model said X" turn that the server never issued)
    // breaks the chain server-side.
    //
    // For local dev (wrangler not running, no Gemini key) there is a
    // gated stub that activates only on localhost. Stub responses use a
    // clearly-marked "STUB_..." token format. The production server's
    // HMAC validator rejects those tokens, so a localhost preview can't
    // be screenshotted into a fake "real" run.
    // -----------------------------------------------------------------
    const IS_LOCAL = (function () {
        const h = location.hostname;
        return h === 'localhost' || h === '127.0.0.1' || h === '' || h.endsWith('.local');
    })();

    const STUB_OPENERS = [
        { guess: 'STARE', reasoning: "Opening move. STARE is a textbook starter — covers two of the most common vowels (A, E) plus three high-frequency consonants (S, T, R). Best information yield with no prior data." },
        { guess: 'ROUND', reasoning: "Expanding vowel coverage. ROUND brings in O and U, plus N and D which appear often in the middle and end of 5-letter words." },
        { guess: 'CLIMP', reasoning: "Filling in remaining frequent consonants. C, L, M, P haven't been tested yet, and I haven't tried I as a vowel either." },
        { guess: 'BUNGY', reasoning: "I've explored the common letters. Now to test some less-frequent ones — B, G, Y — to narrow the candidate set." },
        { guess: 'WHACK', reasoning: "Running out of guesses. Throwing W, H, K into the mix to eliminate more dead ends." },
        { guess: 'JUMPS', reasoning: "Last shot. J and the remaining frequent letters." }
    ];

    async function stubGenerate(history) {
        await new Promise((r) => setTimeout(r, 700)); // honest-feeling latency
        const turn = history.length;
        const opener = STUB_OPENERS[Math.min(turn, STUB_OPENERS.length - 1)];
        let summary = '';
        if (turn > 0) {
            const prev = history[turn - 1];
            const correctLetters = [];
            const presentLetters = [];
            const absentLetters = [];
            for (let i = 0; i < prev.word.length; i++) {
                const ch = prev.word[i];
                const c = prev.colors[i];
                if (c === 'correct') correctLetters.push(ch + '@' + (i + 1));
                else if (c === 'present') presentLetters.push(ch);
                else absentLetters.push(ch);
            }
            summary = [
                correctLetters.length ? `Locked: ${correctLetters.join(', ')}.` : 'No exact letters yet.',
                presentLetters.length ? `In the word, wrong spot: ${presentLetters.join(', ')}.` : '',
                absentLetters.length ? `Eliminated: ${absentLetters.join(', ')}.` : ''
            ].filter(Boolean).join(' ') + ' ';
        }
        return {
            guess: opener.guess,
            reasoning: '[local stub — not the real model] ' + summary + opener.reasoning,
            token: 'STUB_' + Math.random().toString(36).slice(2, 10) + '_' + turn
        };
    }

    async function llmGenerate(history) {
        let res;
        try {
            res = await fetch('/api/guess', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    word_length: WORD_LEN,
                    history: history.map((h) => ({
                        word: h.word,
                        colors: h.colors,
                        token: h.token
                    }))
                })
            });
        } catch (netErr) {
            if (IS_LOCAL) {
                console.warn('[watcher] /api/guess unreachable, using local stub:', netErr);
                return stubGenerate(history);
            }
            throw netErr;
        }
        if (!res.ok) {
            if (IS_LOCAL && (res.status === 404 || res.status === 405 || res.status >= 500)) {
                console.warn('[watcher] /api/guess returned', res.status, '— using local stub');
                return stubGenerate(history);
            }
            const detail = await res.text().catch(() => '');
            throw new Error(`api ${res.status}: ${detail.slice(0, 160)}`);
        }
        const data = await res.json();
        if (
            typeof data.guess !== 'string' ||
            typeof data.reasoning !== 'string' ||
            typeof data.token !== 'string'
        ) {
            throw new Error('bad response shape from /api/guess');
        }
        const word = data.guess.toUpperCase().replace(/[^A-Z]/g, '');
        if (word.length !== WORD_LEN) throw new Error('bad guess length');
        return { guess: word, reasoning: data.reasoning, token: data.token };
    }

    // -----------------------------------------------------------------
    // Wordle scoring — used to pre-fill the user's confirmation defaults.
    // Two-pass algorithm: lock in greens first, then yellows from the
    // remaining letters. The user can override any tile.
    // -----------------------------------------------------------------
    function scoreGuess(secret, guess) {
        const colors = new Array(WORD_LEN).fill('absent');
        const remaining = [];
        for (let i = 0; i < WORD_LEN; i++) {
            if (guess[i] === secret[i]) {
                colors[i] = 'correct';
            } else {
                remaining.push(secret[i]);
            }
        }
        for (let i = 0; i < WORD_LEN; i++) {
            if (colors[i] === 'correct') continue;
            const idx = remaining.indexOf(guess[i]);
            if (idx >= 0) {
                colors[i] = 'present';
                remaining.splice(idx, 1);
            }
        }
        return colors;
    }

    // -----------------------------------------------------------------
    // State
    // -----------------------------------------------------------------
    const state = {
        secret: '',
        history: [],         // confirmed turns: [{ word, colors, reasoning, token }]
        pending: null,       // current turn waiting on confirm: { word, colors, reasoning, token }
        thinking: false,
        gameOver: false
    };

    // -----------------------------------------------------------------
    // DOM refs
    // -----------------------------------------------------------------
    const $ = (id) => document.getElementById(id);
    const els = {
        entryPanel: null, entryForm: null, secretInput: null, entrySubmit: null,
        gamePanel: null, board: null, secretDisplay: null, turnIndicator: null,
        confirmBar: null, confirmBtn: null, resetBtn: null,
        notebook: null, notebookDot: null,
        howToBtn: null, howToModal: null, themeToggle: null,
        endModal: null, endTitle: null, endAnswer: null, endBlurb: null, playAgainBtn: null,
        toast: null, year: null
    };

    // -----------------------------------------------------------------
    // Init
    // -----------------------------------------------------------------
    document.addEventListener('DOMContentLoaded', () => {
        for (const k of Object.keys(els)) els[k] = $(k);
        els.notebookDot = document.querySelector('.notebook-dot');
        if (els.year) els.year.textContent = new Date().getFullYear();

        renderBoard();
        bindEntry();
        bindGame();
        bindModals();
        bindTheme();
    });

    // -----------------------------------------------------------------
    // Phase 1 — word entry
    // -----------------------------------------------------------------
    function bindEntry() {
        // Force uppercase + strip non-letters as the user types
        els.secretInput.addEventListener('input', () => {
            const cleaned = els.secretInput.value
                .toUpperCase()
                .replace(/[^A-Z]/g, '')
                .slice(0, WORD_LEN);
            els.secretInput.value = cleaned;
        });

        els.entryForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const word = els.secretInput.value.trim().toUpperCase();
            if (!/^[A-Z]{5}$/.test(word)) {
                toast('Five letters, A–Z.');
                els.secretInput.focus();
                return;
            }
            startGame(word);
        });
    }

    function startGame(secret) {
        state.secret = secret;
        state.history = [];
        state.pending = null;
        state.thinking = false;
        state.gameOver = false;

        els.secretDisplay.textContent = secret;
        els.entryPanel.hidden = true;
        els.gamePanel.hidden = false;

        renderBoard();
        renderNotebook();
        renderTurn();

        requestNextGuess();
    }

    // -----------------------------------------------------------------
    // Phase 2 — game
    // -----------------------------------------------------------------
    function bindGame() {
        els.confirmBtn.addEventListener('click', () => {
            if (!state.pending || state.thinking || state.gameOver) return;
            confirmCurrentRow();
        });
        els.resetBtn.addEventListener('click', () => {
            if (!confirm('Pick a different word? This abandons the current run.')) return;
            resetToEntry();
        });
    }

    function resetToEntry() {
        state.secret = '';
        state.history = [];
        state.pending = null;
        state.thinking = false;
        state.gameOver = false;
        els.secretInput.value = '';
        els.gamePanel.hidden = true;
        els.entryPanel.hidden = false;
        els.endModal.hidden = true;
        els.secretInput.focus();
    }

    function renderBoard() {
        els.board.innerHTML = '';
        for (let r = 0; r < MAX_GUESSES; r++) {
            const row = document.createElement('div');
            row.className = 'row';
            row.dataset.row = String(r);
            for (let c = 0; c < WORD_LEN; c++) {
                const tile = document.createElement('button');
                tile.type = 'button';
                tile.className = 'tile';
                tile.dataset.col = String(c);
                tile.tabIndex = -1;
                tile.addEventListener('click', () => onTileClick(r, c));
                row.appendChild(tile);
            }
            els.board.appendChild(row);
        }
        paintBoard();
    }

    function paintBoard() {
        const rows = els.board.querySelectorAll('.row');
        rows.forEach((row, r) => {
            row.classList.remove('active', 'appearing');
            const tiles = row.querySelectorAll('.tile');

            if (r < state.history.length) {
                const h = state.history[r];
                tiles.forEach((tile, c) => {
                    tile.textContent = h.word[c];
                    setTileState(tile, h.colors[c], true);
                });
            } else if (r === state.history.length && state.pending) {
                row.classList.add('active');
                tiles.forEach((tile, c) => {
                    tile.textContent = state.pending.word[c];
                    setTileState(tile, state.pending.colors[c], true);
                });
            } else {
                tiles.forEach((tile) => {
                    tile.textContent = '';
                    setTileState(tile, null, false);
                });
            }
        });
    }

    function setTileState(tile, stateName, filled) {
        tile.classList.remove('state-correct', 'state-present', 'state-absent', 'filled');
        if (filled) tile.classList.add('filled');
        if (stateName) tile.classList.add('state-' + stateName);
    }

    function onTileClick(row, col) {
        if (!state.pending) return;
        if (row !== state.history.length) return;
        const cur = state.pending.colors[col];
        const idx = TILE_STATES.indexOf(cur);
        const next = TILE_STATES[(idx + 1) % TILE_STATES.length];
        state.pending.colors[col] = next;
        paintBoard();
    }

    function renderTurn() {
        const turn = state.history.length + (state.pending ? 1 : (state.thinking ? 1 : 0));
        const shown = Math.min(turn || 1, MAX_GUESSES);
        els.turnIndicator.textContent = `Guess ${shown} of ${MAX_GUESSES}`;
        els.confirmBar.hidden = !state.pending || state.gameOver;
        els.confirmBtn.disabled = !state.pending || state.thinking || state.gameOver;
    }

    // -----------------------------------------------------------------
    // Notebook rendering
    // -----------------------------------------------------------------
    function renderNotebook() {
        const body = els.notebook;
        body.innerHTML = '';

        if (!state.history.length && !state.pending && !state.thinking) {
            const p = document.createElement('p');
            p.className = 'notebook-placeholder';
            p.textContent = 'Waiting for the first guess…';
            body.appendChild(p);
            setNotebookDot(false);
            return;
        }

        // Render confirmed turns
        state.history.forEach((h, i) => {
            body.appendChild(noteEntry(i + 1, h.word, h.reasoning));
        });

        // Render pending turn (current guess waiting on confirmation)
        if (state.pending) {
            body.appendChild(noteEntry(
                state.history.length + 1,
                state.pending.word,
                state.pending.reasoning
            ));
        }

        // Render "thinking" placeholder if we're waiting on the model
        if (state.thinking) {
            const t = document.createElement('div');
            t.className = 'note-thinking';
            t.textContent = 'Watcher is thinking…';
            body.appendChild(t);
        }

        setNotebookDot(state.thinking);

        // Auto-scroll to bottom
        body.scrollTop = body.scrollHeight;
    }

    function noteEntry(turn, guess, reasoning) {
        const wrap = document.createElement('div');
        wrap.className = 'note-entry';

        const head = document.createElement('div');
        head.className = 'note-head';
        const t = document.createElement('span');
        t.className = 'note-turn';
        t.textContent = `Turn ${turn}`;
        const g = document.createElement('span');
        g.className = 'note-guess';
        g.textContent = guess;
        head.appendChild(t);
        head.appendChild(g);

        const r = document.createElement('div');
        r.className = 'note-reasoning';
        r.textContent = reasoning;

        wrap.appendChild(head);
        wrap.appendChild(r);
        return wrap;
    }

    function setNotebookDot(thinking) {
        if (!els.notebookDot) return;
        els.notebookDot.classList.toggle('thinking', !!thinking);
    }

    // -----------------------------------------------------------------
    // Turn flow
    // -----------------------------------------------------------------
    async function requestNextGuess() {
        if (state.gameOver) return;
        if (state.history.length >= MAX_GUESSES) return endGame(false);

        state.thinking = true;
        state.pending = null;
        renderNotebook();
        renderTurn();
        paintBoard();

        let result;
        try {
            result = await llmGenerate(state.history);
        } catch (err) {
            console.error('[watcher] generate failed:', err);
            state.thinking = false;
            renderNotebook();
            renderTurn();
            toast('Watcher couldn\'t come up with a guess. Try again.');
            return;
        }

        const guessWord = (result.guess || '').toUpperCase();
        if (!/^[A-Z]{5}$/.test(guessWord)) {
            console.error('[watcher] invalid guess:', guessWord);
            state.thinking = false;
            renderNotebook();
            renderTurn();
            toast('Bad guess from the model. Try again.');
            return;
        }

        // Pre-fill colours from the actual secret. The user can override.
        const colors = scoreGuess(state.secret, guessWord);

        state.thinking = false;
        state.pending = {
            word: guessWord,
            colors: colors,
            reasoning: result.reasoning || '',
            token: result.token
        };

        renderNotebook();
        renderTurn();
        paintBoard();

        // The "appearing" animation runs once on the new active row.
        const activeRow = els.board.querySelector('.row.active');
        if (activeRow) {
            activeRow.classList.add('appearing');
            setTimeout(() => activeRow.classList.remove('appearing'), 600);
        }
    }

    function confirmCurrentRow() {
        if (!state.pending) return;
        const turn = {
            word: state.pending.word,
            colors: state.pending.colors.slice(),
            reasoning: state.pending.reasoning,
            token: state.pending.token
        };
        state.history.push(turn);
        state.pending = null;

        paintBoard();
        renderNotebook();
        renderTurn();

        // Win condition: every tile correct
        const allCorrect = turn.colors.every((c) => c === 'correct');
        if (allCorrect) return endGame(true);
        if (state.history.length >= MAX_GUESSES) return endGame(false);

        requestNextGuess();
    }

    function endGame(won) {
        state.gameOver = true;
        state.pending = null;
        state.thinking = false;
        renderTurn();
        paintBoard();
        renderNotebook();

        els.endTitle.textContent = won ? 'Watcher cracked it.' : 'Watcher gave up.';
        els.endAnswer.textContent = state.secret;
        if (won) {
            const turns = state.history.length;
            els.endBlurb.textContent = `Solved in ${turns} ${turns === 1 ? 'guess' : 'guesses'}. Read back through the notebook to see how the model narrowed it down.`;
        } else {
            els.endBlurb.textContent = `Six guesses, no dice. The notebook still has the model's full reasoning — sometimes the wrong path is the most interesting one.`;
        }
        els.endModal.hidden = false;
    }

    // -----------------------------------------------------------------
    // Modals + theme + toast
    // -----------------------------------------------------------------
    function bindModals() {
        els.howToBtn.addEventListener('click', () => { els.howToModal.hidden = false; });
        els.playAgainBtn.addEventListener('click', () => {
            els.endModal.hidden = true;
            resetToEntry();
        });
        document.addEventListener('click', (e) => {
            const t = e.target;
            if (t && t.dataset && t.dataset.close === '1') {
                const modal = t.closest('.modal');
                if (modal) modal.hidden = true;
            }
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                els.howToModal.hidden = true;
                els.endModal.hidden = true;
            }
        });
    }

    function bindTheme() {
        els.themeToggle.addEventListener('click', () => {
            const root = document.documentElement;
            const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
            root.setAttribute('data-theme', next);
            localStorage.setItem('theme', next);
        });
    }

    let toastTimer = null;
    function toast(msg) {
        els.toast.textContent = msg;
        els.toast.hidden = false;
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => { els.toast.hidden = true; }, 2200);
    }

})();
