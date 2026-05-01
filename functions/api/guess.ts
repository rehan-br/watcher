/**
 * POST /api/guess
 *
 * The browser sends the model's prior guesses + the tile colours the user
 * confirmed. The server turns that into a prompt, asks the configured LLM
 * for the next guess + reasoning, validates the JSON, and returns it.
 *
 * The secret word is NEVER sent. The model can only reason from its own
 * past guesses and the colours coming back.
 *
 * Provider choice is driven by the LLM_PROVIDER env var:
 *   - "gemini"        → Gemini 2.5 Flash via Google AI Studio (default)
 *   - "cloudflare"    → Workers AI (Llama 3.3 70B fp8-fast)
 *
 * Adding a new provider is a single new branch in `generate()`.
 *
 * ANTI-CHEAT
 * ----------
 * Because the secret stays client-side, full server-authoritative scoring
 * isn't possible. Instead the server enforces two things:
 *
 *   1. HMAC token chain. Every guess we generate is signed with
 *      WATCHER_HMAC_SECRET, binding it to the prior history snapshot.
 *      The client echoes tokens back on every turn; we validate the chain.
 *      Result: the client cannot forge a turn that "we generated" — every
 *      guess in the history must have actually come from /api/guess.
 *
 *   2. History consistency. We reject color histories that are internally
 *      contradictory (e.g. position 3 marked "correct=A" in turn 1 and
 *      "correct=B" in turn 2; or the same word repeated with different
 *      colours). This catches casual color-tampering through dev tools.
 *
 * What we explicitly cannot prevent: a user lying about the colours of
 * their MOST RECENT turn. There's no way to detect that without the
 * secret. CLAUDE.md is fine with this — the lab is honest by design, not
 * by enforcement, and the marketing value is in honest play anyway.
 */

type TileColor = 'correct' | 'present' | 'absent';

interface HistoryTurn {
    word: string;          // model's prior guess, 5 letters, uppercase
    colors: TileColor[];   // user-confirmed tile colours
    token: string;         // HMAC issued by /api/guess when this guess was generated
}

interface RequestBody {
    word_length: number;
    history: HistoryTurn[];
}

interface LlmResponse {
    reasoning: string;
    guess: string;
}

interface Env {
    LLM_PROVIDER?: string;
    GEMINI_API_KEY?: string;
    GEMINI_MODEL?: string;
    WATCHER_HMAC_SECRET?: string;
    AI?: {
        run: (model: string, input: unknown) => Promise<unknown>;
    };
}

const DEFAULT_GEMINI_MODEL = 'gemini-2.5-flash';
const DEFAULT_CF_MODEL = '@cf/meta/llama-3.3-70b-instruct-fp8-fast';
const MAX_TURNS = 6;
const WORD_LEN = 5;

const SYSTEM_PROMPT = `You are playing Wordle in reverse. A human has chosen a secret 5-letter English word. You are guessing.

Tile colour meanings:
  - "correct": that letter is in the secret AND in that exact position.
  - "present": that letter is in the secret BUT in a different position.
  - "absent":  that letter is NOT in the secret at that position. (Note: if a letter appears more than once in your guess, an "absent" tile only rules out *that occurrence* — the letter may still appear elsewhere if other tiles for the same letter are correct/present.)

Each turn you must:
  1. Reason out loud about what you've learned from prior turns.
  2. Pick the next 5-letter guess. It MUST be a real common English word, exactly 5 letters, A-Z only.
  3. Avoid repeating any prior guess.
  4. On turn 1, prefer a high-information opener that covers common letters (e.g. STARE, CRANE, SLATE).

Output STRICT JSON with exactly two keys:
  - "reasoning": 2-4 sentences of natural language. Reference specific letters and what you've ruled in / out. Be concise; this text is shown live to the user.
  - "guess": one 5-letter word in UPPERCASE.

Do not output anything outside the JSON object. No markdown fences, no preamble, no trailing commentary.`;

function buildUserPrompt(history: HistoryTurn[]): string {
    if (history.length === 0) {
        return 'No guesses yet. Make your opening guess.';
    }
    const lines: string[] = ['Prior turns:'];
    history.forEach((h, i) => {
        const tiles = h.word
            .split('')
            .map((ch, idx) => `${ch}=${h.colors[idx] || 'absent'}`)
            .join(', ');
        lines.push(`Turn ${i + 1}: guessed ${h.word} → ${tiles}`);
    });
    lines.push('');
    lines.push('Make your next guess.');
    return lines.join('\n');
}

// ---------------------------------------------------------------------
// Gemini
// ---------------------------------------------------------------------
async function generateWithGemini(
    history: HistoryTurn[],
    env: Env,
    strict: boolean
): Promise<LlmResponse> {
    if (!env.GEMINI_API_KEY) {
        throw new Error('GEMINI_API_KEY not configured');
    }
    const model = env.GEMINI_MODEL || DEFAULT_GEMINI_MODEL;
    const url =
        `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent` +
        `?key=${encodeURIComponent(env.GEMINI_API_KEY)}`;

    const sys = strict
        ? SYSTEM_PROMPT +
          '\n\nIMPORTANT: your previous reply could not be parsed as JSON. Output ONLY a valid JSON object with keys "reasoning" and "guess". No fences, no extra text.'
        : SYSTEM_PROMPT;

    const body = {
        contents: [
            { role: 'user', parts: [{ text: buildUserPrompt(history) }] }
        ],
        systemInstruction: { parts: [{ text: sys }] },
        generationConfig: {
            responseMimeType: 'application/json',
            responseSchema: {
                type: 'OBJECT',
                properties: {
                    reasoning: { type: 'STRING' },
                    guess: { type: 'STRING' }
                },
                required: ['reasoning', 'guess']
            },
            temperature: 0.7,
            maxOutputTokens: 512
        }
    };

    const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    if (!res.ok) {
        const txt = await res.text().catch(() => '');
        throw new Error(`gemini ${res.status}: ${txt.slice(0, 200)}`);
    }
    const data = (await res.json()) as {
        candidates?: { content?: { parts?: { text?: string }[] } }[];
    };
    const text = data?.candidates?.[0]?.content?.parts?.[0]?.text || '';
    return parseJsonResponse(text);
}

// ---------------------------------------------------------------------
// Cloudflare Workers AI
// ---------------------------------------------------------------------
async function generateWithCloudflare(
    history: HistoryTurn[],
    env: Env,
    strict: boolean
): Promise<LlmResponse> {
    if (!env.AI) throw new Error('Workers AI binding (AI) not configured');
    const sys = strict
        ? SYSTEM_PROMPT +
          '\n\nIMPORTANT: your previous reply could not be parsed as JSON. Output ONLY a valid JSON object with keys "reasoning" and "guess". No fences, no extra text.'
        : SYSTEM_PROMPT;

    const out = (await env.AI.run(DEFAULT_CF_MODEL, {
        messages: [
            { role: 'system', content: sys },
            { role: 'user', content: buildUserPrompt(history) }
        ],
        response_format: { type: 'json_object' },
        max_tokens: 512,
        temperature: 0.7
    })) as { response?: string } | string;

    const text = typeof out === 'string' ? out : out?.response || '';
    return parseJsonResponse(text);
}

// ---------------------------------------------------------------------
// Provider dispatch
// ---------------------------------------------------------------------
async function generate(
    history: HistoryTurn[],
    env: Env,
    strict = false
): Promise<LlmResponse> {
    const provider = (env.LLM_PROVIDER || 'gemini').toLowerCase();
    if (provider === 'cloudflare' || provider === 'workers-ai') {
        return generateWithCloudflare(history, env, strict);
    }
    return generateWithGemini(history, env, strict);
}

// ---------------------------------------------------------------------
// JSON parsing — tolerant of stray text / fenced code blocks
// ---------------------------------------------------------------------
function parseJsonResponse(raw: string): LlmResponse {
    if (!raw) throw new Error('empty model response');
    let txt = raw.trim();
    txt = txt.replace(/^```(?:json)?\s*/i, '').replace(/```\s*$/, '').trim();
    if (!txt.startsWith('{')) {
        const start = txt.indexOf('{');
        const end = txt.lastIndexOf('}');
        if (start >= 0 && end > start) txt = txt.slice(start, end + 1);
    }
    let parsed: unknown;
    try {
        parsed = JSON.parse(txt);
    } catch {
        throw new Error('invalid JSON from model');
    }
    if (
        !parsed ||
        typeof parsed !== 'object' ||
        typeof (parsed as LlmResponse).reasoning !== 'string' ||
        typeof (parsed as LlmResponse).guess !== 'string'
    ) {
        throw new Error('JSON missing required keys');
    }
    const out = parsed as LlmResponse;
    const guess = out.guess.toUpperCase().replace(/[^A-Z]/g, '');
    if (guess.length !== WORD_LEN) throw new Error('guess is not exactly 5 A-Z letters');
    return { reasoning: out.reasoning.trim(), guess };
}

function defensiveFallback(history: HistoryTurn[]): { reasoning: string; guess: string } {
    const tried = new Set(history.map((h) => h.word.toUpperCase()));
    const fallback = ['ABOUT', 'CRANE', 'SLATE', 'POINT', 'GHOST', 'FROST'];
    const pick = fallback.find((w) => !tried.has(w)) || 'ABOUT';
    return {
        reasoning:
            "I couldn't shape my reply into valid JSON twice in a row. Falling back to a safe common word so the run can continue. Re-prompt me and I should recover.",
        guess: pick
    };
}

// ---------------------------------------------------------------------
// HMAC token chain — prevents the client from injecting fake turns.
// Token_k = HMAC(canonical_json(prior_history_with_colors) + "|" + word_k)
// where prior_history_with_colors is history[0..k-1].
//
// On a request with N prior turns, the server validates ALL N tokens
// against the colors the client is currently sending. Mutating any prior
// color or any prior word breaks the chain.
// ---------------------------------------------------------------------
async function hmacSha256Hex(secret: string, data: string): Promise<string> {
    const key = await crypto.subtle.importKey(
        'raw',
        new TextEncoder().encode(secret),
        { name: 'HMAC', hash: 'SHA-256' },
        false,
        ['sign']
    );
    const sig = await crypto.subtle.sign('HMAC', key, new TextEncoder().encode(data));
    const arr = new Uint8Array(sig);
    let hex = '';
    for (let i = 0; i < arr.length; i++) hex += arr[i].toString(16).padStart(2, '0');
    return hex;
}

function constantTimeEqual(a: string, b: string): boolean {
    if (a.length !== b.length) return false;
    let diff = 0;
    for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
    return diff === 0;
}

function canonicalize(prior: HistoryTurn[]): string {
    return JSON.stringify(
        prior.map((h) => ({ word: h.word, colors: h.colors }))
    );
}

async function makeToken(
    secret: string,
    prior: HistoryTurn[],
    word: string
): Promise<string> {
    return hmacSha256Hex(secret, canonicalize(prior) + '|' + word);
}

async function validateTokenChain(
    secret: string,
    history: HistoryTurn[]
): Promise<void> {
    for (let k = 0; k < history.length; k++) {
        const prior = history.slice(0, k);
        const expected = await makeToken(secret, prior, history[k].word);
        if (!constantTimeEqual(expected, history[k].token || '')) {
            throw new Error(`token mismatch at turn ${k + 1}`);
        }
    }
}

// ---------------------------------------------------------------------
// History consistency — catches casual color tampering.
// Rejects:
//   1. The same word repeated with different colour patterns.
//   2. Two turns claiming "correct" at the same position with different
//      letters (impossible — that position only has one letter).
//   3. Two turns claiming the same letter at the same position with
//      different colours (deterministic scoring would always yield the
//      same colour for that position).
// ---------------------------------------------------------------------
function validateHistoryConsistency(history: HistoryTurn[]): void {
    if (history.length > MAX_TURNS) {
        throw new Error(`history exceeds ${MAX_TURNS} turns`);
    }

    const positionLetter: Record<number, string> = {}; // position → known letter via "correct"
    const seenWord: Record<string, TileColor[]> = {};

    for (let k = 0; k < history.length; k++) {
        const h = history[k];

        // Same-word check
        const prior = seenWord[h.word];
        if (prior) {
            for (let i = 0; i < WORD_LEN; i++) {
                if (prior[i] !== h.colors[i]) {
                    throw new Error(`turn ${k + 1}: ${h.word} repeated with different colors`);
                }
            }
        } else {
            seenWord[h.word] = h.colors.slice();
        }

        // Position checks
        for (let i = 0; i < WORD_LEN; i++) {
            const ch = h.word[i];
            const c = h.colors[i];
            if (c === 'correct') {
                const prev = positionLetter[i];
                if (prev !== undefined && prev !== ch) {
                    throw new Error(
                        `turn ${k + 1}: position ${i + 1} marked correct=${ch} but prior turn locked it to ${prev}`
                    );
                }
                positionLetter[i] = ch;
            } else {
                // If we already know position i is letter X via a prior "correct",
                // then this turn putting X at position i must also be "correct".
                if (positionLetter[i] !== undefined && positionLetter[i] === ch) {
                    throw new Error(
                        `turn ${k + 1}: position ${i + 1} has ${ch} but was previously locked correct=${ch}`
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// Pages Function handler
// ---------------------------------------------------------------------
function jsonResponse(obj: unknown, status = 200): Response {
    return new Response(JSON.stringify(obj), {
        status,
        headers: {
            'Content-Type': 'application/json; charset=utf-8',
            'Cache-Control': 'no-store'
        }
    });
}

function validateBody(body: unknown): RequestBody {
    if (!body || typeof body !== 'object') {
        throw new Error('body must be an object');
    }
    const b = body as Partial<RequestBody>;
    if (b.word_length !== WORD_LEN) throw new Error(`word_length must be ${WORD_LEN}`);
    if (!Array.isArray(b.history)) throw new Error('history must be an array');
    if (b.history.length > MAX_TURNS) throw new Error('history too long');
    const history: HistoryTurn[] = b.history.map((h, i) => {
        if (!h || typeof h !== 'object') throw new Error(`history[${i}] not an object`);
        const word = String((h as HistoryTurn).word || '').toUpperCase();
        if (!/^[A-Z]{5}$/.test(word)) {
            throw new Error(`history[${i}].word must be 5 letters`);
        }
        const colors = (h as HistoryTurn).colors;
        if (!Array.isArray(colors) || colors.length !== WORD_LEN) {
            throw new Error(`history[${i}].colors must be array of ${WORD_LEN}`);
        }
        for (const c of colors) {
            if (c !== 'correct' && c !== 'present' && c !== 'absent') {
                throw new Error(`history[${i}].colors invalid value`);
            }
        }
        const token = String((h as HistoryTurn).token || '');
        // Token must be a 64-char hex string (HMAC-SHA256 hex). On turn 0
        // history is empty so this branch never fires.
        if (!/^[0-9a-f]{64}$/.test(token)) {
            throw new Error(`history[${i}].token missing or malformed`);
        }
        return { word, colors: colors.slice() as TileColor[], token };
    });
    return { word_length: WORD_LEN, history };
}

export const onRequestPost: PagesFunction<Env> = async (ctx) => {
    if (!ctx.env.WATCHER_HMAC_SECRET) {
        // Without an HMAC secret we cannot honestly run the chain.
        // Fail loud so deployment doesn't silently lose anti-cheat.
        return jsonResponse(
            { error: 'server misconfigured: WATCHER_HMAC_SECRET not set' },
            500
        );
    }

    let body: RequestBody;
    try {
        const raw = await ctx.request.json().catch(() => null);
        body = validateBody(raw);
    } catch (err) {
        return jsonResponse(
            { error: err instanceof Error ? err.message : 'bad request' },
            400
        );
    }

    // Anti-cheat: history must be internally consistent and token chain
    // must validate against our HMAC secret.
    try {
        validateHistoryConsistency(body.history);
        await validateTokenChain(ctx.env.WATCHER_HMAC_SECRET, body.history);
    } catch (err) {
        return jsonResponse(
            {
                error: 'history rejected',
                detail: err instanceof Error ? err.message : 'unknown'
            },
            400
        );
    }

    // Done with anti-cheat. Now ask the model.
    let result: LlmResponse;
    try {
        result = await generate(body.history, ctx.env, false);
    } catch (firstErr) {
        console.warn('[watcher] first generate failed:', firstErr);
        try {
            result = await generate(body.history, ctx.env, true);
        } catch (secondErr) {
            console.error('[watcher] second generate failed:', secondErr);
            result = defensiveFallback(body.history);
        }
    }

    const newToken = await makeToken(
        ctx.env.WATCHER_HMAC_SECRET,
        body.history,
        result.guess
    );

    return jsonResponse({
        reasoning: result.reasoning,
        guess: result.guess,
        token: newToken
    });
};

export const onRequest: PagesFunction<Env> = async (ctx) => {
    if (ctx.request.method !== 'POST') {
        return jsonResponse({ error: 'POST only' }, 405);
    }
    return onRequestPost(ctx);
};
