// photo-geolocator-01 entry: a vision-model geolocator, one call per photo.
// For each of the instance's photos it asks claude-sonnet-5-5 where the photo was
// taken (first naming the visible clues, then giving lat/lon as JSON), reads the
// coordinates out of the answer, and posts every answer it read to the env at
// the end. A photo whose model call the gateway refuses (no key pushed, budget
// spent) or whose answer holds no coordinates after one retry is answered with
// the fallback point below instead of being left unanswered, because an
// unanswered photo counts the maximum error. Nothing else is guessed from.
//
// It runs as the grader runs an entry, `node /bot/main.ts` on Node 22 (erasable
// TypeScript, the standard library only), with the trial's variables:
// MAKEFUN_ENV_URL (the env), ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY (the
// gateway and the dummy key it swaps for the real one).
//
// Whose key pays: every call runs on the Anthropic key pushed to the competition
// before submitting (ANTHROPIC_API_KEY=<your key> in a .env file, then
// `makefun secrets push photo-geolocator-01 --env-file .env`, or Profile > API
// keys): the house pays for none of an entry's model calls. With no Anthropic
// key pushed every call is answered 401 and every photo takes the fallback.
//
// Deterministic given the model's answers: every request is a function of the
// photo's bytes and the fixed prompt, the calls run four at a time and their
// answers are posted in photo order, so a replay sends the same calls and
// leaves the same answers. A photo whose answer holds no coordinates after the
// retry takes the fallback, and so does one whose model call the gateway
// refuses (its budget, its request cap: a refusal is final); every photo gets
// its one call (plus a retry only when the first answer held no coordinates)
// whatever the others' answers, so a replay makes the same calls whatever
// order they come back in.

const MODEL = 'claude-sonnet-5-5';
const MAX_TOKENS = 512;
const CONCURRENCY = 4;
const PROMPT =
  'Look at this street-level photo and work out where it was taken. Use the visible clues: ' +
  'the country and language on signs, the side of the road traffic drives on, road markings, ' +
  'license plates, architecture, vegetation and climate, and any landmark. ' +
  'Then give your best estimate as one line of JSON: {"lat": <degrees north>, "lon": <degrees east>}. ' +
  'Give coordinates for the specific street or area shown, not just the country centre.';
// Fallback point: the point minimising total haversine error over the 75
// practice photos (a European-skewed street-photo distribution). It is far
// worse than a vision answer, but far better than leaving a photo unanswered
// (20,015 km). Measured on the practice set: mean error 6,926 km vs 20,015 km.
const FALLBACK_LAT = 56.0678;
const FALLBACK_LON = 9.9332;
/** Retried with a fixed backoff: a provider's transient errors. A gateway refusal (4xx) is final. */
const RETRY_STATUSES = [408, 429, 500, 502, 503, 504, 529];
const RETRIES = 3;

interface Photo {
  id: string;
  url: string;
  answered: boolean;
}

interface Guess {
  id: string;
  lat: number;
  lon: number;
}

function need(name: string): string {
  const v = process.env[name];
  if (v === undefined || v === '') throw new Error(`${name} is not set`);
  return v;
}

const envUrl = need('MAKEFUN_ENV_URL').replace(/\/+$/, '');
const anthropicUrl = need('ANTHROPIC_BASE_URL').replace(/\/+$/, '');
const apiKey = need('ANTHROPIC_API_KEY');

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

/** The model's answer to one photo, as text; null when the gateway refused the call (final).
 * A 401 means no provider key is behind the gateway: permanent for the trial,
 * so the caller stops calling the model and falls back for the rest. */
let noKey = false;

async function ask(jpeg: Buffer): Promise<string | null> {
  const body = JSON.stringify({
    model: MODEL,
    max_tokens: MAX_TOKENS,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: jpeg.toString('base64') } },
          { type: 'text', text: PROMPT },
        ],
      },
    ],
  });
  for (let attempt = 0; ; attempt += 1) {
    const res = await fetch(`${anthropicUrl}/v1/messages`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': apiKey, 'anthropic-version': '2023-06-01' },
      body,
    });
    if (res.ok) {
      const msg = (await res.json()) as { content?: { type: string; text?: string }[] };
      return (msg.content ?? [])
        .filter((b) => b.type === 'text')
        .map((b) => b.text ?? '')
        .join('\n');
    }
    const text = await res.text();
    if (res.status === 401) {
      noKey = true;
      process.stderr.write(`photo: the model call was refused (HTTP 401, no provider key): falling back for the rest\n`);
      return null;
    }
    if (RETRY_STATUSES.includes(res.status) && res.headers.get('x-should-retry') !== 'false' && attempt < RETRIES) {
      await sleep(2000 * (attempt + 1));
      continue;
    }
    process.stderr.write(`photo: the model call was refused (HTTP ${res.status}): ${text.slice(0, 300)}\n`);
    return null;
  }
}

const isLat = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= -90 && v <= 90;
const isLon = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= -180 && v <= 180;

/** The latitude and longitude an answer gives: its first JSON object with lat and lon, else its first two numbers in range. */
function coordinates(text: string): { lat: number; lon: number } | null {
  for (const m of text.matchAll(/\{[^{}]*\}/g)) {
    try {
      const o = JSON.parse(m[0]) as Record<string, unknown>;
      const lat = o.lat ?? o.latitude;
      const lon = o.lon ?? o.lng ?? o.longitude;
      if (isLat(lat) && isLon(lon)) return { lat, lon };
    } catch {
      // Not JSON: the next one, or the numbers below.
    }
  }
  const nums = [...text.matchAll(/-?\d+(?:\.\d+)?/g)].map((m) => Number(m[0]));
  for (let i = 0; i + 1 < nums.length; i += 1) {
    if (isLat(nums[i]) && isLon(nums[i + 1])) return { lat: nums[i]!, lon: nums[i + 1]! };
  }
  return null;
}

async function main(): Promise<number> {
  const listed = (await (await fetch(`${envUrl}/photos`)).json()) as { photos: Photo[] };
  const photos = listed.photos.filter((p) => !p.answered);
  const guesses: (Guess | null)[] = new Array(photos.length).fill(null);
  let refused = 0;
  let next = 0;
  const worker = async (): Promise<void> => {
    while (next < photos.length) {
      const i = next;
      next += 1;
      const p = photos[i]!;
      const jpeg = Buffer.from(await (await fetch(`${envUrl}${p.url}`)).arrayBuffer());
      if (noKey) {
        // No provider key behind the gateway: no more model calls, fallback.
        refused += 1;
        guesses[i] = { id: p.id, lat: FALLBACK_LAT, lon: FALLBACK_LON };
        continue;
      }
      const text = await ask(jpeg);
      if (text === null) {
        // Gateway refusal (no key, budget spent, request cap): fallback, not unanswered.
        refused += 1;
        guesses[i] = { id: p.id, lat: FALLBACK_LAT, lon: FALLBACK_LON };
        continue;
      }
      let c = coordinates(text);
      if (c === null) {
        // One retry when the first answer held no coordinates; the retry is a
        // function of the same photo, so a replay makes the same calls.
        const text2 = await ask(jpeg);
        if (text2 !== null) c = coordinates(text2);
        if (c === null) process.stderr.write(`photo: no coordinates in the answer for ${p.id}, fallback used\n`);
      }
      if (c === null) guesses[i] = { id: p.id, lat: FALLBACK_LAT, lon: FALLBACK_LON };
      else guesses[i] = { id: p.id, lat: c.lat, lon: c.lon };
    }
  };
  await Promise.all(Array.from({ length: CONCURRENCY }, () => worker()));
  const answers = guesses.filter((g): g is Guess => g !== null);
  if (answers.length === 0) return refused > 0 ? 1 : 0;
  const res = await fetch(`${envUrl}/answers`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ answers }) });
  process.stdout.write(`photo: ${answers.length} of ${photos.length} photos answered (${refused} fallback): HTTP ${res.status} ${await res.text()}`);
  return res.ok ? 0 : 1;
}

try {
  process.exitCode = await main();
} catch (err) {
  process.stderr.write(`photo: ${(err as Error).stack ?? String(err)}\n`);
  process.exitCode = 1;
}
