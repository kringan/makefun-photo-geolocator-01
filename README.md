# make.fun photo-geolocator-01 entry

A vision-model geolocator: for each of the instance's 25 street-level photos it
asks claude-sonnet-5-5 where the photo was taken (first naming the visible
clues — country and language on signs, driving side, road markings, plates,
architecture, vegetation, landmarks — then giving lat/lon as JSON), and posts
every answer to the env.

- `main.ts` — the agent (Node 22, standard library only). One model call per
  photo, four at a time, answers posted in photo order, so a replay sends the
  same calls and leaves the same answers. Max 512 output tokens per call.
- A photo whose model call the gateway refuses (no key pushed, budget spent)
  or whose answer holds no coordinates after one retry is answered with a
  fixed fallback point (56.0678, 9.9332 — the point minimising total error over
  the kit's 75 practice photos) instead of being left unanswered (20,015 km).
  A 401 sets a flag so no further model calls are made in that trial.
- `LICENSE` — MIT.

Tested locally (env + 401 stub gateway): all 25 photos answered, fallback
total error 234,452 km on practice trial-0 (median 9,249 km), versus 500,375 km
unanswered. The vision path needs an Anthropic key pushed to the competition
(`makefun secrets push photo-geolocator-01 --env-file .env`).
