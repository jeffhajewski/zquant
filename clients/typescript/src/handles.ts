/**
 * Native handle ownership.
 *
 * `close()` is the way to free an index or codec, and it frees at once. The
 * FinalizationRegistry here is the safety net for one that is dropped without being
 * closed: when the garbage collector reclaims the wrapper, the native memory goes too,
 * rather than leaking until the process exits. Collection timing is up to the engine,
 * so code that creates many indexes should still close them.
 *
 * Each wrapper's native handles live in a small state object that the registry holds,
 * and that object must never refer back to the wrapper, or the wrapper could never be
 * collected. Searchers sit in their index's state, so they are always freed before the
 * index they belong to, whichever way it goes.
 */
import { native } from './native.js';

/** Handles currently allocated, by kind. For tests and leak hunting. */
export const live = { index: 0, searcher: 0, codec: 0 };

export interface IndexState {
  handle: unknown;
  /** Searchers created from this index, keyed by their (k, threads) shape. */
  searchers: Map<string, unknown>;
}

export interface CodecState {
  handle: unknown;
}

function freeIndex(s: IndexState): void {
  if (s.handle === null) return;
  const n = native();
  for (const h of s.searchers.values()) {
    n.searcherFree(h);
    live.searcher--;
  }
  s.searchers.clear();
  n.indexFree(s.handle);
  s.handle = null;
  live.index--;
}

function freeCodec(s: CodecState): void {
  if (s.handle === null) return;
  native().codecFree(s.handle);
  s.handle = null;
  live.codec--;
}

const indexes = new FinalizationRegistry<IndexState>(freeIndex);
const codecs = new FinalizationRegistry<CodecState>(freeCodec);

/** Start tracking a freshly created index handle on behalf of `owner`. */
export function adoptIndex(owner: object, handle: unknown): IndexState {
  const state: IndexState = { handle, searchers: new Map() };
  live.index++;
  indexes.register(owner, state, state);
  return state;
}

/** Record a searcher created from this index, so it is freed with it. */
export function addSearcher(state: IndexState, key: string, handle: unknown): void {
  state.searchers.set(key, handle);
  live.searcher++;
}

/** Free an index and its searchers now. Safe to call more than once. */
export function releaseIndex(state: IndexState): void {
  indexes.unregister(state);
  freeIndex(state);
}

export function adoptCodec(owner: object, handle: unknown): CodecState {
  const state: CodecState = { handle };
  live.codec++;
  codecs.register(owner, state, state);
  return state;
}

export function releaseCodec(state: CodecState): void {
  codecs.unregister(state);
  freeCodec(state);
}
