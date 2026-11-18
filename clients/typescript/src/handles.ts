/**
 * Native handle ownership.
 *
 * Each wrapper's native handles live in a small state object. Searchers sit in their
 * index's state, so they are always freed before the index they belong to, and a
 * searcher can only ever be used with the index it was created from.
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

/** Start tracking a freshly created index handle on behalf of `owner`. */
export function adoptIndex(_owner: object, handle: unknown): IndexState {
  const state: IndexState = { handle, searchers: new Map() };
  live.index++;
  return state;
}

/** Record a searcher created from this index, so it is freed with it. */
export function addSearcher(state: IndexState, key: string, handle: unknown): void {
  state.searchers.set(key, handle);
  live.searcher++;
}

/** Free an index and its searchers now. Safe to call more than once. */
export function releaseIndex(state: IndexState): void {
  freeIndex(state);
}

export function adoptCodec(_owner: object, handle: unknown): CodecState {
  const state: CodecState = { handle };
  live.codec++;
  return state;
}

export function releaseCodec(state: CodecState): void {
  freeCodec(state);
}
