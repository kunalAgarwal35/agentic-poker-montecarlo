// A stable, anonymous session id stored in localStorage. Ties a visitor's
// feedback to a session without any login. SSR-safe: returns 'server' when
// there's no window/localStorage.
const KEY = 'apm_session_id';

function newId(): string {
  // crypto.randomUUID isn't available in every runtime/test env — fall back.
  try {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
      return crypto.randomUUID();
    }
  } catch {
    // ignore and fall through to the cheap fallback
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

export function getSessionId(): string {
  if (typeof window === 'undefined') return 'server';
  try {
    const existing = window.localStorage.getItem(KEY);
    if (existing) return existing;
    const id = newId();
    window.localStorage.setItem(KEY, id);
    return id;
  } catch {
    // localStorage can throw (private mode, blocked cookies) — degrade.
    return newId();
  }
}
