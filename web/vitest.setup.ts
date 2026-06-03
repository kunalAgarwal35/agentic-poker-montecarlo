import '@testing-library/jest-dom/vitest';

// Minimal IntersectionObserver polyfill so components that observe a sentinel
// (e.g. infinite-scroll galleries) don't crash under the test DOM.
if (typeof globalThis.IntersectionObserver === 'undefined') {
  class MockIntersectionObserver {
    constructor() {}
    observe() {}
    unobserve() {}
    disconnect() {}
    takeRecords() {
      return [];
    }
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  globalThis.IntersectionObserver = MockIntersectionObserver as any;
}
