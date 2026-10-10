// plotly.js-cartesian-dist-min ships no type definitions. Only the three functions the
// Plot wrapper calls are declared here.
declare module 'plotly.js-cartesian-dist-min' {
  export function newPlot(
    root: HTMLElement,
    data: unknown[],
    layout?: unknown,
    config?: unknown,
  ): Promise<HTMLElement>
  export function react(
    root: HTMLElement,
    data: unknown[],
    layout?: unknown,
    config?: unknown,
  ): Promise<HTMLElement>
  export function purge(root: HTMLElement): void
}
