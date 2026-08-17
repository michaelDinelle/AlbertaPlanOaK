export {};

declare global {
  interface GlassOptions {
    scale?: number;
    chroma?: number;
    border?: number;
    mapBlur?: number;
    blur?: number;
    saturate?: number;
    radius?: number;
    fallbackBlur?: number;
  }

  interface GlassHandle {
    supported: boolean;
    refresh(): void;
    destroy(): void;
  }

  interface Window {
    liquidGlass(el: Element, opts?: GlassOptions): GlassHandle;
  }
}
