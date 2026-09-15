import { defineConfig } from 'vite';

export default defineConfig({
  base: '/Townscaper/',
  resolve: {
    // Addons import 'three'; point that at the WebGPU build so the app and
    // addons share one copy of the library.
    alias: [{ find: /^three$/, replacement: 'three/webgpu' }],
  },
});
