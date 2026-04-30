import { defineConfig } from "vite"

export default defineConfig({
  publicDir: false,
  build: {
    emptyOutDir: true,
    outDir: "dist-package/dist",
    lib: {
      entry: "src/shot-boundary.ts",
      formats: ["es"],
      fileName: () => "shot-boundary.js",
    },
    rollupOptions: {
      external: ["mediabunny", "onnxruntime-web"],
    },
  },
})
