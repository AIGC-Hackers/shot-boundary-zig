import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"

const here = dirname(fileURLToPath(import.meta.url))
const webRoot = join(here, "..")
const packageRoot = join(webRoot, "dist-package")
const sourcePackage = JSON.parse(
  await Bun.file(join(webRoot, "package.json")).text()
)

copyRuntimeAssets()
copyModelAssets()
writePackageJson()
copyFileSync(join(webRoot, "README.md"), join(packageRoot, "README.md"))

function writePackageJson() {
  const packageJson = {
    name: sourcePackage.name,
    version: sourcePackage.version,
    type: "module",
    description:
      "Browser shot-boundary detection helpers for ONNX Runtime Web.",
    packageManager: sourcePackage.packageManager,
    exports: {
      ".": {
        types: "./dist/shot-boundary.d.ts",
        import: "./dist/shot-boundary.js",
      },
    },
    files: ["dist", "assets", "README.md"],
    dependencies: {
      mediabunny: sourcePackage.dependencies.mediabunny,
      "onnxruntime-web": sourcePackage.dependencies["onnxruntime-web"],
    },
  }

  writeFileSync(
    join(packageRoot, "package.json"),
    `${JSON.stringify(packageJson, null, 2)}\n`
  )
}

function copyRuntimeAssets() {
  const sourceDir = join(webRoot, "node_modules", "onnxruntime-web", "dist")
  const outputDir = join(packageRoot, "assets", "ort-wasm")

  rmSync(outputDir, { force: true, recursive: true })
  mkdirSync(outputDir, { recursive: true })

  if (!existsSync(sourceDir)) {
    throw new Error(
      "onnxruntime-web dist directory is missing. Run bun install first."
    )
  }

  for (const file of readdirSync(sourceDir)) {
    if (
      file.startsWith("ort-wasm") &&
      (file.endsWith(".wasm") || file.endsWith(".mjs"))
    ) {
      copyFileSync(join(sourceDir, file), join(outputDir, file))
    }
  }
}

function copyModelAssets() {
  const modelPath = join(webRoot, "..", "models", "transnetv2.onnx")
  const outputDir = join(packageRoot, "assets", "models")
  const outputPath = join(outputDir, "transnetv2.onnx")

  rmSync(outputDir, { force: true, recursive: true })
  mkdirSync(outputDir, { recursive: true })

  if (!existsSync(modelPath)) {
    throw new Error("models/transnetv2.onnx is missing.")
  }

  const modelSize = statSync(modelPath).size
  if (modelSize < 20_000_000) {
    throw new Error(
      "models/transnetv2.onnx looks like a Git LFS pointer. Fetch LFS assets before packaging."
    )
  }

  copyFileSync(modelPath, outputPath)
}
