/**
 * Loads the distilled rock SDF networks exported by training/neural_sdf/export.py
 * and packs every species into one weight array for a GPU storage buffer.
 */

const SUPPORTED_FORMAT_VERSION = 1;

export type Activation = 'relu' | 'silu';
const ACTIVATIONS: readonly Activation[] = ['relu', 'silu'];

/** Network shape shared by all species; it specializes the WGSL evaluator. */
export interface MlpArchitecture {
  readonly numFrequencies: number;
  readonly inputDim: number;
  readonly hiddenWidth: number;
  readonly hiddenLayers: number;
  readonly activation: Activation;
  readonly floatsPerModel: number;
}

export interface RockModels {
  readonly architecture: MlpArchitecture;
  /** All species back to back; species `s` starts at `s * floatsPerModel`. */
  readonly weights: Float32Array;
}

interface LayerManifest {
  in_features: number;
  out_features: number;
  weight_offset: number;
  bias_offset: number;
}

interface ModelManifest {
  format_version: number;
  encoding: {type: string; num_frequencies: number};
  activation: string;
  weights: {file: string; dtype: string; byte_order: string; count: number};
  layers: LayerManifest[];
}

/** Used to compile the shader when no models are available. */
export const FALLBACK_ARCHITECTURE: MlpArchitecture = {
  numFrequencies: 6,
  inputDim: 39,
  hiddenWidth: 64,
  hiddenLayers: 2,
  activation: 'silu',
  floatsPerModel: 6785,
};

export async function loadRockModels(
  baseUrl: string,
  speciesCount: number,
): Promise<RockModels> {
  const models = await Promise.all(
    Array.from({length: speciesCount}, (_, species) =>
      loadModel(new URL(`rock_species${species}.json`, baseUrl)),
    ),
  );
  const architecture = models[0].architecture;
  for (const [species, model] of models.entries()) {
    if (!sameArchitecture(model.architecture, architecture)) {
      throw new Error(
        `rock_species${species}: architecture differs from species 0.`,
      );
    }
  }
  const weights = new Float32Array(speciesCount * architecture.floatsPerModel);
  models.forEach((model, species) =>
    weights.set(model.weights, species * architecture.floatsPerModel),
  );
  return {architecture, weights};
}

/** WGSL constants that size the arrays and loops of the neural evaluator. */
export function architectureToWgsl(architecture: MlpArchitecture): string {
  return [
    `const NN_FREQUENCIES: u32 = ${architecture.numFrequencies}u;`,
    `const NN_INPUT_DIM: u32 = ${architecture.inputDim}u;`,
    `const NN_HIDDEN_WIDTH: u32 = ${architecture.hiddenWidth}u;`,
    `const NN_HIDDEN_LAYERS: u32 = ${architecture.hiddenLayers}u;`,
    `const NN_FLOATS_PER_MODEL: u32 = ${architecture.floatsPerModel}u;`,
    `const NN_ACTIVATION_SILU: bool = ${architecture.activation === 'silu'};`,
  ].join('\n');
}

async function loadModel(
  manifestUrl: URL,
): Promise<{architecture: MlpArchitecture; weights: Float32Array}> {
  const manifest = (await fetchOk(manifestUrl).then(r =>
    r.json(),
  )) as ModelManifest;
  const architecture = validateManifest(manifest, manifestUrl.pathname);
  const buffer = await fetchOk(
    new URL(manifest.weights.file, manifestUrl),
  ).then(r => r.arrayBuffer());
  if (buffer.byteLength !== manifest.weights.count * 4) {
    throw new Error(
      `${manifest.weights.file}: unexpected size ${buffer.byteLength} bytes.`,
    );
  }
  // DataView reads little-endian explicitly, independent of host byte order.
  // DataView 显式按小端读取，不依赖运行平台的字节序。
  const view = new DataView(buffer);
  const weights = new Float32Array(manifest.weights.count);
  for (let i = 0; i < weights.length; i++) {
    weights[i] = view.getFloat32(i * 4, true);
  }
  return {architecture, weights};
}

/**
 * Checks the manifest against what the WGSL evaluator assumes: positional
 * encoding, a supported activation, equal hidden widths, scalar output, and
 * each layer stored contiguously as weights followed by bias.
 */
function validateManifest(
  manifest: ModelManifest,
  name: string,
): MlpArchitecture {
  const fail = (reason: string): never => {
    throw new Error(`${name}: ${reason}`);
  };
  if (manifest.format_version !== SUPPORTED_FORMAT_VERSION) {
    fail(`unsupported format_version ${manifest.format_version}`);
  }
  if (manifest.encoding.type !== 'positional')
    fail('expected positional encoding');
  const activation = manifest.activation as Activation;
  if (!ACTIVATIONS.includes(activation)) {
    fail(`unsupported activation ${manifest.activation}`);
  }
  const {dtype, byte_order: byteOrder} = manifest.weights;
  if (dtype !== 'float32' || byteOrder !== 'little')
    fail('expected little-endian float32');

  const layers = manifest.layers;
  if (layers.length < 2) fail('expected at least one hidden layer');
  const numFrequencies = manifest.encoding.num_frequencies;
  const inputDim = 3 * (1 + 2 * numFrequencies);
  const hiddenWidth = layers[0].out_features;
  let expectedIn = inputDim;
  let offset = 0;
  for (const [index, layer] of layers.entries()) {
    const isOutput = index === layers.length - 1;
    const expectedOut = isOutput ? 1 : hiddenWidth;
    if (
      layer.in_features !== expectedIn ||
      layer.out_features !== expectedOut
    ) {
      fail(
        `layer ${index} has shape ${layer.in_features}->${layer.out_features}`,
      );
    }
    const weightCount = layer.in_features * layer.out_features;
    if (
      layer.weight_offset !== offset ||
      layer.bias_offset !== offset + weightCount
    ) {
      fail(`layer ${index} is not stored contiguously`);
    }
    offset += weightCount + layer.out_features;
    expectedIn = hiddenWidth;
  }
  if (offset !== manifest.weights.count)
    fail('weight count does not match layers');

  return {
    numFrequencies,
    inputDim,
    hiddenWidth,
    hiddenLayers: layers.length - 1,
    activation,
    floatsPerModel: offset,
  };
}

function sameArchitecture(a: MlpArchitecture, b: MlpArchitecture): boolean {
  return (
    a.numFrequencies === b.numFrequencies &&
    a.hiddenWidth === b.hiddenWidth &&
    a.hiddenLayers === b.hiddenLayers &&
    a.activation === b.activation &&
    a.floatsPerModel === b.floatsPerModel
  );
}

async function fetchOk(url: URL): Promise<Response> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url.pathname}: HTTP ${response.status}`);
  }
  return response;
}
