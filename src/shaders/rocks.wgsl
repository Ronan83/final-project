const ROCK_CELL_SIZE: f32 = 14.0;
const ROCK_SPAWN_PROBABILITY: f32 = 0.45;
const ROCK_SPECIES_COUNT: u32 = 4u;

// Local-space rock shape: unit sphere displaced by fbm3, then scaled for safe stepping.
// Keep in sync with training/neural_sdf/procedural.py.
// 局部空间的石头：单位球加 fbm3 位移，再乘缩放系数保证步进安全。
// 必须和 training/neural_sdf/procedural.py 保持一致。
const ROCK_NOISE_FREQUENCY: f32 = 1.6;
const ROCK_DISPLACEMENT_SCALE: f32 = 1.1;
const ROCK_MAX_DISPLACEMENT: f32 = 0.55;
const ROCK_LIPSCHITZ_SCALE: f32 = 0.6;
const ROCK_BOUND_MARGIN: f32 = 0.8;
// The surface satisfies |x| <= |rockShapeSpace(x)| <= 1 + max displacement, so
// every rock lies inside this local-space sphere.
// 石头表面满足 |x| <= |rockShapeSpace(x)| <= 1 + 最大位移，所以整块石头都在这个局部球内。
const ROCK_MAX_RADIUS: f32 = 1.0 + ROCK_MAX_DISPLACEMENT;

// Split in two so culling only pays for one hash: the pose needs a second hash,
// yaw trigonometry and, most expensively, a terrain lookup.
// 拆成两部分，让剔除只需要一次哈希：姿态还需要第二次哈希、偏航角三角函数，
// 以及最贵的地形高度查询。
struct RockPlacement {
  centerXZ: vec2f,
  scale: f32,
  exists: bool,
};

struct RockPose {
  center: vec3f,
  cosYaw: f32,
  sinYaw: f32,
  species: u32,
};

fn rockPlacement(cell: vec2f) -> RockPlacement {
  let r = hash33(vec3f(cell, 7.0));
  let centerXZ = (cell + 0.5) * ROCK_CELL_SIZE + (r.xy - 0.5) * ROCK_CELL_SIZE * 0.4;
  return RockPlacement(centerXZ, 1.5 + 2.5 * r.x, r.z <= ROCK_SPAWN_PROBABILITY);
}

fn rockPose(cell: vec2f, placement: RockPlacement) -> RockPose {
  let s = hash33(vec3f(cell, 11.0));
  let yaw = s.x * 6.2831853;
  let center = vec3f(placement.centerXZ.x,
                     terrainHeight(placement.centerXZ) - 0.15 * placement.scale,
                     placement.centerXZ.y);
  return RockPose(center, cos(yaw), sin(yaw),
                  min(u32(s.y * f32(ROCK_SPECIES_COUNT)), ROCK_SPECIES_COUNT - 1u));
}

// Space in which a species' base shape is the unit sphere.
// 在这个坐标空间里，每种石头的基础形状都是单位球。
fn rockShapeSpace(x: vec3f, species: u32) -> vec3f {
  return x * vec3f(1.0, 1.0 + 0.1 * f32(species), 1.0);
}

// Conservative distance for points the displacement cannot reach.
// 位移够不到的远处点，返回保守的距离下界。
fn rockFarBound(base: f32) -> f32 {
  return (base - ROCK_MAX_DISPLACEMENT) * ROCK_LIPSCHITZ_SCALE;
}

// Procedural SDF of one species in local space. This is the teacher the neural SDF learns.
// 某一种石头在局部空间的程序化 SDF，也就是神经 SDF 要学习的“老师”。
fn rockLocalSdf(x: vec3f, species: u32) -> f32 {
  let y = rockShapeSpace(x, species);
  let base = length(y) - 1.0;
  // The same margin bounds the network's training domain.
  // 这个范围同时也是神经网络的训练区域。
  if (base > ROCK_BOUND_MARGIN) {
    return rockFarBound(base);
  }
  let seed = vec3f(17.0 * f32(species), 3.0, 0.0);
  let displacement = (fbm3(y * ROCK_NOISE_FREQUENCY + seed) - 0.5) * ROCK_DISPLACEMENT_SCALE;
  return (base + displacement) * ROCK_LIPSCHITZ_SCALE;
}
