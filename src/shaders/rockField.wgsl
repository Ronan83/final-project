// Chosen per pipeline: procedural teacher (false) or distilled network (true).
// 每条 pipeline 单独指定：false 用程序化“老师”，true 用蒸馏出的神经网络。
override USE_NEURAL_ROCKS: bool = false;

// `limit` for rockSdf() when the caller has no closer surface to compare against.
// 调用者没有更近的表面可比较时，作为 rockSdf() 的 limit 传入。
const ROCK_NO_LIMIT: f32 = 1e5;

fn rockShapeSdf(x: vec3f, species: u32) -> f32 {
  if (!USE_NEURAL_ROCKS) {
    return rockLocalSdf(x, species);
  }
  let base = length(rockShapeSpace(x, species)) - 1.0;
  if (base > ROCK_BOUND_MARGIN) {
    return rockFarBound(base);
  }
  return rockNeuralSdf(x, species);
}

// Returns min(best, distance to this cell's rock), skipping the terrain lookup and
// the shape SDF whenever a cheap bound shows the rock cannot be closer than `best`.
// 返回 min(best, 到本格石头的距离)；如果廉价的下界已经说明石头不可能比 best 更近，
// 就跳过地形查询和形状 SDF。
fn rockSdfInCell(p: vec3f, cell: vec2f, best: f32) -> f32 {
  let placement = rockPlacement(cell);
  if (!placement.exists) {
    return best;
  }
  // The rock lies in a vertical cylinder of radius ROCK_MAX_RADIUS * scale. Scaling by
  // ROCK_LIPSCHITZ_SCALE keeps this below every value rockLocalSdf() can return, so
  // culling never changes the result for procedural rocks.
  // 石头在半径为 ROCK_MAX_RADIUS * scale 的竖直圆柱内。乘上 ROCK_LIPSCHITZ_SCALE 后，
  // 这个下界不超过 rockLocalSdf() 的任何返回值，所以剔除不会改变程序化石头的结果。
  let cylinderBound =
      (length(p.xz - placement.centerXZ) - ROCK_MAX_RADIUS * placement.scale) *
      ROCK_LIPSCHITZ_SCALE;
  if (cylinderBound >= best) {
    return best;
  }
  let pose = rockPose(cell, placement);
  let w = (p - pose.center) / placement.scale;
  let local = vec3f(pose.cosYaw * w.x - pose.sinYaw * w.z, w.y,
                    pose.sinYaw * w.x + pose.cosYaw * w.z);
  return min(best, rockShapeSdf(local, pose.species) * placement.scale);
}

// Distance to the nearest rock, or `limit` if no rock is closer. Passing the
// terrain distance as `limit` lets most cells be culled. A rock can extend into
// neighboring cells, so all 3x3 neighbors are tested.
// 到最近石头的距离；没有更近的石头时返回 limit。把地形距离作为 limit 传进来，
// 大部分格子都能被剔除。石头可能伸进相邻格子，所以要检查周围 3x3 个格子。
fn rockSdf(p: vec3f, limit: f32) -> f32 {
  let cell = floor(p.xz / ROCK_CELL_SIZE);
  var d = limit;
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      d = rockSdfInCell(p, cell + vec2f(f32(i), f32(j)), d);
    }
  }
  return d;
}
