// Chosen per pipeline: procedural teacher (false) or distilled network (true).
// 每条 pipeline 单独指定：false 用程序化“老师”，true 用蒸馏出的神经网络。
override USE_NEURAL_ROCKS: bool = false;

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

fn rockSdfInCell(p: vec3f, cell: vec2f) -> f32 {
  let rock = rockInstance(cell);
  if (!rock.exists) {
    return 1e5;
  }
  let w = (p - rock.center) / rock.scale;
  let local = vec3f(rock.cosYaw * w.x - rock.sinYaw * w.z, w.y,
                    rock.sinYaw * w.x + rock.cosYaw * w.z);
  return rockShapeSdf(local, rock.species) * rock.scale;
}

// A rock can extend into neighboring cells, so all 3x3 neighbors are tested.
// 石头可能伸进相邻格子，所以要检查周围 3x3 个格子。
fn rockSdf(p: vec3f) -> f32 {
  let cell = floor(p.xz / ROCK_CELL_SIZE);
  var d = 1e5;
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      d = min(d, rockSdfInCell(p, cell + vec2f(f32(i), f32(j))));
    }
  }
  return d;
}
