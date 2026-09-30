// Exact scalar-joint rigid-body point Jacobians, without Python body masks.
#include <cstdint>

extern "C" void point_jacobians(
    const double* offsets, const std::int64_t* bodies, const double* body_jac,
    std::int64_t count, std::int64_t dofs, double* output) {
  constexpr std::int64_t axes = 3;
  constexpr std::int64_t jacobian_kinds = 2;
  for (std::int64_t point = 0; point < count; ++point) {
    const double* delta = offsets + point * axes;
    const double* linear = body_jac + bodies[point] * jacobian_kinds * axes * dofs;
    const double* angular = linear + axes * dofs;
    double* result = output + point * axes * dofs;
    for (std::int64_t joint = 0; joint < dofs; ++joint) {
      result[joint] = linear[joint]
          + (angular[dofs + joint] * delta[2] - angular[2 * dofs + joint] * delta[1]);
      result[dofs + joint] = linear[dofs + joint]
          + (angular[2 * dofs + joint] * delta[0] - angular[joint] * delta[2]);
      result[2 * dofs + joint] = linear[2 * dofs + joint]
          + (angular[joint] * delta[1] - angular[dofs + joint] * delta[0]);
    }
  }
}
