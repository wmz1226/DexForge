"""Tile Cholesky mass solve with an explicit implicit-function adjoint."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import warp as wp


TILE_BLOCK_SIZE = 64


@wp.struct
class MassSolveOutput:
  solution: wp.array2d(dtype=float)
  factor: wp.array3d(dtype=float)


@wp.struct
class MassAdjointInput:
  factor: wp.array3d(dtype=float)
  solution: wp.array2d(dtype=float)
  solution_gradient: wp.array2d(dtype=float)


@wp.struct
class MassAdjointOutput:
  matrix_gradient: wp.array3d(dtype=float)
  rhs_gradient: wp.array2d(dtype=float)


@dataclass(frozen=True)
class FastMassWorkspace:
  forward: MassSolveOutput
  backward: MassAdjointOutput


@cache
def _solve_kernel(dofs: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(matrix: wp.array3d(dtype=float),
             rhs: wp.array2d(dtype=float), output: MassSolveOutput):
    world = wp.tid()
    size = wp.static(dofs)
    matrix_tile = wp.tile_load(
        matrix[world], shape=(size, size), bounds_check=False)
    rhs_tile = wp.tile_load(rhs[world], shape=size, bounds_check=False)
    factor = wp.tile_cholesky(matrix_tile)
    solution = wp.tile_cholesky_solve(factor, rhs_tile)
    wp.tile_store(output.factor[world], factor, bounds_check=False)
    wp.tile_store(output.solution[world], solution, bounds_check=False)

  return kernel


@cache
def _rhs_adjoint_kernel(dofs: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(inputs: MassAdjointInput,
             rhs_gradient: wp.array2d(dtype=float)):
    world = wp.tid()
    size = wp.static(dofs)
    factor = wp.tile_load(
        inputs.factor[world], shape=(size, size), bounds_check=False)
    gradient = wp.tile_load(
        inputs.solution_gradient[world], shape=size, bounds_check=False)
    adjoint = wp.tile_cholesky_solve(factor, gradient)
    wp.tile_store(rhs_gradient[world], adjoint, bounds_check=False)

  return kernel


@wp.kernel(enable_backward=False)
def _matrix_adjoint(inputs: MassAdjointInput,
                    rhs_gradient: wp.array2d(dtype=float),
                    matrix_gradient: wp.array3d(dtype=float)):
  world, row, column = wp.tid()
  matrix_gradient[world, row, column] = (
      -rhs_gradient[world, row] * inputs.solution[world, column])


def allocate(worlds: int, dofs: int, device,
             *, requires_grad: bool = False) -> FastMassWorkspace:
  forward = MassSolveOutput()
  forward.solution = wp.empty(
      (worlds, dofs), dtype=float, device=device,
      requires_grad=requires_grad, retain_grad=requires_grad)
  forward.factor = wp.empty(
      (worlds, dofs, dofs), dtype=float, device=device)
  backward = MassAdjointOutput()
  backward.matrix_gradient = wp.empty(
      (worlds, dofs, dofs), dtype=float, device=device)
  backward.rhs_gradient = wp.empty(
      (worlds, dofs), dtype=float, device=device)
  return FastMassWorkspace(forward, backward)


def solve(matrix: wp.array, rhs: wp.array,
          workspace: FastMassWorkspace) -> wp.array:
  worlds, dofs = rhs.shape
  wp.launch_tiled(
      _solve_kernel(dofs), dim=worlds, inputs=[matrix, rhs],
      outputs=[workspace.forward], block_dim=TILE_BLOCK_SIZE,
      device=rhs.device)
  return workspace.forward.solution


def backward(solution_gradient: wp.array,
             workspace: FastMassWorkspace) -> MassAdjointOutput:
  worlds, dofs = solution_gradient.shape
  inputs = MassAdjointInput()
  inputs.factor = workspace.forward.factor
  inputs.solution = workspace.forward.solution
  inputs.solution_gradient = solution_gradient
  wp.launch_tiled(
      _rhs_adjoint_kernel(dofs), dim=worlds, inputs=[inputs],
      outputs=[workspace.backward.rhs_gradient], block_dim=TILE_BLOCK_SIZE,
      device=solution_gradient.device)
  wp.launch(
      _matrix_adjoint, dim=(worlds, dofs, dofs),
      inputs=[inputs, workspace.backward.rhs_gradient],
      outputs=[workspace.backward.matrix_gradient],
      device=solution_gradient.device)
  return workspace.backward
