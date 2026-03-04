from __future__ import annotations

from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import ufl
from dolfinx import default_real_type

from fenicsx_ii import Average, MappedRestriction, NonlinearProblem

UPPER_TAG = 1


def _mark_upper_half(mesh: dolfinx.mesh.Mesh) -> dolfinx.mesh.MeshTags:
    tdim = mesh.topology.dim
    cell_map = mesh.topology.index_map(tdim)
    num_local_cells = cell_map.size_local
    local_cells = np.arange(num_local_cells, dtype=np.int32)
    cell_values = np.zeros(num_local_cells, dtype=np.int32)

    upper_cells = dolfinx.mesh.locate_entities(
        mesh, tdim, lambda x: x[1] >= 0.5 - 1e-14
    )
    upper_cells = upper_cells[upper_cells < num_local_cells]
    cell_values[upper_cells] = UPPER_TAG
    return dolfinx.mesh.meshtags(mesh, tdim, local_cells, cell_values)


def _periodic_shift_down_half(x: np.ndarray) -> np.ndarray:
    x_out = x.copy()
    x_out[1] = np.mod(x_out[1] - 0.5, 1.0)
    return x_out


def _solve_level(n: int) -> tuple[float, float]:
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, cell_type=dolfinx.mesh.CellType.triangle
    )
    cell_tags = _mark_upper_half(mesh)

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    q_el = basix.ufl.quadrature_element(mesh.basix_cell(), value_shape=(), degree=4)
    Q = dolfinx.fem.functionspace(mesh, q_el)

    u_h = dolfinx.fem.Function(V)
    u_h.interpolate(lambda x: np.full(x.shape[1], 0.5, dtype=default_real_type))
    v = ufl.TestFunction(V)

    restriction = MappedRestriction(mesh, _periodic_shift_down_half)
    u_shift = Average(u_h, restriction, Q)

    x = ufl.SpatialCoordinate(mesh)
    u_exact = 2.0 + ufl.sin(ufl.pi * x[0]) * ufl.cos(2.0 * ufl.pi * x[1])
    u_exact_shift = 2.0 + ufl.sin(ufl.pi * x[0]) * ufl.cos(
        2.0 * ufl.pi * (x[1] - 0.5)
    )

    def q(u):
        return 1.0 + u * u

    def f_lin(w):
        return w

    dx = ufl.Measure("dx", domain=mesh)
    dx_sub = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)

    rhs_bulk = -ufl.div(q(u_exact) * ufl.grad(u_exact)) - f_lin(u_exact)
    rhs_upper = -f_lin(u_exact_shift)

    F = q(u_h) * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * dx
    F -= f_lin(u_h) * v * dx
    F -= f_lin(u_shift) * v * dx_sub(UPPER_TAG)
    F -= rhs_bulk * v * dx
    F -= rhs_upper * v * dx_sub(UPPER_TAG)

    u_D = dolfinx.fem.Function(V)
    u_exact_expr = dolfinx.fem.Expression(u_exact, V.element.interpolation_points)
    u_D.interpolate(u_exact_expr)
    fdim = mesh.topology.dim - 1
    boundary_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
    )
    dofs_bc = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = dolfinx.fem.dirichletbc(u_D, dofs_bc)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "none",
        "snes_atol": 1e-10,
        "snes_rtol": 1e-10,
        "snes_max_it": 20,
        "snes_error_if_not_converged": True,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
    }
    problem = NonlinearProblem(
        F,
        u_h,
        bcs=[bc],
        petsc_options=petsc_options,
        petsc_options_prefix=f"test_periodic_nonlinear_poisson_{n}_",
    )
    problem.solve()
    assert problem.solver.getConvergedReason() > 0

    u_exact_h = dolfinx.fem.Function(V)
    u_exact_h.interpolate(u_exact_expr)
    l2_form = dolfinx.fem.form((u_h - u_exact_h) ** 2 * dx)
    l2_local = dolfinx.fem.assemble_scalar(l2_form)
    l2_error = np.sqrt(mesh.comm.allreduce(l2_local, op=MPI.SUM))

    linf_local = (
        np.max(np.abs(u_h.x.array - u_exact_h.x.array)) if u_h.x.array.size else 0.0
    )
    linf_error = mesh.comm.allreduce(linf_local, op=MPI.MAX)
    return l2_error, linf_error


def test_nonlinear_poisson_periodic_coupling_convergence() -> None:
    levels = [12, 24]
    hs = [1.0 / n for n in levels]
    errors = [_solve_level(n) for n in levels]
    l2_errors = [err[0] for err in errors]
    linf_errors = [err[1] for err in errors]

    rate = np.log(l2_errors[0] / l2_errors[1]) / np.log(hs[0] / hs[1])
    assert rate > 1.4
    assert linf_errors[-1] < 2e-2
