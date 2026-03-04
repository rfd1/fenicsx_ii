"""Tests for NonlinearProblem"""

from mpi4py import MPI

import numpy as np
import pytest
import basix
from fenicsx_ii import Average, MappedRestriction

import ufl
from dolfinx import default_real_type
from dolfinx.fem import Function, dirichletbc, form, functionspace, locate_dofs_geometrical
from dolfinx.fem.forms import extract_function_spaces
from dolfinx.mesh import create_unit_square
from ufl import TestFunction, TrialFunction, derivative, dx, grad, inner
from fenicsx_ii import NonlinearProblem
import dolfinx
from petsc4py import PETSc


petsc_options_linear = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_error_if_not_converged": True
}

petsc_options_nonlinear = {
    **petsc_options_linear,
    "snes_rtol": 10 * np.finfo(PETSc.ScalarType).eps,  # type: ignore[attr-defined]
    "snes_max_it": 10,
    "snes_monitor": None,
    "snes_error_if_not_converged": True
}


def test_plain_nonlinear_solver() -> None:
    """Test solution of a nonlinear problem with single form with no restrictions."""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    # Define function space and trial/test functions
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)
    # Define forms
    x = ufl.SpatialCoordinate(mesh)
    f = x[0] + 3 * x[1]
    F = (
        ufl.inner(u, v) * dx + 0.001 * ufl.inner(u**2 - f * u, v) * dx
        - ufl.inner(f, v) * dx
    )
    # Define boundary conditions
    f_expr = dolfinx.fem.Expression(f, V.element.interpolation_points)
    f_bc = dolfinx.fem.Function(V)
    f_bc.interpolate(f_expr)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[1], 1.0))
    dofs_bc = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bcs = [dolfinx.fem.dirichletbc(f_bc, dofs_bc)]
    # Solve
    problem = NonlinearProblem(
        F, u, bcs=bcs, petsc_options_prefix="test_plain_nonlinear_solver_", petsc_options=petsc_options_nonlinear)
    problem.solve()
    assert problem.solver.getConvergedReason() > 0
    # Compute error
    error_ufl = dolfinx.fem.form(ufl.inner(u - f, u - f) * dx)
    error = np.sqrt(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(error_ufl), op=MPI.SUM))
    tol = 500 * np.finfo(PETSc.ScalarType).eps  # type: ignore[attr-defined]
    assert error < tol


def test_nonlinear_poisson(linear=False):
    from dolfinx import mesh, fem

    def q(u):
        if linear:
            return 1

        return 1 + u*u

    msh1 = mesh.create_unit_square(
        MPI.COMM_WORLD, 10, 10,
        cell_type=mesh.CellType.triangle
    )

    # create second mesh [4,3] \times [5,4]
    translation_vector = np.array([4., 3.])
    msh2 = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [translation_vector, translation_vector + np.ones(2)],
        [20, 20],
        cell_type=mesh.CellType.quadrilateral,
    )

    def translate_domain_to_domain2(x):
        x_out = x.copy()
        for i, ti in enumerate(translation_vector):
            x_out[i] += ti
        return x_out

    restriction = MappedRestriction(msh1, translate_domain_to_domain2)

    def g(x):
        return x[0] + 2 * x[1] * x[1]

    x1 = ufl.SpatialCoordinate(msh1)
    u_ufl = 1 + x1[0] + 2 * x1[1]
    f = -ufl.div(q(u_ufl) * ufl.grad(u_ufl)) - g(x1 + ufl.as_vector(translation_vector.tolist()))

    V1 = fem.functionspace(msh1, ("Lagrange", 1))
    def u_exact(x):
        return eval(str(u_ufl))

    u_D = fem.Function(V1)
    u_D.interpolate(u_exact)
    fdim = msh1.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh1, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
    )
    bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V1, fdim, boundary_facets))

    # define u_g and interpolate g on it on msh2
    V2 = fem.functionspace(msh2, ("Lagrange", 1))
    u_g = fem.Function(V2)
    u_g.interpolate(g)

    # define u_g on msh1
    q_degree = 2
    quadrature = basix.ufl.quadrature_element(
        msh1.basix_cell(), value_shape=(), degree=q_degree
    )
    Q2 = fem.functionspace(msh1, quadrature)
    u_g__on_msh1 = Average(u_g, restriction, Q2)

    uh = fem.Function(V1)
    v = ufl.TestFunction(V1)
    dx1 = ufl.Measure("dx", domain=msh1)
    F = q(uh) * ufl.dot(ufl.grad(uh), ufl.grad(v))*dx1 - u_g__on_msh1*v*dx1 - f*v*dx1

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "none",
        "snes_atol": 1e-8,
        "snes_rtol": 1e-8,
        "snes_monitor": None,
        "ksp_error_if_not_converged": True,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    problem = NonlinearProblem(
        F, uh, bcs=[bc],
        petsc_options=petsc_options,
        petsc_options_prefix="nonlinpoisson"
    )

    problem.solve()
    converged = problem.solver.getConvergedReason()
    num_iter = problem.solver.getIterationNumber()
    assert converged > 0, f"Solver did not converge, got {converged}"
    assert num_iter <= 11, f"Solver did not converge within 8 iterations, took {num_iter}"
    print(
        f"Solver converged after {num_iter} iterations with converged reason {converged}"
    )

    # Compute L2 error and error at nodes
    V_ex = fem.functionspace(msh1, ("Lagrange", 2))
    u_ex = fem.Function(V_ex)
    u_ex.interpolate(u_exact)
    error_local = fem.assemble_scalar(fem.form((uh - u_ex) ** 2 * ufl.dx))
    error_L2 = np.sqrt(msh1.comm.allreduce(error_local, op=MPI.SUM))
    if msh1.comm.rank == 0:
        print(f"L2-error: {error_L2:.2e}")
        assert error_L2 < 1e-4

    # Compute values at mesh vertices
    error_max = msh1.comm.allreduce(
        np.max(np.abs(uh.x.array - u_D.x.array)), op=MPI.MAX
    )
    if msh1.comm.rank == 0:
        print(f"Error_max: {error_max:.2e}")
        assert error_max < 1e-4


def test_linear_poisson():
    test_nonlinear_poisson(linear=True)


if __name__ == '__main__':
    test_nonlinear_poisson()
    test_linear_poisson()
