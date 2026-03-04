"""Tests for NonlinearProblem"""

from mpi4py import MPI

import numpy as np
import pytest
import basix
from fenicsx_ii import Average, MappedRestriction

import ufl
from dolfinx import default_real_type
from dolfinx.fem import (
    Function,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_geometrical,
)
from dolfinx.fem.forms import extract_function_spaces
from dolfinx.mesh import create_unit_square
from ufl import TestFunction, TrialFunction, derivative, dx, grad, inner
from fenicsx_ii import NonlinearProblem
from fenicsx_ii.petsc import assemble_jacobian, assemble_residual
import fenicsx_ii.petsc as fenicsx_ii_petsc
import fenicsx_ii.forms as fenicsx_ii_forms
import dolfinx
from petsc4py import PETSc


petsc_options_linear = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_error_if_not_converged": True,
}

petsc_options_nonlinear = {
    **petsc_options_linear,
    "snes_rtol": 10 * np.finfo(PETSc.ScalarType).eps,  # type: ignore[attr-defined]
    "snes_max_it": 10,
    "snes_monitor": None,
    "snes_error_if_not_converged": True,
}


def _build_dirichlet_laplace_problem(
    bc_value: float, prefix: str
) -> tuple[
    dolfinx.mesh.Mesh,
    dolfinx.fem.FunctionSpace,
    dolfinx.fem.Function,
    dolfinx.fem.Function,
    ufl.Form,
    dolfinx.fem.DirichletBC,
    NonlinearProblem,
]:
    """Create a simple nonlinear problem used to inspect BC handling in callbacks."""
    msh, V, u, u_D, F, J, bc = _build_dirichlet_laplace_forms_and_bc(bc_value)
    problem = NonlinearProblem(F, u, J=J, bcs=[bc], petsc_options_prefix=prefix)
    return msh, V, u, u_D, J, bc, problem


def _build_dirichlet_laplace_forms_and_bc(
    bc_value: float,
) -> tuple[
    dolfinx.mesh.Mesh,
    dolfinx.fem.FunctionSpace,
    dolfinx.fem.Function,
    dolfinx.fem.Function,
    ufl.Form,
    ufl.Form,
    dolfinx.fem.DirichletBC,
]:
    """Create a simple scalar Laplace form with global Dirichlet boundary condition."""
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 6, 6)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)
    F = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    J = ufl.derivative(F, u, ufl.TrialFunction(V))

    u_D = dolfinx.fem.Function(V)
    u_D.interpolate(lambda x: np.full(x.shape[1], bc_value, dtype=default_real_type))
    fdim = msh.topology.dim - 1
    boundary_facets = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
    )
    dofs_bc = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = dolfinx.fem.dirichletbc(u_D, dofs_bc)
    return msh, V, u, u_D, F, J, bc


def test_dirichlet_residual_boundary_rows_are_x_minus_g() -> None:
    """Residual callback should set constrained rows to x-g."""
    msh, _V, _u, _u_D, _J, bc, problem = _build_dirichlet_laplace_problem(
        bc_value=1.25, prefix="test_dirichlet_residual_boundary_rows_are_x_minus_g_"
    )

    # Evaluate residual at x=0 so constrained rows should be -g.
    with problem.x.localForm() as x_loc:
        x_loc.set(0.0)
    assemble_residual(
        problem.u, problem.F, problem.J, [bc], problem.solver, problem.x, problem.b
    )

    dofs, lz = bc._cpp_object.dof_indices()
    with problem.b.localForm() as b_loc:
        b_array = np.array(b_loc.array_r, copy=True)

    bc_owned = dofs[:lz]
    expected = np.full(len(bc_owned), -1.25, dtype=b_array.dtype)
    np.testing.assert_allclose(
        b_array[bc_owned], expected, atol=1e4 * np.finfo(default_real_type).eps
    )


def test_dirichlet_residual_applies_lifting_to_interior_rows() -> None:
    """Interior residual entries should be affected by lifting for non-zero BCs."""
    msh, V, _u, _u_D, _J, bc, problem = _build_dirichlet_laplace_problem(
        bc_value=2.0, prefix="test_dirichlet_residual_applies_lifting_to_interior_rows_"
    )

    with problem.x.localForm() as x_loc:
        x_loc.set(0.0)
    assemble_residual(
        problem.u, problem.F, problem.J, [bc], problem.solver, problem.x, problem.b
    )

    dofs, lz = bc._cpp_object.dof_indices()
    owned_bc = dofs[:lz]
    num_owned = V.dofmap.index_map.size_local * V.dofmap.bs
    interior_owned = np.setdiff1d(
        np.arange(num_owned, dtype=np.int32),
        owned_bc.astype(np.int32),
        assume_unique=False,
    )
    with problem.b.localForm() as b_loc:
        b_array = np.array(b_loc.array_r, copy=True)

    local_norm = (
        float(np.linalg.norm(b_array[interior_owned]))
        if len(interior_owned) > 0
        else 0.0
    )
    global_norm = msh.comm.allreduce(local_norm, op=MPI.SUM)
    assert global_norm > 1e3 * np.finfo(default_real_type).eps


def test_dirichlet_jacobian_zeroes_boundary_rows() -> None:
    """Jacobian callback should enforce identity rows at constrained dofs."""
    msh, V, _u, _u_D, _J_ufl, bc, problem = _build_dirichlet_laplace_problem(
        bc_value=0.5, prefix="test_dirichlet_jacobian_zeroes_boundary_rows_"
    )

    with problem.x.localForm() as x_loc:
        x_loc.set(0.0)
    assemble_jacobian(
        problem.u, problem.J, None, [bc], problem.solver, problem.x, problem.A, problem.A
    )

    dofs, lz = bc._cpp_object.dof_indices()
    owned_local = dofs[:lz].astype(np.int32, copy=False)
    owned_global = V.dofmap.index_map.local_to_global(owned_local)

    local_ok = 1
    local_checked = 0
    tol = 1e3 * np.finfo(default_real_type).eps
    for gdof in owned_global:
        cols, vals = problem.A.getRow(int(gdof))
        local_checked += 1
        diag_val = None
        for col, val in zip(cols, vals):
            if int(col) == int(gdof):
                diag_val = val
            elif not np.isclose(val, 0.0, atol=tol):
                local_ok = 0
        if diag_val is None or not np.isclose(diag_val, 1.0, atol=tol):
            local_ok = 0
        if hasattr(problem.A, "restoreRow"):
            problem.A.restoreRow(int(gdof), cols, vals)

    checked = msh.comm.allreduce(local_checked, op=MPI.SUM)
    ok = msh.comm.allreduce(local_ok, op=MPI.MIN)
    assert checked > 0
    assert ok == 1


def test_constructor_raises_if_dirichlet_has_no_compiled_lifting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction should fail if Jacobian bundle has no compiled lifting path."""
    _msh, _V, u, _u_D, F, J_ufl, bc = _build_dirichlet_laplace_forms_and_bc(1.0)

    original_compile = fenicsx_ii_petsc.compile_form_bundle
    direct_F_bundle = original_compile(F)

    def _compile_form_bundle_stub(
        form, form_compiler_options=None, jit_options=None, entity_maps=None
    ):
        if isinstance(form, ufl.Form) and len(form.arguments()) == 2:
            return fenicsx_ii_forms.CompiledFormBundle(
                direct=None,
                contributions=[],
                rank=2,
                test_space=form.arguments()[0].ufl_function_space(),
                trial_space=form.arguments()[1].ufl_function_space(),
            )
        return direct_F_bundle

    monkeypatch.setattr(fenicsx_ii_petsc, "compile_form_bundle", _compile_form_bundle_stub)

    with pytest.raises(RuntimeError, match="algebraic lifting fallback is disabled"):
        NonlinearProblem(
            F,
            u,
            J=J_ufl,
            bcs=[bc],
            petsc_options_prefix="test_constructor_raises_if_dirichlet_has_no_compiled_lifting_",
        )


def test_constructor_allows_no_lifting_path_without_bcs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-BC problems should not fail on missing Jacobian lifting metadata."""
    _msh, _V, u, _u_D, F, J_ufl, _bc = _build_dirichlet_laplace_forms_and_bc(1.0)

    original_compile = fenicsx_ii_petsc.compile_form_bundle
    direct_F_bundle = original_compile(F)

    def _compile_form_bundle_stub(
        form, form_compiler_options=None, jit_options=None, entity_maps=None
    ):
        if isinstance(form, ufl.Form) and len(form.arguments()) == 2:
            return fenicsx_ii_forms.CompiledFormBundle(
                direct=None,
                contributions=[],
                rank=2,
                test_space=form.arguments()[0].ufl_function_space(),
                trial_space=form.arguments()[1].ufl_function_space(),
            )
        return direct_F_bundle

    monkeypatch.setattr(fenicsx_ii_petsc, "compile_form_bundle", _compile_form_bundle_stub)

    problem = NonlinearProblem(
        F,
        u,
        J=J_ufl,
        bcs=[],
        petsc_options_prefix="test_constructor_allows_no_lifting_path_without_bcs_",
    )
    assert problem is not None


def test_average_problem_no_runtime_form_compilation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Solve should not call dolfinx.fem.form once the problem is constructed."""
    from dolfinx import mesh, fem

    msh1 = mesh.create_unit_square(
        MPI.COMM_WORLD, 6, 6, cell_type=mesh.CellType.triangle
    )
    translation_vector = np.array([1.0, 1.0])
    msh2 = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [translation_vector, translation_vector + np.ones(2)],
        [10, 10],
        cell_type=mesh.CellType.quadrilateral,
    )

    def translate_msh1_to_msh2(x):
        x_out = x.copy()
        for i, ti in enumerate(translation_vector):
            x_out[i] += ti
        return x_out

    restriction = MappedRestriction(msh1, translate_msh1_to_msh2)
    V1 = fem.functionspace(msh1, ("Lagrange", 1))
    V2 = fem.functionspace(msh2, ("Lagrange", 1))
    u = fem.Function(V1)
    v = ufl.TestFunction(V1)
    g_fun = fem.Function(V2)
    g_fun.interpolate(lambda x: x[0] + x[1])

    quadrature = basix.ufl.quadrature_element(msh1.basix_cell(), value_shape=(), degree=2)
    Q = fem.functionspace(msh1, quadrature)
    g_on_msh1 = Average(g_fun, restriction, Q)
    dx1 = ufl.Measure("dx", domain=msh1)
    F = ufl.inner((1 + u * u) * ufl.grad(u), ufl.grad(v)) * dx1 - g_on_msh1 * v * dx1

    u_D = fem.Function(V1)
    u_D.interpolate(lambda x: np.zeros(x.shape[1], dtype=default_real_type))
    fdim = msh1.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh1, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
    )
    bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V1, fdim, boundary_facets))

    problem = NonlinearProblem(
        F,
        u,
        bcs=[bc],
        petsc_options_prefix="test_average_problem_no_runtime_form_compilation_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "none",
            "snes_atol": 1e-8,
            "snes_rtol": 1e-8,
            "ksp_error_if_not_converged": True,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )

    def _raise_form(*args, **kwargs):
        raise RuntimeError("dolfinx.fem.form should not be called during solve")

    monkeypatch.setattr(dolfinx.fem, "form", _raise_form)
    problem.solve()
    assert problem.solver.getConvergedReason() > 0


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
        ufl.inner(u, v) * dx
        + 0.001 * ufl.inner(u**2 - f * u, v) * dx
        - ufl.inner(f, v) * dx
    )
    # Define boundary conditions
    f_expr = dolfinx.fem.Expression(f, V.element.interpolation_points)
    f_bc = dolfinx.fem.Function(V)
    f_bc.interpolate(f_expr)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[1], 1.0)
    )
    dofs_bc = dolfinx.fem.locate_dofs_topological(
        V, mesh.topology.dim - 1, boundary_facets
    )
    bcs = [dolfinx.fem.dirichletbc(f_bc, dofs_bc)]
    # Solve
    problem = NonlinearProblem(
        F,
        u,
        bcs=bcs,
        petsc_options_prefix="test_plain_nonlinear_solver_",
        petsc_options=petsc_options_nonlinear,
    )
    problem.solve()
    assert problem.solver.getConvergedReason() > 0
    # Compute error
    error_ufl = dolfinx.fem.form(ufl.inner(u - f, u - f) * dx)
    error = np.sqrt(
        mesh.comm.allreduce(dolfinx.fem.assemble_scalar(error_ufl), op=MPI.SUM)
    )
    tol = 500 * np.finfo(PETSc.ScalarType).eps  # type: ignore[attr-defined]
    assert error < tol


def test_nonlinear_poisson(linear=False):
    from dolfinx import mesh, fem

    def q(u):
        if linear:
            return 1

        return 1 + u * u

    msh1 = mesh.create_unit_square(
        MPI.COMM_WORLD, 10, 10, cell_type=mesh.CellType.triangle
    )

    # create second mesh [4,3] \times [5,4]
    translation_vector = np.array([4.0, 3.0])
    msh2 = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [translation_vector, translation_vector + np.ones(2)],
        [20, 20],
        cell_type=mesh.CellType.quadrilateral,
    )

    def translate_msh1_to_msh2(x):
        x_out = x.copy()
        for i, ti in enumerate(translation_vector):
            x_out[i] += ti
        return x_out

    restriction = MappedRestriction(msh1, translate_msh1_to_msh2)

    def g(x):
        return x[0] + 2 * x[1]

    x1 = ufl.SpatialCoordinate(msh1)
    u_ufl = 1 + x1[0] + 2 * x1[1]
    f = -ufl.div(q(u_ufl) * ufl.grad(u_ufl)) - g(
        x1 + ufl.as_vector(translation_vector.tolist())
    )

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
    F = (
        q(uh) * ufl.dot(ufl.grad(uh), ufl.grad(v)) * dx1
        - u_g__on_msh1 * v * dx1
        - f * v * dx1
    )

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
        F,
        uh,
        bcs=[bc],
        petsc_options=petsc_options,
        petsc_options_prefix="nonlinpoisson",
    )

    problem.solve()
    converged = problem.solver.getConvergedReason()
    num_iter = problem.solver.getIterationNumber()
    assert converged > 0, f"Solver did not converge, got {converged}"
    assert num_iter <= 11, (
        f"Solver did not converge within 8 iterations, took {num_iter}"
    )
    print(
        f"Solver converged after {num_iter} iterations with converged reason {converged}"
    )

    # Compute L2 error and error at nodes
    V_ex = fem.functionspace(msh1, ("Lagrange", 2))
    u_ex = fem.Function(V_ex)
    u_ex.interpolate(u_exact)
    error_local = fem.assemble_scalar(fem.form((uh - u_ex) ** 2 * ufl.dx))
    error_L2 = np.sqrt(msh1.comm.allreduce(error_local, op=MPI.SUM))

    # Compute values at mesh vertices
    error_max = msh1.comm.allreduce(
        np.max(np.abs(uh.x.array - u_D.x.array)), op=MPI.MAX
    )
    if msh1.comm.rank == 0:
        if linear:
            tol = 500 * np.finfo(PETSc.ScalarType).eps  # type: ignore[attr-defined]
        else:
            tol = np.sqrt(np.finfo(PETSc.ScalarType).eps)
        print(f"L2-error: {error_L2:.2e}")
        print(f"Error_max: {error_max:.2e}")
        print(f"Tol: {tol}")
        assert error_max < tol, "Max error too large"


def test_linear_poisson():
    test_nonlinear_poisson(linear=True)


if __name__ == "__main__":
    test_nonlinear_poisson()
    test_linear_poisson()
