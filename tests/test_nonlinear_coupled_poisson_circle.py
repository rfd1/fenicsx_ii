from __future__ import annotations

import inspect

from mpi4py import MPI

import basix
import basix.ufl
import dolfinx
import numpy as np
import ufl

from fenicsx_ii import Average, Circle, NonlinearProblem


def _create_embedded_line_mesh(
    num_nodes: int, z_min: float = -0.5, z_max: float = 0.5
) -> dolfinx.mesh.Mesh:
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        nodes = np.zeros((num_nodes, 3), dtype=np.float64)
        nodes[:, 2] = np.linspace(z_min, z_max, num_nodes)
        connectivity = np.repeat(np.arange(num_nodes), 2)[1:-1].reshape(
            num_nodes - 1, 2
        )
    else:
        nodes = np.zeros((0, 3), dtype=np.float64)
        connectivity = np.zeros((0, 2), dtype=np.int64)

    c_el = ufl.Mesh(
        basix.ufl.element("Lagrange", basix.CellType.interval, 1, shape=(3,))
    )
    sig = inspect.signature(dolfinx.mesh.create_cell_partitioner)
    kwargs = {}
    if "max_facet_to_cell_links" in list(sig.parameters.keys()):
        kwargs["max_facet_to_cell_links"] = 2

    return dolfinx.mesh.create_mesh(
        comm,
        x=nodes,
        cells=connectivity,
        e=c_el,
        partitioner=dolfinx.mesh.create_cell_partitioner(
            dolfinx.mesh.GhostMode.shared_facet, **kwargs
        ),
        **kwargs,
    )


def test_nonlinear_coupled_poisson_circle_explicit_jacobian() -> None:
    comm = MPI.COMM_WORLD
    m = 6
    n = m
    omega = dolfinx.mesh.create_box(
        comm,
        [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        [m, m, m],
        cell_type=dolfinx.mesh.CellType.tetrahedron,
    )
    lmbda = _create_embedded_line_mesh(n)

    v_space = dolfinx.fem.functionspace(omega, ("Lagrange", 1))
    q_space = dolfinx.fem.functionspace(lmbda, ("Lagrange", 1))
    uh = dolfinx.fem.Function(v_space, name="u")
    ph = dolfinx.fem.Function(q_space, name="p")
    uh.interpolate(
        lambda x: np.full(x.shape[1], 0.2, dtype=dolfinx.default_scalar_type)
    )
    ph.interpolate(
        lambda x: np.full(x.shape[1], -0.1, dtype=dolfinx.default_scalar_type)
    )

    v = ufl.TestFunction(v_space)
    q = ufl.TestFunction(q_space)
    du = ufl.TrialFunction(v_space)
    dp = ufl.TrialFunction(q_space)

    radius = 0.05
    quadrature_degree = 5
    restriction_trial = Circle(lmbda, radius, degree=quadrature_degree)
    restriction_test = Circle(lmbda, radius, degree=quadrature_degree)
    q_el = basix.ufl.quadrature_element(
        lmbda.basix_cell(), value_shape=(), degree=quadrature_degree
    )
    rs = dolfinx.fem.functionspace(lmbda, q_el)

    avg_u = Average(uh, restriction_trial, rs)
    avg_v = Average(v, restriction_test, rs)
    avg_du = Average(du, restriction_trial, rs)

    dx_3d = ufl.Measure("dx", domain=omega)
    dx_1d = ufl.Measure("dx", domain=lmbda)

    area = ufl.pi * radius**2
    perimeter = 2 * ufl.pi * radius
    beta = dolfinx.fem.Constant(omega, 1e-3)

    # Zero is the exact solution; reaction terms ensure uniqueness with no BCs.
    f0 = (uh * v + ufl.inner(ufl.grad(uh), ufl.grad(v))) * dx_3d
    f0 += perimeter * ufl.inner(avg_u - ph, avg_v) * dx_1d
    f0 += beta * (avg_u - ph) ** 3 * avg_v * dx_1d

    f1 = (ph * q + area * ufl.inner(ufl.grad(ph), ufl.grad(q))) * dx_1d
    f1 += perimeter * ufl.inner(ph - avg_u, q) * dx_1d
    f1 -= beta * (avg_u - ph) ** 3 * q * dx_1d

    j00 = (du * v + ufl.inner(ufl.grad(du), ufl.grad(v))) * dx_3d
    j00 += perimeter * ufl.inner(avg_du, avg_v) * dx_1d
    j00 += 3 * beta * (avg_u - ph) ** 2 * avg_du * avg_v * dx_1d

    j01 = -perimeter * ufl.inner(dp, avg_v) * dx_1d
    j01 += -3 * beta * (avg_u - ph) ** 2 * dp * avg_v * dx_1d

    j10 = -perimeter * ufl.inner(avg_du, q) * dx_1d
    j10 += 3 * beta * (avg_u - ph) ** 2 * avg_du * q * dx_1d

    j11 = (dp * q + area * ufl.inner(ufl.grad(dp), ufl.grad(q))) * dx_1d
    j11 += perimeter * ufl.inner(dp, q) * dx_1d
    j11 += -3 * beta * (avg_u - ph) ** 2 * dp * q * dx_1d

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
        [f0, f1],
        [uh, ph],
        J=[[j00, j01], [j10, j11]],
        bcs=[],
        petsc_options=petsc_options,
        petsc_options_prefix="test_nonlinear_coupled_poisson_circle_",
    )
    problem.solve()

    assert problem.solver.getConvergedReason() > 0
    assert problem.solver.getIterationNumber() <= 8

    u_l2_local = dolfinx.fem.assemble_scalar(dolfinx.fem.form(uh * uh * dx_3d))
    u_l2 = np.sqrt(comm.allreduce(u_l2_local, op=MPI.SUM))
    p_l2_local = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ph * ph * dx_1d))
    p_l2 = np.sqrt(comm.allreduce(p_l2_local, op=MPI.SUM))

    assert u_l2 < 1e-9
    assert p_l2 < 1e-9
