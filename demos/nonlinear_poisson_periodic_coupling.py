# # Nonlinear Poisson with periodic coupling
#
# This demo follows the periodic-coupling setup discussed in
# <https://fenicsproject.discourse.group/t/nonlinear-poisson-with-periodic-coupling/19520>,
# but uses a simpler linear reaction term.
#
# We solve on $\Omega=(0,1)^2$ and define
# $\Omega_2=\{(x,y)\in\Omega\;|\; y\geq 0.5\}$.
# The periodic map is
#
# $$
# T(x,y)=\left(x,\; y-\frac{1}{2}\; \mathrm{mod}\;1\right).
# $$
#
# The model problem is
#
# $$
# -\nabla\cdot\left(q(u)\nabla u\right)
# = f_{\mathrm{lin}}(u) + \chi_{\Omega_2} f_{\mathrm{lin}}(u\circ T) + s
# \quad\text{in }\Omega,
# $$
#
# with
#
# $$
# q(u)=1+u^2,\qquad f_{\mathrm{lin}}(w)=w.
# $$
#
# We choose a manufactured exact solution and build $s$ from it, then solve one
# case and check the error.

# +
from __future__ import annotations

from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import ufl
from dolfinx import default_real_type

from fenicsx_ii import Average, MappedRestriction, NonlinearProblem

# -

# ## 1) Mesh and subdomain marker
# We use one triangular unit-square mesh and mark cells in the upper half
# with tag `UPPER_TAG`.

# +
UPPER_TAG = 1
n = 24

mesh = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, n, n, cell_type=dolfinx.mesh.CellType.triangle
)

tdim = mesh.topology.dim
cell_map = mesh.topology.index_map(tdim)
num_local_cells = cell_map.size_local
local_cells = np.arange(num_local_cells, dtype=np.int32)
cell_values = np.zeros(num_local_cells, dtype=np.int32)

upper_cells = dolfinx.mesh.locate_entities(mesh, tdim, lambda x: x[1] >= 0.5 - 1e-14)
upper_cells = upper_cells[upper_cells < num_local_cells]
cell_values[upper_cells] = UPPER_TAG
cell_tags = dolfinx.mesh.meshtags(mesh, tdim, local_cells, cell_values)

# -

# ## 2) Function spaces and unknown
# `V` stores the PDE solution and `Q` is a quadrature space used by the
# `Average` operator for the mapped coupling term.

# +
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
q_el = basix.ufl.quadrature_element(mesh.basix_cell(), value_shape=(), degree=4)
Q = dolfinx.fem.functionspace(mesh, q_el)

u_h = dolfinx.fem.Function(V, name="u_h")
u_h.interpolate(lambda x: np.full(x.shape[1], 0.5, dtype=default_real_type))
v = ufl.TestFunction(V)

# -

# ## 3) Periodic coupling map and averaged field
# The mapped value $u_h\circ T$ is represented with `Average`.

# +
def periodic_shift_down_half(x: np.ndarray) -> np.ndarray:
    x_out = x.copy()
    x_out[1] = np.mod(x_out[1] - 0.5, 1.0)
    return x_out


restriction = MappedRestriction(mesh, periodic_shift_down_half)
u_shift = Average(u_h, restriction, Q)

# -

# ## 4) Manufactured solution and source terms
# We pick
#
# $$
# u_{\mathrm{ex}}(x,y)=2+\sin(\pi x)\cos(2\pi y).
# $$
#
# Then we derive:
# - a bulk right-hand side term on all of $\Omega$
# - a coupling-source correction on $\Omega_2$.

# +
x = ufl.SpatialCoordinate(mesh)
u_exact = 2.0 + ufl.sin(ufl.pi * x[0]) * ufl.cos(2.0 * ufl.pi * x[1])
u_exact_shift = 2.0 + ufl.sin(ufl.pi * x[0]) * ufl.cos(2.0 * ufl.pi * (x[1] - 0.5))


def q(u):
    return 1.0 + u * u


def f_lin(w):
    return w


dx = ufl.Measure("dx", domain=mesh)
dx_sub = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)

rhs_bulk = -ufl.div(q(u_exact) * ufl.grad(u_exact)) - f_lin(u_exact)
rhs_upper = -f_lin(u_exact_shift)

# -

# ## 5) Weak form
# The residual is
#
# $$
# F(u_h;v) =
# \int_{\Omega} q(u_h)\nabla u_h\cdot\nabla v
# - \int_{\Omega} f_{\mathrm{lin}}(u_h)v
# - \int_{\Omega_2} f_{\mathrm{lin}}(u_h\circ T)\,v
# - \int_{\Omega} \mathrm{rhs}_{\mathrm{bulk}}\,v
# - \int_{\Omega_2} \mathrm{rhs}_{\mathrm{upper}}\,v.
# $$

# +
F = q(u_h) * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * dx
F -= f_lin(u_h) * v * dx
F -= f_lin(u_shift) * v * dx_sub(UPPER_TAG)
F -= rhs_bulk * v * dx
F -= rhs_upper * v * dx_sub(UPPER_TAG)

# -

# ## 6) Dirichlet boundary condition from manufactured solution

# +
u_D = dolfinx.fem.Function(V, name="u_D")
u_exact_expr = dolfinx.fem.Expression(u_exact, V.element.interpolation_points)
u_D.interpolate(u_exact_expr)

fdim = mesh.topology.dim - 1
boundary_facets = dolfinx.mesh.locate_entities_boundary(
    mesh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
)
dofs_bc = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = dolfinx.fem.dirichletbc(u_D, dofs_bc)

# -

# ## 7) Nonlinear solve

# +
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
    petsc_options_prefix="demo_periodic_nonlinear_poisson_single_case_",
)
problem.solve()

# -

# ## 8) Error check for one case
# We compare against the manufactured solution on this mesh only.

# +
u_exact_h = dolfinx.fem.Function(V, name="u_exact_h")
u_exact_h.interpolate(u_exact_expr)

l2_form = dolfinx.fem.form((u_h - u_exact_h) ** 2 * dx)
l2_local = dolfinx.fem.assemble_scalar(l2_form)
l2_error = np.sqrt(mesh.comm.allreduce(l2_local, op=MPI.SUM))

linf_local = (
    np.max(np.abs(u_h.x.array - u_exact_h.x.array)) if u_h.x.array.size else 0.0
)
linf_error = mesh.comm.allreduce(linf_local, op=MPI.MAX)

if MPI.COMM_WORLD.rank == 0:
    print(f"n={n}")
    print(f"Newton iterations: {problem.solver.getIterationNumber()}")
    print(f"L2 error:   {l2_error:.3e}")
    print(f"Linf error: {linf_error:.3e}")

assert l2_error < 5e-3
assert linf_error < 2e-2

# -
