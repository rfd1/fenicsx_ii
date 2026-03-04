# Copyright (C) 2018-2025 Garth N. Wells, Nathan Sime and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""High-level solver classes and functions for assembling PETSc objects.

Functions in this module generally apply functions in :mod:`dolfinx.fem`
to PETSc linear algebra objects and handle any PETSc-specific
preparation.

Note:
    The following does not apply to the high-level classes
    :class:`dolfinx.fem.petsc.LinearProblem`
    :class:`dolfinx.fem.petsc.NonlinearProblem`.

    Due to subtle issues in the interaction between petsc4py memory
    management and the Python garbage collector, it is recommended that
    the PETSc method ``destroy()`` is called on returned PETSc objects
    once the object is no longer required. Note that ``destroy()`` is
    collective over the object's MPI communicator.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Sequence

from petsc4py import PETSc

# ruff: noqa: E402
import dolfinx

assert dolfinx.has_petsc4py
from functools import partial

import numpy as np
from numpy import typing as npt

import dolfinx.la.petsc
import ufl
from dolfinx.fem import IntegralType, pack_coefficients, pack_constants
from dolfinx.fem.assemble import apply_lifting as _apply_lifting
from dolfinx.fem.bcs import DirichletBC
from dolfinx.fem.bcs import bcs_by_block as _bcs_by_block
from dolfinx.fem.forms import Form, derivative_block
from dolfinx.fem.forms import extract_function_spaces as _extract_function_spaces
from dolfinx.fem.forms import form as _create_form
from dolfinx.fem.function import Function as _Function
from dolfinx.mesh import EntityMap as _EntityMap

from .assembly import average_coefficients
from .matrix_assembler import assemble_matrix
from .matrix_assembler import create_matrix
from .ufl_operations import apply_replacer
from .vector_assembler import assemble_vector
from .vector_assembler import create_vector

__all__ = [
    "NonlinearProblem",
    "apply_lifting",
    "assemble_jacobian",
    "assemble_residual",
    "assign",
    "set_bc",
]


# -- Modifiers for Dirichlet conditions -----------------------------------
def apply_lifting(
    b: PETSc.Vec,  # type: ignore[name-defined]
    a: Sequence[Form] | Sequence[Sequence[Form]],
    bcs: Sequence[DirichletBC] | Sequence[Sequence[DirichletBC]] | None,
    x0: Sequence[PETSc.Vec] | None = None,  # type: ignore[name-defined]
    alpha: float = 1,
    constants: Sequence[npt.NDArray] | Sequence[Sequence[npt.NDArray]] | None = None,
    coeffs: (
        dict[tuple[IntegralType, int], npt.NDArray]
        | Sequence[Sequence[dict[tuple[IntegralType, int], npt.NDArray]]]
        | None
    ) = None,
) -> None:
    r"""Modify the right-hand side PETSc vector ``b`` to account for
    constraints (Dirichlet boundary conitions).

    See :func:`dolfinx.fem.apply_lifting` for a mathematical
    descriptions of the lifting operation.

    Args:
        b: Vector to modify in-place.
        a: List of bilinear forms. If ``b`` is not blocked or a nest,
            then ``a`` is a 1D sequence. If ``b`` is blocked or a nest,
            then ``a`` is  a 2D array of forms, with the ``a[i]`` forms
            used to modify the block/nest vector ``b[i]``.
        bcs: Boundary conditions to apply, which form a 2D array.
            If ``b`` is nested or blocked then ``bcs[i]`` are the
            boundary conditions to apply to block/nest ``i``.
            The function :func:`dolfinx.fem.bcs_by_block` can be
            used to prepare the 2D array of ``DirichletBC`` objects
            from the 2D sequence ``a``::

                bcs1 = fem.bcs_by_block(
                    fem.extract_function_spaces(a, 1),
                    bcs
                )

            If ``b`` is not blocked or nest, then ``len(bcs)`` must be
            equal to 1. The function :func:`dolfinx.fem.bcs_by_block`
            can be used to prepare the 2D array of ``DirichletBC``
            from the 1D sequence ``a``::

                bcs1 = fem.bcs_by_block(
                    fem.extract_function_spaces([a], 1),
                    bcs
                )

        x0: Vector to use in modify ``b`` (see
            :func:`dolfinx.fem.apply_lifting`). Treated as zero if
            ``None``.
        alpha: Scalar parameter in lifting (see
            :func:`dolfinx.fem.apply_lifting`).
        constants: Packed constant data appearing in the forms ``a``. If
            ``None``, the constant data will be packed by the function.
        coeffs: Packed coefficient data appearing in the forms ``a``. If
            ``None``, the coefficient data will be packed by the
            function.

    Note:
        Ghost contributions are not accumulated (not sent to owner).
        Caller is responsible for reverse-scatter to update the ghosts.

    Note:
        Boundary condition values are *not* set in ``b`` by this
        function. Use :func:`dolfinx.fem.DirichletBC.set` to set values
        in ``b``.
    """
    if b.getType() == PETSc.Vec.Type.NEST:  # type: ignore[attr-defined]
        x0 = [] if x0 is None else x0.getNestSubVecs()  # type: ignore[attr-defined]
        constants = (
            [pack_constants(forms) for forms in a] if constants is None else constants
        )  # type: ignore[assignment]
        coeffs = [pack_coefficients(forms) for forms in a] if coeffs is None else coeffs  # type: ignore[misc]
        for b_sub, a_sub, const, coeff in zip(b.getNestSubVecs(), a, constants, coeffs):  # type: ignore[arg-type]
            const_ = list(
                map(
                    lambda x: np.array([], dtype=PETSc.ScalarType) if x is None else x,
                    const,
                )  # type: ignore[attr-defined, call-overload]
            )
            apply_lifting(b_sub, a_sub, bcs, x0, alpha, const_, coeff)  # type: ignore[arg-type]
    else:
        with contextlib.ExitStack() as stack:
            if b.getAttr("_blocks") is not None:
                if x0 is not None:
                    offset0, offset1 = x0.getAttr("_blocks")  # type: ignore[attr-defined]
                    xl = stack.enter_context(x0.localForm())  # type: ignore[attr-defined]
                    xlocal = [
                        np.concatenate((xl[off0:off1], xl[offg0:offg1]))
                        for (off0, off1, offg0, offg1) in zip(
                            offset0, offset0[1:], offset1, offset1[1:]
                        )
                    ]
                else:
                    xlocal = None

                offset0, offset1 = b.getAttr("_blocks")
                with b.localForm() as b_l:
                    for i, (a_, off0, off1, offg0, offg1) in enumerate(
                        zip(a, offset0, offset0[1:], offset1, offset1[1:])
                    ):
                        const = (
                            pack_constants(a_) if constants is None else constants[i]
                        )  # type: ignore[arg-type]
                        coeff = pack_coefficients(a_) if coeffs is None else coeffs[i]  # type: ignore[arg-type, assignment, index]
                        const_ = [
                            np.empty(0, dtype=PETSc.ScalarType) if val is None else val  # type: ignore[attr-defined]
                            for val in const
                        ]
                        bx_ = np.concatenate((b_l[off0:off1], b_l[offg0:offg1]))
                        _apply_lifting(
                            bx_, a_, bcs, xlocal, float(alpha), const_, coeff
                        )  # type: ignore[arg-type]
                        size = off1 - off0
                        b_l.array_w[off0:off1] = bx_[:size]
                        b_l.array_w[offg0:offg1] = bx_[size:]
            else:
                x0 = [] if x0 is None else x0
                x0 = [stack.enter_context(x.localForm()) for x in x0]
                x0_r = [x.array_r for x in x0]
                b_local = stack.enter_context(b.localForm())
                _apply_lifting(b_local.array_w, a, bcs, x0_r, alpha, constants, coeffs)  # type: ignore[arg-type]

    return b


def set_bc(
    b: PETSc.Vec,  # type: ignore[name-defined]
    bcs: Sequence[DirichletBC] | Sequence[Sequence[DirichletBC]],
    x0: PETSc.Vec | None = None,  # type: ignore[name-defined]
    alpha: float = 1,
) -> None:
    r"""Set constraint (Dirchlet boundary condition) values in an vector.

    For degrees-of-freedoms that are constrained by a Dirichlet boundary
    condition, this function sets that degrees-of-freedom to ``alpha *
    (g - x0)``, where ``g`` is the boundary condition value.

    Only owned entries in ``b`` (owned by the MPI process) are modified
    by this function.

    Args:
        b: Vector to modify by setting  boundary condition values.
        bcs: Boundary conditions to apply. If ``b`` is nested or
            blocked, ``bcs`` is a 2D array and ``bcs[i]`` are the
            boundary conditions to apply to block/nest ``i``. Otherwise
            ``bcs`` should be a sequence of ``DirichletBC``\s. For
            block/nest problems, :func:`dolfinx.fem.bcs_by_block` can be
            used to prepare the 2D array of ``DirichletBC`` objects.
        x0: Vector used in the value that constrained entries are set
            to. If ``None``, ``x0`` is treated as zero.
        alpha: Scalar value used in the value that constrained entries
            are set to.
    """
    if len(bcs) == 0:
        return

    if not isinstance(bcs[0], Sequence):
        x0 = x0.array_r if x0 is not None else None
        for bc in bcs:
            bc.set(b.array_w, x0, alpha)  # type: ignore[union-attr]
    elif b.getType() == PETSc.Vec.Type.NEST:  # type: ignore[attr-defined]
        _b = b.getNestSubVecs()
        x0 = len(_b) * [None] if x0 is None else x0.getNestSubVecs()
        for b_sub, bc, x_sub in zip(_b, bcs, x0):  # type: ignore[assignment, arg-type]
            set_bc(b_sub, bc, x_sub, alpha)  # type: ignore[arg-type]
    else:  # block vector
        offset0, _ = b.getAttr("_blocks")
        b_array = b.getArray(readonly=False)
        x_array = x0.getArray(readonly=True) if x0 is not None else None
        for bcs, off0, off1 in zip(bcs, offset0, offset0[1:]):  # type: ignore[assignment]
            x0_sub = x_array[off0:off1] if x0 is not None else None  # type: ignore[index]
            for bc in bcs:
                bc.set(b_array[off0:off1], x0_sub, alpha)  # type: ignore[arg-type, union-attr]


def assemble_residual(
    u: _Function | Sequence[_Function],
    residual: ufl.Form,
    jacobian: Form | ufl.Form | Sequence[Sequence[Form]],
    bcs: Sequence[DirichletBC],
    _snes: PETSc.SNES,  # type: ignore[name-defined]
    x: PETSc.Vec,  # type: ignore[name-defined]
    b: PETSc.Vec,  # type: ignore[name-defined]
    lifting_forms: Sequence[Form] | None = None,
    lifting_forms_ufl: Sequence[ufl.Form] | None = None,
):
    """Assemble the residual at ``x`` into the vector ``b``.

    A function conforming to the interface expected by ``SNES.setFunction``
    can be created by fixing the first four arguments, e.g.:

    Example::

        snes = PETSc.SNES().create(mesh.comm)
        assemble_residual = functools.partial(
            dolfinx.fem.petsc.assemble_residual,
            u, residual, jacobian, bcs)
        snes.setFunction(assemble_residual, x, b)

    Args:
        u: Function(s) tied to the solution vector within the residual and
           Jacobian.
        residual: Form of the residual. It can be a sequence of forms.
        jacobian: Form of the Jacobian. It can be a nested sequence of
            forms.
        bcs: List of Dirichlet boundary conditions to lift the residual.
        _snes: The solver instance.
        x: The vector containing the point to evaluate the residual at.
        b: Vector to assemble the residual into.
    """
    # Update input vector before assigning
    dolfinx.la.petsc._ghost_update(
        x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD
    )  # type: ignore[attr-defined]

    # Assign the input vector to the unknowns
    assign(x, u)

    # Assign block data if block assembly is requested
    # Assemble the residual
    dolfinx.la.petsc._zero_vector(b)
    # Assemble raw residual first. Dirichlet BC contributions are handled
    # below via lifting and set_bc in SNES residual assembly.
    assemble_vector(residual, b=b)

    # Lift vector
    if isinstance(jacobian, Sequence):
        # Nest and blocked lifting
        bcs1 = _bcs_by_block(_extract_function_spaces(jacobian, 1), bcs)  # type: ignore[arg-type]
        if bcs:
            if all(
                (form is None) or hasattr(form, "_cpp_object")
                for row in jacobian
                for form in row
            ):  # type: ignore[arg-type]
                apply_lifting(b, jacobian, bcs=bcs1, x0=x, alpha=-1.0)
                dolfinx.la.petsc._ghost_update(
                    b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE
                )  # type: ignore[attr-defined]
            else:
                raise RuntimeError(
                    "Dirichlet residual lifting for block/nest problems requires "
                    "compiled Jacobian blocks; algebraic lifting fallback is disabled."
                )
            bcs0 = _bcs_by_block(_extract_function_spaces(residual), bcs)  # type: ignore[arg-type]
            set_bc(b, bcs0, x0=x, alpha=-1.0)
    else:
        # Single form lifting
        if bcs:
            if lifting_forms is not None:
                # apply_lifting expects one x0/bcs block per bilinear form.
                # For split/replaced forms we apply lifting contribution form-wise.
                if lifting_forms_ufl is None:
                    form_pairs = [
                        (lifting_form, None) for lifting_form in lifting_forms
                    ]
                else:
                    form_pairs = list(zip(lifting_forms, lifting_forms_ufl))

                for lifting_form, lifting_form_ufl in form_pairs:
                    if lifting_form_ufl is not None:
                        average_coefficients(lifting_form_ufl)
                    apply_lifting(b, [lifting_form], bcs=[bcs], x0=[x], alpha=-1.0)
                dolfinx.la.petsc._ghost_update(
                    b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE
                )  # type: ignore[attr-defined]
            elif hasattr(jacobian, "_cpp_object"):
                apply_lifting(b, [jacobian], bcs=[bcs], x0=[x], alpha=-1.0)
                dolfinx.la.petsc._ghost_update(
                    b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE
                )  # type: ignore[attr-defined]
            else:
                raise RuntimeError(
                    "Dirichlet residual lifting requires a compiled Jacobian form; "
                    "algebraic lifting fallback is disabled."
                )
            set_bc(b, bcs, x0=x, alpha=-1.0)
    dolfinx.la.petsc._ghost_update(
        b, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD
    )  # type: ignore[attr-defined]


def assemble_jacobian(
    u: Sequence[_Function] | _Function,
    jacobian: ufl.Form,
    preconditioner: ufl.Form | None,
    bcs: Sequence[DirichletBC],
    _snes: PETSc.SNES,  # type: ignore[name-defined]
    x: PETSc.Vec,  # type: ignore[name-defined]
    J: PETSc.Mat,  # type: ignore[name-defined]
    P_mat: PETSc.Mat,  # type: ignore[name-defined]
):
    """Assemble the Jacobian and preconditioner matrices at ``x``
    into ``J`` and ``P_mat``.

    A function conforming to the interface expected by ``SNES.setJacobian``
    can be created by fixing the first four arguments e.g.:

    Example::

        snes = PETSc.SNES().create(mesh.comm)
        assemble_jacobian = functools.partial(
            dolfinx.fem.petsc.assemble_jacobian,
            u, jacobian, preconditioner, bcs)
        snes.setJacobian(assemble_jacobian, A, P_mat)

    Args:
        u: Function tied to the solution vector within the residual and
            jacobian.
        jacobian: Compiled form of the Jacobian.
        preconditioner: Compiled form of the preconditioner.
        bcs: List of Dirichlet boundary conditions to apply to the Jacobian
             and preconditioner matrices.
        _snes: The solver instance.
        x: The vector containing the point to evaluate at.
        J: Matrix to assemble the Jacobian into.
        P_mat: Matrix to assemble the preconditioner into.
    """
    # Copy existing soultion into the function used in the residual and
    # Jacobian
    dolfinx.la.petsc._ghost_update(
        x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD
    )  # type: ignore[attr-defined]
    assign(x, u)

    # Assemble Jacobian
    J.zeroEntries()
    assemble_matrix(jacobian, bcs=bcs, A=J)
    J.assemble()
    if preconditioner is not None:
        P_mat.zeroEntries()
        assemble_matrix(preconditioner, bcs=bcs, A=P_mat)
        P_mat.assemble()


class NonlinearProblem:
    """High-level class for solving nonlinear variational problems
    with PETSc SNES.

    Solves problems of the form
    :math:`F_i(u, v) = 0, i=0,\\ldots,N\\ \\forall v \\in V` where
    :math:`u=(u_0,\\ldots,u_N), v=(v_0,\\ldots,v_N)` using PETSc
    SNES as the non-linear solver.

    Note:
        The deprecated version of this class for use with
        :class:`dolfinx.nls.petsc.NewtonSolver` has been renamed
        :class:`dolfinx.fem.petsc.NewtonSolverNonlinearProblem`.

    Note:
        This high-level class automatically handles PETSc memory
        management. The user does not need to manually call
        ``.destroy()`` on returned PETSc objects.
    """

    def __init__(
        self,
        F: ufl.form.Form | Sequence[ufl.form.Form],
        u: _Function | Sequence[_Function],
        *,
        petsc_options_prefix: str,
        bcs: Sequence[DirichletBC] | None = None,
        J: ufl.form.Form | Sequence[Sequence[ufl.form.Form]] | None = None,
        P: ufl.form.Form | Sequence[Sequence[ufl.form.Form]] | None = None,
        kind: str | Sequence[Sequence[str]] | None = None,
        petsc_options: dict | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: Sequence[_EntityMap] | None = None,
    ):
        """
        Initialize solver for a nonlinear variational problem.

        By default, the underlying SNES solver uses PETSc's default
        options. To use the robust combination of LU via MUMPS with
        a backtracking linesearch, pass:

        Example::

            petsc_options = {"ksp_type": "preonly",
                             "pc_type": "lu",
                             "pc_factor_mat_solver_type": "mumps",
                             "snes_linesearch_type": "bt",
            }

        Every PETSc object will have a unique options prefix set. We
        recommend discovering these prefixes dynamically via the
        petsc4py API rather than hard-coding each prefix value into
        the programme.

        Example::

            snes_options_prefix = problem.solver.getOptionsPrefix()
            jacobian_options_prefix = problem.A.getOptionsPrefix()

        Args:
            F: UFL form(s) representing the residual :math:`F_i`.
            u: Function(s) used to define the residual and Jacobian.
            bcs: Dirichlet boundary conditions.
            J: UFL form(s) representing the Jacobian
                :math:`J_{ij} = dF_i/du_j`. If not passed, derived
                automatically.
            P: UFL form(s) representing the preconditioner.
            kind: The PETSc matrix and vector kind. Common choices
                are ``mpi`` and ``nest``. See
                :func:`dolfinx.fem.petsc.create_matrix` and
                :func:`dolfinx.fem.petsc.create_vector` for more
                information.
            petsc_options_prefix: Mandatory named argument.
                Options prefix used as root prefix on all
                internally created PETSc objects. Typically ends with `_`.
                Must be the same on all ranks, and is usually unique within
                the programme.
            petsc_options: Options set on the underlying PETSc SNES only.
                The options must be the same on all ranks. For available
                choices for ``petsc_options``, see the
                `PETSc SNES documentation
                <https://petsc4py.readthedocs.io/en/stable/manual/snes/>`_.
                Options on other objects (matrices, vectors) should be set
                explicitly by the user.
            form_compiler_options: Options used in FFCx compilation of all
                forms. Run ``ffcx --help`` at the command line to see all
                available options.
            jit_options: Options used in CFFI JIT compilation of C code
                generated by FFCx. See ``python/dolfinx/jit.py`` for all
                available options. Takes priority over all other option
                values.
            entity_maps: If any trial functions, test functions, or
                coefficients in the form are not defined over the same mesh
                as the integration domain, a corresponding :class:
                `EntityMap<dolfinx.mesh.EntityMap>` must be provided.
        """
        # Ensure attributes exist even if initialization fails part-way
        self._snes = None
        self._A = None
        self._b = None
        self._x = None
        self._P_mat = None

        # Keep UFL forms for fenicsx_ii assemblers
        self._F_ufl = F

        if J is None:
            J = derivative_block(F, u)
        self._J_ufl = J
        self._preconditioner_ufl = P
        bcs = [] if bcs is None else bcs

        # Keep residual in UFL form. It may contain custom operators
        # (e.g. Average) that are not directly compilable by FFCx.
        self._F = F

        # Prefer a compiled Jacobian form for lifting in residual assembly.
        # For custom operators that FFCx cannot process directly, compile
        # the apply_replacer-transformed Jacobian contributions instead.
        self._J_lifting: Sequence[Form] | None = None
        self._J_lifting_ufl: Sequence[ufl.Form] | None = None
        direct_compile_error: Exception | None = None
        replacer_compile_error: Exception | None = None
        try:
            self._J = _create_form(
                J,
                form_compiler_options=form_compiler_options,
                jit_options=jit_options,
                entity_maps=entity_maps,
            )
            if isinstance(self._J, Form):
                self._J_lifting = [self._J]
        except Exception as exc:
            direct_compile_error = exc
            self._J = J
            if isinstance(J, ufl.Form):
                try:
                    replaced_forms = apply_replacer(J)
                    self._J_lifting_ufl = replaced_forms
                    self._J_lifting = [
                        _create_form(
                            j_form,
                            form_compiler_options=form_compiler_options,
                            jit_options=jit_options,
                            entity_maps=entity_maps,
                        )
                        for j_form in replaced_forms
                    ]
                except Exception as exc_replaced:
                    replacer_compile_error = exc_replaced
                    self._J_lifting = None
                    self._J_lifting_ufl = None

        def _has_compiled_lifting_path() -> bool:
            if isinstance(self._J, Sequence):
                return all(
                    (form is None) or hasattr(form, "_cpp_object")
                    for row in self._J
                    for form in row
                )

            if hasattr(self._J, "_cpp_object"):
                return True

            return (
                self._J_lifting is not None
                and len(self._J_lifting) > 0
                and all(hasattr(j_form, "_cpp_object") for j_form in self._J_lifting)
            )

        if bcs and not _has_compiled_lifting_path():
            diagnostics = []
            if direct_compile_error is not None:
                diagnostics.append(
                    "direct Jacobian compilation failed: "
                    f"{type(direct_compile_error).__name__}: {direct_compile_error}"
                )
            if replacer_compile_error is not None:
                diagnostics.append(
                    "replacer Jacobian compilation failed: "
                    f"{type(replacer_compile_error).__name__}: {replacer_compile_error}"
                )
            if not diagnostics:
                diagnostics.append("no compiled Jacobian lifting forms were produced")

            error = RuntimeError(
                "Dirichlet residual lifting requires compiled Jacobian form(s); "
                "algebraic lifting fallback is disabled. Compile the Jacobian "
                "directly or make apply_replacer(J) compilable.\n"
                "Compilation diagnostics:\n- " + "\n- ".join(diagnostics)
            )
            raise error from (replacer_compile_error or direct_compile_error)

        if P is not None:
            self._preconditioner = P
        else:
            self._preconditioner = None

        self._u = u

        # Create PETSc structures for the residual, Jacobian and solution
        # vector
        self._A = create_matrix(
            self._J_ufl,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )
        # Create PETSc structure for preconditioner if provided
        if self._preconditioner_ufl is not None:
            self._P_mat = create_matrix(
                self._preconditioner_ufl,
                form_compiler_options=form_compiler_options,
                jit_options=jit_options,
                entity_maps=entity_maps,
            )
        else:
            self._P_mat = None

        self._b = create_vector(
            self._F_ufl,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )
        self._x = create_vector(
            self._F_ufl,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )

        # Create the SNES solver and attach the corresponding Jacobian and
        # residual computation functions
        self._snes = PETSc.SNES().create(self.A.comm)  # type: ignore[attr-defined]
        self.solver.setJacobian(
            partial(assemble_jacobian, u, self._J_ufl, self._preconditioner_ufl, bcs),
            self.A,
            self.P_mat,
        )
        self.solver.setFunction(
            partial(
                assemble_residual,
                u,
                self._F_ufl,
                self.J,
                bcs,
                lifting_forms=self._J_lifting,
                lifting_forms_ufl=self._J_lifting_ufl,
            ),
            self.b,
        )

        if petsc_options_prefix == "":
            raise ValueError("PETSc options prefix cannot be empty.")

        self.solver.setOptionsPrefix(petsc_options_prefix)
        self.A.setOptionsPrefix(f"{petsc_options_prefix}A_")
        if self.P_mat is not None:
            self.P_mat.setOptionsPrefix(f"{petsc_options_prefix}P_mat_")
        self.b.setOptionsPrefix(f"{petsc_options_prefix}b_")
        self.x.setOptionsPrefix(f"{petsc_options_prefix}x_")

        # Set options for SNES only
        if petsc_options is not None:
            opts = PETSc.Options()  # type: ignore[attr-defined]
            opts.prefixPush(self.solver.getOptionsPrefix())

            for k, v in petsc_options.items():
                opts[k] = v

            self.solver.setFromOptions()

            # Tidy up global options
            for k in petsc_options.keys():
                del opts[k]

            opts.prefixPop()

        if self.P_mat is not None and kind == "nest":
            # Transfer nest IS on self.P_mat to PC of main KSP. This allows
            # fieldsplit preconditioning to be applied, if desired.
            nest_IS = self.P_mat.getNestISs()
            fieldsplit_IS = tuple(
                [
                    (f"{u.name + '_' if u.name != 'f' else ''}{i}", IS)
                    for i, (u, IS) in enumerate(zip(self.u, nest_IS[0]))
                ]
            )
            self.solver.getKSP().getPC().setFieldSplitIS(*fieldsplit_IS)

    def solve(self) -> _Function | Sequence[_Function]:
        """Solve the problem.

        This method updates the solution ``u`` function(s) stored in the
        problem instance.

        Note:
            The user is responsible for asserting convergence of the SNES
            solver e.g. ``assert problem.solver.getConvergedReason() > 0``.
            Alternatively, pass ``"snes_error_if_not_converged": True`` and
            ``"ksp_error_if_not_converged" : True`` in ``petsc_options`` to
            raise a ``PETScError`` on failure.

        Returns:
            The solution function(s).
        """
        # Copy current iterate into the work array.
        assign(self.u, self.x)

        # Solve problem
        self.solver.solve(None, self.x)
        dolfinx.la.petsc._ghost_update(
            self.x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD
        )  # type: ignore[attr-defined]

        # Copy solution back to function
        assign(self.x, self.u)

        return self.u

    def __del__(self):
        for obj in filter(
            lambda obj: obj is not None,
            (
                getattr(self, "_snes", None),
                getattr(self, "_A", None),
                getattr(self, "_b", None),
                getattr(self, "_x", None),
                getattr(self, "_P_mat", None),
            ),
        ):
            obj.destroy()

    @property
    def F(self) -> Form | Sequence[Form]:
        """The compiled residual."""
        return self._F

    @property
    def J(self) -> Form | Sequence[Sequence[Form]]:
        """The compiled Jacobian."""
        return self._J

    @property
    def preconditioner(self) -> Form | Sequence[Sequence[Form]] | None:
        """The compiled preconditioner."""
        return self._preconditioner

    @property
    def A(self) -> PETSc.Mat:  # type: ignore[name-defined]
        """Jacobian matrix."""
        return self._A

    @property
    def P_mat(self) -> PETSc.Mat | None:  # type: ignore[name-defined]
        """Preconditioner matrix."""
        return self._P_mat

    @property
    def b(self) -> PETSc.Vec:  # type: ignore[name-defined]
        """Residual vector."""
        return self._b

    @property
    def x(self) -> PETSc.Vec:  # type: ignore[name-defined]
        """Solution vector.

        Note:
            The vector does not share memory with the
            solution function(s) ``u``.
        """
        return self._x

    @property
    def solver(self) -> PETSc.SNES:  # type: ignore[name-defined]
        """The SNES solver."""
        return self._snes

    @property
    def u(self) -> _Function | Sequence[_Function]:
        """Solution function(s).

        Note:
            The function(s) do not share memory with the solution
            vector ``x``.
        """
        return self._u


@functools.singledispatch
def assign(u: _Function | Sequence[_Function], x: PETSc.Vec):  # type: ignore[name-defined]
    """Assign :class:`Function` degrees-of-freedom to a vector.

    Assigns degree-of-freedom values in ``u``, which is possibly a
    sequence of ``Function``s, to ``x``. When ``u`` is a sequence of
    ``Function``s, degrees-of-freedom for the ``Function``s in ``u`` are
    'stacked' and assigned to ``x``. See :func:`assign` for
    documentation on how stacked assignment is handled.

    Args:
        u: ``Function`` (s) to assign degree-of-freedom value from.
        x: Vector to assign degree-of-freedom values in ``u`` to.
    """
    if x.getType() == PETSc.Vec.Type().NEST:  # type: ignore[attr-defined]
        dolfinx.la.petsc.assign([v.x.array for v in u], x)
    else:
        if isinstance(u, Sequence):
            data0, data1 = [], []
            for v in u:
                bs = v.function_space.dofmap.bs
                n = v.function_space.dofmap.index_map.size_local
                data0.append(v.x.array[: bs * n])
                data1.append(v.x.array[bs * n :])
            dolfinx.la.petsc.assign(data0 + data1, x)
        else:
            dolfinx.la.petsc.assign(u.x.array, x)


@assign.register
def _(x: PETSc.Vec, u: _Function | Sequence[_Function]):  # type: ignore[name-defined]
    """Assign vector entries to :class:`Function` degrees-of-freedom.

    Assigns values in ``x`` to the degrees-of-freedom of ``u``, which is
    possibly a Sequence of ``Function``s. When ``u`` is a Sequence of
    ``Function``s, values in ``x`` are assigned block-wise to the
    ``Function``s. See :func:`assign` for documentation on how blocked
    assignment is handled.

    Args:
        x: Vector with values to assign values from.
        u: ``Function`` (s) to assign degree-of-freedom values to.
    """
    if x.getType() == PETSc.Vec.Type().NEST:  # type: ignore[attr-defined]
        dolfinx.la.petsc.assign(x, [v.x.array for v in u])
    else:
        if isinstance(u, Sequence):
            data0, data1 = [], []
            for v in u:
                bs = v.function_space.dofmap.bs
                n = v.function_space.dofmap.index_map.size_local
                data0.append(v.x.array[: bs * n])
                data1.append(v.x.array[bs * n :])
            dolfinx.la.petsc.assign(x, data0 + data1)
        else:
            dolfinx.la.petsc.assign(x, u.x.array)
