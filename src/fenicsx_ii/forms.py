from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from petsc4py import PETSc

import dolfinx
import ufl
from ufl.algorithms.analysis import has_type
from dolfinx.fem.forms import Form
from dolfinx.fem.forms import form as _dolfinx_create_form
from dolfinx.fem.function import Function as _Function

from .interpolation import create_interpolation_matrix
from .ufl_operations import Average, apply_replacer, get_replaced_argument_indices

__all__ = [
    "CompiledContribution",
    "CompiledFormBundle",
    "_create_form",
    "compile_form_bundle",
    "compile_with_replacer",
    "derivative_block",
]


@dataclass
class CompiledContribution:
    """Metadata for one replacer-generated compiled form contribution."""

    ufl_form: ufl.Form
    form: Form
    replacement_indices: tuple[int, ...]
    rank: int = 0
    row_operator: PETSc.Mat | None = None  # type: ignore[name-defined]
    col_operator: PETSc.Mat | None = None  # type: ignore[name-defined]
    test_space: dolfinx.fem.FunctionSpace | None = None
    trial_space: dolfinx.fem.FunctionSpace | None = None


@dataclass
class CompiledFormBundle:
    """Compiled representation for one (possibly replaced) UFL form."""

    direct: Form | Sequence[Form] | Sequence[Sequence[Form]] | None
    contributions: list[CompiledContribution]
    rank: int
    test_space: dolfinx.fem.FunctionSpace | None = None
    trial_space: dolfinx.fem.FunctionSpace | None = None


def _create_form(
    form: ufl.Form | Sequence[ufl.Form] | Sequence[Sequence[ufl.Form]],
    *,
    form_compiler_options: dict | None = None,
    jit_options: dict | None = None,
    entity_maps: Sequence[dolfinx.mesh.EntityMap] | None = None,
) -> Form | Sequence[Form] | Sequence[Sequence[Form]]:
    """Local wrapper around DOLFINx form compilation."""
    return _dolfinx_create_form(
        form,
        form_compiler_options=form_compiler_options,
        jit_options=jit_options,
        entity_maps=entity_maps,
    )


def _build_contribution(
    replaced_form: ufl.Form,
    compiled_form: Form,
) -> CompiledContribution:
    replacement_indices = tuple(get_replaced_argument_indices(replaced_form))
    args = replaced_form.arguments()
    rank = len(args)
    contribution = CompiledContribution(
        ufl_form=replaced_form,
        form=compiled_form,
        rank=rank,
        replacement_indices=replacement_indices,
    )

    if rank == 1:
        test_arg = args[0]
        if replacement_indices == ():
            contribution.test_space = test_arg.ufl_function_space()
        elif replacement_indices == (0,):
            contribution.test_space = test_arg.parent_space
            K, _, _ = create_interpolation_matrix(
                test_arg.parent_space,
                test_arg.ufl_function_space(),
                test_arg.restriction_operator,
                use_petsc=True,
            )
            Kt = K.copy()  # type: ignore[attr-defined]
            Kt.transpose()
            contribution.row_operator = Kt
        else:
            raise ValueError(
                f"Unsupported replacement pattern for rank-1 form: {replacement_indices}"
            )
    elif rank == 2:
        test_arg, trial_arg = args
        contribution.test_space = (
            test_arg.parent_space
            if 0 in replacement_indices
            else test_arg.ufl_function_space()
        )
        contribution.trial_space = (
            trial_arg.parent_space
            if 1 in replacement_indices
            else trial_arg.ufl_function_space()
        )

        if 0 in replacement_indices:
            P, _, _ = create_interpolation_matrix(
                test_arg.parent_space,
                test_arg.ufl_function_space(),
                test_arg.restriction_operator,
                use_petsc=True,
            )
            Pt = P.copy()  # type: ignore[attr-defined]
            Pt.transpose()
            contribution.row_operator = Pt

        if 1 in replacement_indices:
            K, _, _ = create_interpolation_matrix(
                trial_arg.parent_space,
                trial_arg.ufl_function_space(),
                trial_arg.restriction_operator,
                use_petsc=True,
            )
            contribution.col_operator = K

        if replacement_indices not in ((), (0,), (1,), (0, 1)):
            raise ValueError(
                f"Unsupported replacement pattern for rank-2 form: {replacement_indices}"
            )
    elif rank != 0:
        raise ValueError(f"Unsupported form rank {rank} for replacer compilation")

    return contribution


def compile_with_replacer(
    form: ufl.Form,
    *,
    form_compiler_options: dict | None = None,
    jit_options: dict | None = None,
    entity_maps: Sequence[dolfinx.mesh.EntityMap] | None = None,
) -> CompiledFormBundle:
    """Compile replacer-transformed form contributions."""
    target_rank = len(form.arguments())
    replaced_forms = apply_replacer(form)
    contributions: list[CompiledContribution] = []
    for replaced_form in replaced_forms:
        compiled = _create_form(
            replaced_form,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )
        if not isinstance(compiled, Form):
            raise RuntimeError("Expected a compiled scalar contribution form.")
        contribution = _build_contribution(replaced_form, compiled)
        # Derivative-generated transformed contributions can include lower-rank
        # zero terms. They do not contribute to matrix/vector assembly.
        if contribution.rank != target_rank:
            continue
        contributions.append(contribution)

    if len(contributions) == 0:
        raise RuntimeError(
            "No replacer contributions were generated for the target form rank."
        )

    args = form.arguments()
    rank = len(args)
    test_space = args[0].ufl_function_space() if rank >= 1 else None
    trial_space = args[1].ufl_function_space() if rank >= 2 else None
    return CompiledFormBundle(
        direct=None,
        contributions=contributions,
        rank=rank,
        test_space=test_space,
        trial_space=trial_space,
    )


def compile_form_bundle(
    form: (
        ufl.Form
        | Form
        | Sequence[ufl.Form]
        | Sequence[Form]
        | Sequence[Sequence[ufl.Form]]
        | Sequence[Sequence[Form]]
    ),
    *,
    form_compiler_options: dict | None = None,
    jit_options: dict | None = None,
    entity_maps: Sequence[dolfinx.mesh.EntityMap] | None = None,
) -> CompiledFormBundle:
    """Compile a form directly, with replacer fallback for scalar UFL forms."""
    if isinstance(form, Form):
        test_space = form.function_spaces[0] if form.rank >= 1 else None
        trial_space = form.function_spaces[1] if form.rank >= 2 else None
        return CompiledFormBundle(
            direct=form,
            contributions=[],
            rank=form.rank,
            test_space=test_space,
            trial_space=trial_space,
        )

    if isinstance(form, Sequence) and not isinstance(form, ufl.Form):
        direct = _create_form(
            form,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )
        return CompiledFormBundle(direct=direct, contributions=[], rank=0)

    if isinstance(form, ufl.Form):
        args = form.arguments()
        rank = len(args)
        test_space = args[0].ufl_function_space() if rank >= 1 else None
        trial_space = args[1].ufl_function_space() if rank >= 2 else None
        if has_type(form, Average):
            return compile_with_replacer(
                form,
                form_compiler_options=form_compiler_options,
                jit_options=jit_options,
                entity_maps=entity_maps,
            )
        try:
            direct = _create_form(
                form,
                form_compiler_options=form_compiler_options,
                jit_options=jit_options,
                entity_maps=entity_maps,
            )
            if not isinstance(direct, Form):
                raise RuntimeError("Expected compiled scalar form.")
            return CompiledFormBundle(
                direct=direct,
                contributions=[],
                rank=rank,
                test_space=test_space,
                trial_space=trial_space,
            )
        except Exception:
            try:
                return compile_with_replacer(
                    form,
                    form_compiler_options=form_compiler_options,
                    jit_options=jit_options,
                    entity_maps=entity_maps,
                )
            except Exception as replacer_exc:
                raise RuntimeError(
                    "Could not compile form directly or via replacer contributions."
                ) from replacer_exc

    raise RuntimeError("Unsupported form type for compilation.")


def derivative_block(
    F: ufl.Form | Sequence[ufl.Form],
    u: _Function | Sequence[_Function],
    du: ufl.Argument | Sequence[ufl.Argument] | None = None,
) -> ufl.Form | Sequence[Sequence[ufl.Form]]:
    """Return UFL derivative blocks (local copy of dolfinx behavior)."""
    if isinstance(F, ufl.Form):
        if not isinstance(u, _Function):
            raise ValueError("Must provide a single function when F is a UFL form")
        if du is None:
            du = ufl.TrialFunction(u.function_space)
        return ufl.derivative(F, u, du)

    assert all(isinstance(Fi, ufl.Form) for Fi in F), "F must be a sequence of UFL forms"
    assert isinstance(u, Sequence), "u must be a sequence when F is block-form"
    assert len(F) == len(u), "Number of forms and functions must be equal"
    if du is not None:
        assert isinstance(du, Sequence), "du must be a sequence for block derivative"
        assert len(F) == len(du), "Number of forms and du must be equal"
        du_seq = du
    else:
        du_seq = [ufl.TrialFunction(u_i.function_space) for u_i in u]
    return [
        [ufl.derivative(Fi, u_j, du_j) for u_j, du_j in zip(u, du_seq)]
        for Fi in F
    ]
