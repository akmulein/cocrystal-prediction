from __future__ import annotations

import traceback
import warnings
from pathlib import Path

from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.io.cif import CifParser
from pymatgen.io.xyz import XYZ


CURRENT_FILE_FOR_WARNINGS: str | None = None


def _custom_showwarning(message, category, filename, lineno, file=None, line=None):
    prefix = f'[WARNING in {CURRENT_FILE_FOR_WARNINGS}] '
    formatted = warnings.formatwarning(message, category, filename, lineno, line)
    print(prefix + formatted, end='')


warnings.showwarning = _custom_showwarning
warnings.filterwarnings('default')


def _extract_unique_molecules(structure) -> list[dict[str, object]]:
    structure_no_h = structure.copy()
    structure_no_h.remove_species(['H'])

    graph = StructureGraph.from_local_env_strategy(structure_no_h, CrystalNN())
    molecules = graph.get_subgraphs_as_molecules()

    unique_by_formula = {}
    for molecule in molecules:
        formula = molecule.composition.reduced_formula
        if formula not in unique_by_formula:
            unique_by_formula[formula] = molecule

    rows = []
    for mol_index, (formula, molecule) in enumerate(unique_by_formula.items()):
        rows.append(
            {
                'formula': formula,
                'mol_index': mol_index,
                'xyz': str(XYZ(molecule)),
            }
        )

    return rows


def process_one_file(cif_path: str | Path) -> dict[str, object]:
    global CURRENT_FILE_FOR_WARNINGS

    cif_path = Path(cif_path)
    CURRENT_FILE_FOR_WARNINGS = str(cif_path)

    last_error = None

    try:
        parser = CifParser(str(cif_path))
        structures = parser.parse_structures(primitive=False)
        if structures:
            return {
                'status': 'ok_default',
                'molecules': _extract_unique_molecules(structures[0]),
                'error_message': None,
            }
    except Exception:
        last_error = traceback.format_exc()

    try:
        parser = CifParser(str(cif_path), occupancy_tolerance=2.0)
        structures = parser.parse_structures(primitive=False)
        if structures:
            return {
                'status': 'ok_fallback',
                'molecules': _extract_unique_molecules(structures[0]),
                'error_message': last_error,
            }
    except Exception:
        last_error = traceback.format_exc()

    return {
        'status': 'fail',
        'molecules': [],
        'error_message': last_error,
    }
