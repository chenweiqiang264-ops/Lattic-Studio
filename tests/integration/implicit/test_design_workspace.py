"""Focused tests for isolated Design workspace state."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from lattice_studio.application.workspace import (
    AnalyticDesignDomain,
    ManagedDesignWorkspace as DesignWorkspace,
    MeshDesignDomain,
    PersistedDerivedMesh,
)
from lattice_studio.engine.implicit.primitives import ImplicitPrimitive


class DesignWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mesh = trimesh.creation.box(extents=(20.0, 12.0, 8.0))

    def test_documents_keep_field_objects_and_runtime_state_isolated(self) -> None:
        workspace = DesignWorkspace()
        first = workspace.create_mesh_document(self.mesh, "First")
        primitive = ImplicitPrimitive("field-1", "Driver", "sphere", radius_mm=3.0)
        first.upsert_field_primitive(primitive)
        first.runtime.implicit_results["G"] = object()

        second = workspace.duplicate_document(first.identifier, "Second")

        self.assertIs(second, workspace.active_document)
        self.assertEqual(set(second.field_primitives), {"field-1"})
        self.assertIsNot(second.field_primitives["field-1"], primitive)
        self.assertEqual(second.runtime.implicit_results, {})
        second.remove_field_primitive("field-1")
        self.assertEqual(set(first.field_primitives), {"field-1"})

    def test_archive_restore_and_empty_workspace_selection(self) -> None:
        workspace = DesignWorkspace()
        first = workspace.create_mesh_document(self.mesh, "First")
        second = workspace.create_mesh_document(self.mesh, "Second")

        workspace.archive_document(second.identifier)
        self.assertEqual(workspace.active_design_id, first.identifier)
        workspace.archive_document(first.identifier)
        self.assertTrue(workspace.is_empty)
        self.assertIsNone(workspace.active_document)

        restored = workspace.restore_document(second.identifier)
        self.assertEqual(workspace.active_design_id, second.identifier)
        self.assertFalse(restored.archived)
        self.assertIn(second.identifier, workspace.documents)

    def test_analytic_domain_uses_primitive_sdf_without_mesh_conversion(self) -> None:
        primitive = ImplicitPrimitive(
            "sphere-domain",
            "Sphere domain",
            "sphere",
            center_mm=(3.0, -2.0, 1.0),
            radius_mm=4.0,
        )
        domain = AnalyticDesignDomain(primitive, "Sphere design")
        values = domain.evaluate_points(
            np.asarray(((3.0, -2.0, 1.0), (7.0, -2.0, 1.0)), dtype=np.float32)
        )

        np.testing.assert_allclose(values, (-4.0, 0.0), atol=1.0e-6)
        self.assertTrue(domain.is_watertight)
        self.assertEqual(domain.frame_points.shape, (8, 3))
        self.assertEqual(domain.as_implicit_body().name, "Sphere design")

    def test_selected_result_invalidation_preserves_unrelated_outputs(self) -> None:
        workspace = DesignWorkspace()
        document = workspace.create_mesh_document(self.mesh, "First")
        revision = document.revision
        result_keys = ("D", "G", "Transition")
        for key in result_keys:
            document.runtime.implicit_results[key] = object()
            document.runtime.implicit_generation_results[key] = object()
            document.runtime.implicit_fields[key] = object()
            document.runtime.raw_results[key] = object()
            document.runtime.repaired_results[key] = object()
            document.runtime.results[key] = object()
            document.runtime.stl_reconstruction_results[key] = object()
            document.derived_meshes[key] = PersistedDerivedMesh(
                key,
                Path(f"assets/results/{key}.stl"),
                {"source": "test"},
            )
        document.runtime.renderer_cache["scene"] = object()

        document.invalidate_generated_results({"G", "Transition"})

        self.assertEqual(document.revision, revision + 1)
        for mapping in (
            document.runtime.implicit_results,
            document.runtime.implicit_generation_results,
            document.runtime.implicit_fields,
            document.runtime.raw_results,
            document.runtime.repaired_results,
            document.runtime.results,
            document.runtime.stl_reconstruction_results,
        ):
            self.assertEqual(set(mapping), {"D"})
        self.assertEqual(set(document.derived_meshes), {"D"})
        self.assertEqual(document.runtime.renderer_cache, {})

    def test_replacing_a_domain_invalidates_every_dependent_result(self) -> None:
        workspace = DesignWorkspace()
        document = workspace.create_mesh_document(self.mesh, "First")
        revision = document.revision
        document.runtime.domain_implicit_body = object()
        document.runtime.implicit_results["G"] = object()
        document.runtime.results["G"] = object()
        document.derived_meshes["G"] = PersistedDerivedMesh(
            "G",
            Path("assets/results/G.stl"),
            {"source": "test"},
        )
        repaired_mesh = trimesh.creation.icosphere(radius=5.0)

        document.replace_domain(
            MeshDesignDomain(repaired_mesh, "Repaired design")
        )

        self.assertEqual(document.revision, revision + 1)
        self.assertIsInstance(document.domain, MeshDesignDomain)
        self.assertEqual(document.domain.name, "Repaired design")
        self.assertEqual(document.runtime.implicit_results, {})
        self.assertEqual(document.runtime.results, {})
        self.assertIsNone(document.runtime.domain_implicit_body)
        self.assertEqual(document.derived_meshes, {})

    def test_field_scene_edit_persists_without_invalidating_lattices(self) -> None:
        workspace = DesignWorkspace()
        document = workspace.create_mesh_document(self.mesh, "First")
        document.runtime.implicit_results["G"] = object()
        revision = document.revision
        primitive = ImplicitPrimitive(
            "field-1",
            "Driver",
            "sphere",
            radius_mm=3.0,
        )

        changed = document.replace_field_object_scene(
            {primitive.identifier: primitive},
            {primitive.identifier: True},
        )

        self.assertTrue(changed)
        self.assertEqual(document.revision, revision + 1)
        self.assertEqual(document.field_primitives, {primitive.identifier: primitive})
        self.assertIn("G", document.runtime.implicit_results)
        self.assertFalse(
            document.replace_field_object_scene(
                {primitive.identifier: primitive},
                {primitive.identifier: True},
            )
        )

    def test_persistence_uses_managed_assets_and_excludes_runtime_caches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.stl"
            self.mesh.export(source)
            workspace = DesignWorkspace()
            document = workspace.create_mesh_document(self.mesh, "Saved design")
            workspace.package_mesh_domain_asset(document.identifier, source, root)
            document.settings["active_tpms_kind"] = "D"
            document.settings["tpms_parameter_states"] = {
                "G": {"cell_size_mm": [10.0, 10.0, 10.0]},
                "D": {"cell_size_mm": [12.0, 12.0, 12.0]},
            }
            document.runtime.renderer_cache["vtk"] = object()
            document.register_derived_mesh(
                PersistedDerivedMesh(
                    "G",
                    Path("assets/results/g.stl"),
                    {"source": "authoritative", "tolerance_mm": 0.25},
                )
            )

            manifest = root / "workspace.json"
            workspace.save(manifest)
            restored = DesignWorkspace.load(manifest)
            restored_document = restored.active_document

            self.assertIsNotNone(restored_document)
            assert restored_document is not None
            self.assertIsInstance(restored_document.domain, MeshDesignDomain)
            self.assertEqual(restored_document.runtime.renderer_cache, {})
            self.assertIn("G", restored_document.derived_meshes)
            self.assertEqual(restored_document.settings["active_tpms_kind"], "D")
            self.assertEqual(
                restored_document.settings["tpms_parameter_states"]["D"][
                    "cell_size_mm"
                ],
                [12.0, 12.0, 12.0],
            )
            self.assertFalse(restored.dirty)


if __name__ == "__main__":
    unittest.main()
